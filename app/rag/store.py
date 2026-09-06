"""
MongoDB Atlas Vector Search data layer.

This is the storage half of the RAG loop. `app/rag/embeddings.py` turns text
into vectors; this module writes those vectors into Atlas and, at query time,
asks Atlas "which stored vectors are nearest to this one?" via the
`$vectorSearch` aggregation stage.

Two collections use it, with the same code and the same index shape:

  * `documents` (config.DOCUMENTS_COLLECTION) — chunks of the files under
    `data/documents/`, written once by `app/rag/ingest.py`. Searched
    unfiltered: any chunk of any document is fair game for any question.
  * `memory` (config.MEMORY_COLLECTION) — one document per past conversation
    turn, written continuously while the agent runs. Searched *filtered* to a
    single `session_id`, because recalling another user's conversation into
    this user's context would be both wrong and a privacy leak. That filter
    requirement is the reason `vector_search()` takes a `pre_filter` argument
    and the reason `session_id` is declared as a `filter` field in the index
    definition below — see the pre-filter vs post-filter note on
    `_build_search_pipeline`.

--- Driver API, confirmed against the installed package (not from memory) ---
`pymongo` 4.18.0, source at `.venv/lib/python3.12/site-packages/pymongo/`.

  * `pymongo.AsyncMongoClient` EXISTS in this version. This matters: async
    support now lives inside the driver itself (`pymongo/asynchronous/`), and
    the separate `motor` package is no longer the way to do async MongoDB.
    Verified with `hasattr(pymongo, "AsyncMongoClient")` and
    `inspect.signature`, which reports the same constructor shape as the sync
    client: `AsyncMongoClient(host=None, port=None, document_class=None,
    tz_aware=None, connect=None, type_registry=None, **kwargs)`.
  * The collection methods used here are genuine coroutines — checked with
    `inspect.iscoroutinefunction` on
    `pymongo.asynchronous.collection.AsyncCollection`:
      - `insert_many(documents, ordered=True, ...)` -> coroutine, awaited.
      - `aggregate(pipeline, ...)` -> coroutine that RESOLVES TO an
        `AsyncCommandCursor`. Note the double step: you `await` the call to
        get the cursor, then iterate the cursor with `async for` (it defines
        `__aiter__`/`__anext__`) or drain it with `await cursor.to_list()`.
        Forgetting the first `await` hands you a coroutine object and
        `async for` fails confusingly.
      - `create_search_index(model, ...)` -> coroutine returning the index
        name (str). `create_search_indexes(models)` -> list[str].
        `list_search_indexes(name=None, ...)` -> coroutine resolving to an
        `AsyncCommandCursor` of index documents. `drop_search_index` and
        `update_search_index` also exist.
      - `pymongo.operations.SearchIndexModel(definition, name=None,
        type=None, **kwargs)` exists; its `type` parameter defaults to
        "search" and takes "vectorSearch" for a vector index (confirmed by
        reading its source and docstring — the `type` argument was added in
        pymongo 4.7, so this is version-sensitive).
      - `AsyncMongoClient.close()` is itself a coroutine here.

WHY THAT MATTERS SO MUCH HERE: this app runs ONE asyncio event loop serving
every live WebSocket voice session at once. A synchronous `MongoClient` call
made from inside a coroutine blocks that single thread for the entire network
round trip — meaning every other session's audio stops being serviced while
one session waits on a database. Database work is I/O-bound (waiting on a
socket), so the correct tool is a native async client that yields control back
to the loop for the whole flight time — not `asyncio.to_thread`, which would
burn a worker thread to accomplish the same yielding less efficiently. Same
reasoning as `app/audio/providers/sarvam.py` (async HTTP) and the opposite of
`faster_whisper_stt.transcribe()` (CPU-bound, so a thread genuinely helps).

Every public function here is therefore `async def` and must be awaited. From
a plain synchronous script with no loop running (e.g. `python -m
scripts.ingest_docs`), wrap the whole job in a single `asyncio.run(main())` —
not one `asyncio.run()` per call. `AsyncMongoClient` binds its internal
sockets to the loop that first uses it, and the cached singleton below would
be left holding connections belonging to a loop that has since been closed.

--- $vectorSearch syntax sources -------------------------------------------
Field names and index JSON below were taken from MongoDB's own docs, not
recalled:
  * https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/
  * https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-type/
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import OperationFailure, PyMongoError
from pymongo.operations import SearchIndexModel

from app import config

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when a MongoDB Atlas operation fails in a way we can name.

    Kept as its own exception type — exactly like
    `app.audio.providers.sarvam.SarvamAPIError` — so a caller further up the
    stack can catch *store* failures specifically (e.g. to answer without
    retrieved context rather than dropping the whole voice turn) without also
    swallowing unrelated RuntimeErrors from elsewhere in the app.

    This module is infrastructure, not an agent tool that reports failures
    back to an LLM as a string, so raising is the correct behaviour; deciding
    what a failure *means* belongs to the caller. Note that raw `PyMongoError`
    subclasses are deliberately allowed to propagate untouched in most places
    too — wrapping every one of them would only obscure genuinely useful
    driver diagnostics.
    """


# --- Document shape ----------------------------------------------------------
# The field every stored embedding lives under, in both collections. It is a
# module constant rather than a literal sprinkled through the file because it
# appears in three places that MUST agree: the documents `upsert_chunks()`
# writes, the `path` of the `$vectorSearch` stage, and the `path` of the
# vector field in the index definition. If those three ever drift apart, Atlas
# does not error — it just indexes nothing and returns nothing, which is a
# miserable bug to chase.
EMBEDDING_FIELD = "embedding"

# Fields declared as `filter` type in the index, so they can be used in
# `$vectorSearch`'s pre-filter (see `_build_search_pipeline`). A field that is
# NOT declared here cannot be pre-filtered on — Atlas rejects the query rather
# than silently ignoring the filter, which at least fails loudly.
#   * `session_id` — scopes `memory` recall to the current conversation.
#   * `source`     — lets `documents` be narrowed to one source file, useful
#                    for "according to the handbook..." style questions.
# Declaring a filter field the collection never uses is harmless (it just
# indexes nothing), so one shared definition serves both collections.
FILTER_FIELDS: tuple[str, ...] = ("session_id", "source")


# --- The index definition ----------------------------------------------------
# Built from config so it can never drift from what the rest of the code
# expects. This is the single source of truth for the index shape, used both
# by `ensure_vector_index()` (programmatic creation) and, when that isn't
# possible, printed verbatim for the user to paste into the Atlas UI.
#
# WHY `numDimensions` MUST EQUAL config.EMBEDDING_DIMENSION:
# an Atlas vector index is a physical data structure (an HNSW graph) built
# over fixed-width coordinate vectors. Its width is baked in at creation. If
# the index says 768 and you send a 3072-float query vector, Atlas rejects the
# query outright — and the error talks about the *query vector*, which sends
# people hunting through embeddings.py when the index is what's wrong. Worse
# is the reverse-ish case: swapping the embedding MODEL while keeping the same
# width. Nothing errors at all, because a vector's dimensionality says nothing
# about which vector *space* it belongs to. Coordinates from model A are
# meaningless distances away from coordinates from model B, so retrieval
# silently returns plausible-looking nonsense. Changing the embedding model
# therefore invalidates the entire index AND every stored vector: you must
# re-embed the whole corpus and rebuild the index. There is no partial
# migration — a collection holding vectors from two models is broken for
# every query, not just for the new documents.
#
# WHY `similarity` IS "dotProduct":
# the similarity metric has to match how the vectors were actually produced.
# `embeddings.py::_normalize()` scales every vector it returns to unit length
# (L2 norm == 1) precisely so this choice is safe. For unit vectors, cosine
# similarity and the dot product are *mathematically identical*:
#     cos(a, b) = (a · b) / (‖a‖‖b‖)   and   ‖a‖ = ‖b‖ = 1   =>   cos = a · b
# so "cosine" here would compute the same ranking — it would just re-derive
# and divide by norms it already knows are 1.0, on every comparison. Picking
# `dotProduct` skips that redundant work.
#
# The corollary is the trap: this is only true *because* we normalize. If
# `_normalize()` were ever removed, `dotProduct` would start ranking by
# "similar AND long" rather than "similar", quietly favouring whichever chunks
# happen to have large-magnitude embeddings. The two files are coupled; the
# comment in each points at the other. (`euclidean` is the third option and is
# also rank-equivalent to the other two on unit vectors, since
# ‖a - b‖² = 2 - 2(a · b) — a monotonically decreasing function of the dot
# product. It is the right choice only when magnitude carries real meaning.)
VECTOR_INDEX_DEFINITION: dict[str, Any] = {
    "fields": [
        {
            "type": "vector",
            "path": EMBEDDING_FIELD,
            "numDimensions": config.EMBEDDING_DIMENSION,
            "similarity": "dotProduct",
        },
        *({"type": "filter", "path": field} for field in FILTER_FIELDS),
    ]
}


# --- ANN tuning --------------------------------------------------------------
# `numCandidates` vs `limit` — the single most important knob in this file,
# and the one interviewers ask about.
#
# Atlas Vector Search is APPROXIMATE nearest neighbour (ANN), not exhaustive.
# It does not compare your query vector against all N stored vectors; at that
# scale it couldn't and stay fast. Instead it walks an HNSW graph — a layered
# "small world" structure where each vector links to a handful of near
# neighbours, plus a few long-range links in upper layers. A search enters at
# the sparse top layer, greedily hops toward the query, descends a layer,
# hops again, and so on. That's roughly O(log N) hops instead of O(N)
# comparisons.
#
# Greedy graph traversal can get stuck in a local minimum: it reaches a vertex
# whose neighbours are all worse than itself, and stops — even though a truly
# nearer vector exists in a part of the graph it never touched. The defence is
# to keep a priority queue of the best candidates seen so far and keep
# exploring from all of them, not just the current best.
#
#   * `numCandidates` = how big that queue is. How many candidates the search
#     is willing to consider before it stops looking.
#   * `limit`         = how many of those survivors it actually returns.
#
# So the trade-off is RECALL vs LATENCY, and it is a genuine dial, not a
# default to accept blindly:
#   - numCandidates too low  -> fast, but the search gives up early and can
#     miss the true nearest chunk entirely. This failure is INVISIBLE: you
#     still get exactly `limit` results, all of them plausible, just not the
#     best ones. Nothing logs, nothing errors, the bot is merely worse.
#   - numCandidates too high -> better recall (asymptotically approaching an
#     exhaustive search) but more graph traversal, more distance computations,
#     more latency. On a live voice call that latency is heard.
#
# MongoDB's documented rule of thumb is to set numCandidates at least ~20x
# `limit` (commonly quoted as 10-20x), which reportedly lands around 90-95%
# recall overlap with an exact search while staying much faster. With
# config.RETRIEVAL_TOP_K = 4 that's 80 candidates — trivial work for a corpus
# this size, so we take the upper end of the range and buy the recall.
#
# The floor exists because the multiplier alone is silly for tiny limits:
# 20 x limit=1 is 20 candidates, which is a very narrow beam. Atlas also
# requires numCandidates >= limit; the floor makes violating that impossible.
#
# (There is an escape hatch for correctness-critical work: `"exact": true` on
# the stage runs ENN — an exhaustive scan, perfect recall, no `numCandidates`
# at all. Right for evaluating retrieval quality offline; wrong for a live
# voice turn.)
NUM_CANDIDATES_MULTIPLIER = 20
MIN_NUM_CANDIDATES = 100


# --- Lazy, one-time client construction --------------------------------------
# Same pattern, and the same reasoning, as `embeddings.py::_get_client()`.
#
# Not built at import time, because `config.MONGODB_URI` may legitimately be
# empty: importing `app.rag.store` to unit-test `_build_search_pipeline`, or
# to let an editor index the package, must not require a database. Config
# stays lenient; the module that needs the secret demands it at the point of
# use via `config.require(...)`, so the failure names the missing variable
# instead of surfacing as an opaque connection error from inside the driver.
#
# Built exactly once, because an `AsyncMongoClient` is not a connection — it
# is a connection POOL plus a background topology monitor that keeps track of
# which replica-set members are healthy. Constructing one per query would pay
# TCP + TLS + auth handshakes every time and leak monitor tasks. It is
# designed to be created once and shared for the life of the process.
_client: AsyncMongoClient | None = None

# A threading.Lock, not an asyncio.Lock — matching embeddings.py. The public
# functions are coroutines, but an ingestion script may drive them from a
# worker thread, and nothing stops two OS threads from entering `_get_client()`
# simultaneously. Only a threading.Lock actually blocks the second thread.
# Double-checked locking keeps the common path (client already built) free of
# any lock acquisition at all.
_client_lock = threading.Lock()


def _get_client() -> AsyncMongoClient:
    """Build the Atlas client exactly once, on first real use.

    Deliberately NOT a coroutine: constructing an `AsyncMongoClient` performs
    no I/O. The driver connects lazily in the background, so this is a cheap
    synchronous call and making it `async` would only add ceremony.

    Raises:
        RuntimeError: from `config.require` if MONGODB_URI is unset.
    """
    global _client

    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            uri = config.require(config.MONGODB_URI, "MONGODB_URI")
            _client = AsyncMongoClient(uri)
            logger.info(
                "MongoDB Atlas client constructed (db=%s, index=%s, dimension=%d).",
                config.MONGODB_DB_NAME,
                config.VECTOR_INDEX_NAME,
                config.EMBEDDING_DIMENSION,
            )

    return _client


def _get_collection(name: str) -> AsyncCollection:
    """Resolve a collection handle by name.

    Also synchronous and also free: `client[db][coll]` is pure local
    bookkeeping — MongoDB creates a database or collection on first write, so
    naming one that doesn't exist yet costs nothing and touches no network.
    """
    return _get_client()[config.MONGODB_DB_NAME][name]


async def close_client() -> None:
    """Close the shared client and its connection pool.

    Call from a FastAPI shutdown hook, or at the end of an `asyncio.run(...)`
    ingestion job. Skipping it leaks sockets and leaves the topology-monitor
    task running, which on shutdown produces the classic "Task was destroyed
    but it is pending" noise. `AsyncMongoClient.close()` is itself a coroutine
    in pymongo 4.18 (verified with `inspect.iscoroutinefunction`), so it must
    be awaited.
    """
    global _client

    with _client_lock:
        client, _client = _client, None

    if client is not None:
        await client.close()
        logger.info("MongoDB Atlas client closed.")


# --- Writing -----------------------------------------------------------------
async def upsert_chunks(collection_name: str, records: list[dict]) -> int:
    """Bulk-write chunk documents (each already carrying its embedding).

    Each record is a plain dict that must already contain `EMBEDDING_FIELD`
    (a list of `config.EMBEDDING_DIMENSION` floats from
    `embeddings.embed_documents`) alongside whatever metadata that collection
    uses — `text` and `source` for `documents`, `text`, `role`, `session_id`
    and a timestamp for `memory`.

    WHY ONE BULK CALL AND NOT A LOOP OF `insert_one`:
    the cost of a write here is dominated by the network round trip to Atlas,
    not by the insert itself. Atlas is a hosted cluster, often in a different
    region — call it ~50ms per round trip. Ingesting a 400-chunk PDF one
    insert at a time is 400 sequential round trips, i.e. ~20 seconds spent
    almost entirely waiting on the wire, versus a single batched round trip
    (the driver splits internally if the batch exceeds the 48MB/100k-op
    server limits). It is also atomic-ish in the useful sense: one call gives
    you one result to check and one failure to retry, instead of 400 partial
    successes to reconcile.
    (`insert_many` is the right bulk primitive here rather than `bulk_write`:
    these are always fresh chunks. Genuine upsert-by-key would need
    `bulk_write` with `ReplaceOne(..., upsert=True)`; the ingest path deletes
    a source's old chunks and re-inserts instead, so plain inserts suffice.
    The name reflects the caller's intent, not a Mongo upsert operator.)

    `ordered=False` lets the server keep going past a failing document instead
    of aborting at the first one — one malformed chunk shouldn't cost you the
    other 399 — and lets it parallelize the batch internally.

    Args:
        collection_name: `config.DOCUMENTS_COLLECTION` or
            `config.MEMORY_COLLECTION`.
        records: The documents to insert. Must be non-empty, and each must
            carry a non-empty embedding of the configured width.

    Returns:
        How many documents were actually written.

    Raises:
        ValueError: if `records` is empty or a record is missing/malformed
            its embedding.
        RuntimeError: if MONGODB_URI is unset.
        VectorStoreError: if the bulk write fails.
    """
    if not records:
        raise ValueError("upsert_chunks() received an empty list — nothing to write.")

    # Validate before touching the network. A document written without an
    # embedding, or with the wrong width, is not a write error — Mongo is
    # schemaless and accepts it happily. It simply never matches any vector
    # search, forever, invisibly. Catching it here names the offending index
    # instead of leaving a silent hole in the corpus.
    for position, record in enumerate(records):
        vector = record.get(EMBEDDING_FIELD)
        if not isinstance(vector, list) or not vector:
            raise ValueError(
                f"upsert_chunks() record at index {position} has no usable "
                f"{EMBEDDING_FIELD!r} field. Every record must carry its embedding; "
                "see app/rag/embeddings.py::embed_documents."
            )
        if len(vector) != config.EMBEDDING_DIMENSION:
            raise ValueError(
                f"upsert_chunks() record at index {position} has a "
                f"{len(vector)}-dimensional embedding, but config.EMBEDDING_DIMENSION "
                f"is {config.EMBEDDING_DIMENSION}. Atlas would index this document "
                "but never return it."
            )

    collection = _get_collection(collection_name)

    try:
        result = await collection.insert_many(records, ordered=False)
    except PyMongoError as exc:
        # Wrapped (unlike most driver errors in this file) because a bulk
        # write failure mid-ingest needs a message that says WHICH collection
        # and HOW MANY documents were in flight — `raise ... from exc` keeps
        # the driver's own diagnostics attached underneath.
        raise VectorStoreError(
            f"Bulk insert of {len(records)} documents into "
            f"{config.MONGODB_DB_NAME}.{collection_name} failed: {exc}"
        ) from exc

    written = len(result.inserted_ids)
    logger.info(
        "Inserted %d/%d documents into %s.%s.",
        written,
        len(records),
        config.MONGODB_DB_NAME,
        collection_name,
    )
    return written


# --- Query pipeline ----------------------------------------------------------
def _build_search_pipeline(
    query_vector: list[float],
    limit: int,
    pre_filter: dict | None = None,
) -> list[dict]:
    """Build the `$vectorSearch` aggregation pipeline. Pure — touches no network.

    Factored out of `vector_search()` on purpose: this dict is the part most
    likely to be silently wrong (a misspelled `numCandidates`, a `filter`
    placed in the wrong stage) and the part that cannot otherwise be tested
    without a live Atlas cluster. As a pure function it can be asserted
    against directly in a unit test.

    Field names per
    https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/ —
    `index`, `path`, `queryVector`, `numCandidates`, `limit`, `filter`.
    `$vectorSearch` must be the FIRST stage of the pipeline; it is not a stage
    you can slot in after a `$match`.

    --- PRE-FILTERING vs POST-FILTERING (the reason `filter` lives inside the
    `$vectorSearch` stage and not in a following `$match`) ---

    It is tempting to write:

        [{"$vectorSearch": {... limit: 4 ...}},
         {"$match": {"session_id": "abc"}}]          # WRONG

    That is POST-filtering, and it is broken in a specific, quiet way. The
    vector search runs first over the whole collection and commits to its top
    4 documents. Only then does `$match` throw away the ones from other
    sessions. If this session's turns happen to be less similar to the query
    than some other session's turns — which, in a `memory` collection shared
    by every conversation the app has ever had, is the NORMAL case — you get
    2 results, or 1, or zero. You asked for 4 and the vector search genuinely
    found 4; they were just discarded after the fact, and the 4 correct
    in-session turns were never candidates. The bug looks like "memory recall
    randomly stops working as the app gets more users", which is exactly the
    kind of thing that ships.

    Passing `filter` INSIDE the `$vectorSearch` stage is PRE-filtering: the
    predicate is applied *during* the ANN traversal, so the graph search only
    ever considers documents from this session and still returns a full
    `limit` of them. Correct results, and cheaper too — a smaller candidate
    space is faster to search.

    The catch, and it's a real one: a field can only be pre-filtered on if it
    was declared with `"type": "filter"` in the index definition (see
    `VECTOR_INDEX_DEFINITION` / `FILTER_FIELDS`). An undeclared field makes
    Atlas reject the query. So the filter capability is decided at index
    creation time, not query time — adding a new filterable field later means
    editing the index, not just the query.

    A `$match` after `$vectorSearch` is not universally wrong, incidentally —
    it's the right tool for a predicate you deliberately want applied to the
    top-k *as found* (e.g. dropping results below a score threshold, since
    score doesn't exist until after the search). It is only wrong as a
    substitute for scoping.

    --- Score projection ---
    The similarity score is metadata, not a stored field: it only exists as a
    product of this particular search, so it must be pulled out explicitly
    with `{"$meta": "vectorSearchScore"}`. It runs 0..1 (higher = more
    similar) and is what lets a caller apply a relevance floor rather than
    stuffing four barely-related chunks into the LLM prompt.

    `_id` is excluded because it is a `bson.ObjectId`, which is not
    JSON-serializable — leaving it in makes the retrieved documents blow up
    the moment they're sent over the WebSocket or logged as JSON.

    Args:
        query_vector: The embedded question (from `embeddings.embed_query`).
        limit: How many documents to return.
        pre_filter: Optional MongoDB filter document applied during the ANN
            search, e.g. `{"session_id": "abc123"}`.

    Returns:
        The aggregation pipeline, as a list of stage dicts.
    """
    search_stage: dict[str, Any] = {
        "index": config.VECTOR_INDEX_NAME,
        "path": EMBEDDING_FIELD,
        "queryVector": query_vector,
        "numCandidates": max(limit * NUM_CANDIDATES_MULTIPLIER, MIN_NUM_CANDIDATES),
        "limit": limit,
    }

    # Only add `filter` when there is one. An empty `{}` is not equivalent to
    # omitting the key — it makes Atlas do pre-filter work for a predicate
    # that matches everything.
    if pre_filter:
        search_stage["filter"] = pre_filter

    return [
        {"$vectorSearch": search_stage},
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "source": 1,
                "session_id": 1,
                "role": 1,
                "timestamp": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]


async def vector_search(
    collection_name: str,
    query_vector: list[float],
    limit: int = config.RETRIEVAL_TOP_K,
    pre_filter: dict | None = None,
) -> list[dict]:
    """Find the documents whose embeddings are nearest to `query_vector`.

    The read half of RAG, and the one that sits on the live voice latency path
    (user stops speaking -> transcribe -> embed_query -> THIS -> LLM), which
    is the concrete reason it is a native coroutine rather than a blocking
    call: while this awaits Atlas, every other WebSocket session on the loop
    keeps being served.

    Args:
        collection_name: `config.DOCUMENTS_COLLECTION` or
            `config.MEMORY_COLLECTION`.
        query_vector: The embedded question. Must be
            `config.EMBEDDING_DIMENSION` long and should come from
            `embeddings.embed_query` — NOT `embed_document`; see the
            asymmetric-embedding note in embeddings.py for why that
            distinction silently costs recall.
        limit: How many documents to return. Defaults to
            `config.RETRIEVAL_TOP_K`.
        pre_filter: Optional filter applied *during* the search. For the
            `memory` collection pass `{"session_id": <id>}`; the field must be
            declared in `FILTER_FIELDS`.

    Returns:
        The matched documents, nearest first, each with an added `score`
        (0..1, higher is more similar). May be shorter than `limit` — or
        empty — if the collection holds fewer documents than that, or if the
        index is still building.

    Raises:
        ValueError: if the query vector is empty or the wrong width.
        RuntimeError: if MONGODB_URI is unset.
        VectorStoreError: if the index is missing, or the aggregation fails.
    """
    if not query_vector:
        raise ValueError("vector_search() received an empty query vector.")
    if len(query_vector) != config.EMBEDDING_DIMENSION:
        # Caught here it names both numbers. Left to Atlas, the error mentions
        # only the query vector, sending you to read embeddings.py when the
        # index may be what's stale.
        raise ValueError(
            f"vector_search() received a {len(query_vector)}-dimensional query "
            f"vector, but config.EMBEDDING_DIMENSION is "
            f"{config.EMBEDDING_DIMENSION}. Atlas will reject this."
        )

    collection = _get_collection(collection_name)
    pipeline = _build_search_pipeline(query_vector, limit, pre_filter)

    try:
        # TWO awaits, by design (see the module docstring): `aggregate(...)` is
        # a coroutine that resolves to an AsyncCommandCursor, and `to_list()`
        # is a second coroutine that drains it. `to_list()` rather than an
        # `async for` loop because top-k is small and bounded — there is no
        # streaming benefit, and a single await is easier to reason about.
        cursor = await collection.aggregate(pipeline)
        results = await cursor.to_list()
    except OperationFailure as exc:
        # The overwhelmingly common cause is a missing/misnamed/still-building
        # index, or a pre-filter on a field that wasn't declared as
        # "type": "filter". Say so, because the raw server message rarely
        # makes the fix obvious.
        raise VectorStoreError(
            f"$vectorSearch on {config.MONGODB_DB_NAME}.{collection_name} failed: {exc}. "
            f"Check that the Atlas Vector Search index {config.VECTOR_INDEX_NAME!r} "
            f"exists on that collection, is in the READY state, has numDimensions="
            f"{config.EMBEDDING_DIMENSION}, and declares every pre-filtered field "
            f"with \"type\": \"filter\" (see FILTER_FIELDS)."
        ) from exc

    logger.debug(
        "vector_search on %s returned %d/%d results (pre_filter=%s).",
        collection_name,
        len(results),
        limit,
        pre_filter,
    )
    return results


# --- Index management --------------------------------------------------------
async def ensure_vector_index(collection_name: str) -> bool:
    """Create the vector search index on `collection_name` if it's missing.

    Best-effort by design. Programmatic search-index management
    (`create_search_index`, added in pymongo 4.5 with `type="vectorSearch"`
    support in 4.7) requires an Atlas cluster on MongoDB 7.0+, and free/shared
    tiers have historically restricted it. So a failure here is NOT fatal: the
    app can still run perfectly against an index created by hand in the Atlas
    UI, and refusing to boot because we couldn't create an index that may
    already exist would be actively unhelpful.

    This is the one place in this module that swallows an exception, and it's
    a deliberate, narrow exception to the no-swallowing policy the rest of the
    codebase follows: the failure is logged at WARNING with the exact JSON to
    paste into the UI, and if the index really is absent the next
    `vector_search()` fails loudly with a message pointing right back here.

    Args:
        collection_name: The collection to index.

    Returns:
        True if this call created the index; False if it already existed or
        could not be created programmatically.

    Raises:
        RuntimeError: if MONGODB_URI is unset. (Connection problems are a
            configuration error worth failing on, unlike a tier restriction.)
    """
    collection = _get_collection(collection_name)

    # Ask for the one index by name rather than listing them all and
    # filtering client-side.
    try:
        cursor = await collection.list_search_indexes(config.VECTOR_INDEX_NAME)
        existing = await cursor.to_list()
    except PyMongoError as exc:
        # Some deployments reject even *listing* search indexes. Treat that
        # exactly like "can't create" rather than crashing startup.
        _log_manual_index_instructions(collection_name, f"could not list search indexes: {exc}")
        return False

    if existing:
        logger.info(
            "Vector index %r already exists on %s.%s (status=%s).",
            config.VECTOR_INDEX_NAME,
            config.MONGODB_DB_NAME,
            collection_name,
            existing[0].get("status", "unknown"),
        )
        return False

    try:
        await collection.create_search_index(
            SearchIndexModel(
                definition=VECTOR_INDEX_DEFINITION,
                name=config.VECTOR_INDEX_NAME,
                # Without type="vectorSearch" this silently creates an Atlas
                # *Search* (full-text) index instead — the parameter defaults
                # to "search". That index would then not serve $vectorSearch
                # at all, while looking present in the UI.
                type="vectorSearch",
            )
        )
    except PyMongoError as exc:
        _log_manual_index_instructions(collection_name, str(exc))
        return False

    # Creation is ASYNCHRONOUS on Atlas's side: the call returns as soon as
    # the build is accepted, and the index moves PENDING -> BUILDING -> READY
    # over the following seconds. Queries run against a not-yet-READY index
    # return no results rather than an error — the classic "I just ingested
    # everything and search returns nothing" panic. We deliberately don't
    # block here (that would stall startup); we just say so.
    logger.info(
        "Created vector index %r on %s.%s. Atlas builds it asynchronously — "
        "searches may return no results until its status reaches READY.",
        config.VECTOR_INDEX_NAME,
        config.MONGODB_DB_NAME,
        collection_name,
    )
    return True


def _log_manual_index_instructions(collection_name: str, reason: str) -> None:
    """Log the exact index JSON to create by hand in the Atlas UI.

    Rendered from `VECTOR_INDEX_DEFINITION` rather than written out in a
    docstring, so what the user pastes into Atlas is by construction the same
    definition this code expects — including `numDimensions`, which follows
    `config.EMBEDDING_DIMENSION` automatically.
    """
    import json  # local: only needed on this cold, rare path.

    logger.warning(
        "Could not create the Atlas Vector Search index programmatically (%s).\n"
        "This is usually a cluster-tier restriction, not a bug, and is not fatal.\n"
        "Create it by hand instead: Atlas UI -> your cluster -> Atlas Search -> "
        "Create Search Index -> JSON Editor -> Vector Search,\n"
        "  database:   %s\n"
        "  collection: %s\n"
        "  index name: %s\n"
        "  definition:\n%s",
        reason,
        config.MONGODB_DB_NAME,
        collection_name,
        config.VECTOR_INDEX_NAME,
        json.dumps(VECTOR_INDEX_DEFINITION, indent=2),
    )
