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


EMBEDDING_FIELD = "embedding"

FILTER_FIELDS: tuple[str, ...] = ("session_id", "source")


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


NUM_CANDIDATES_MULTIPLIER = 20
MIN_NUM_CANDIDATES = 100


_client: AsyncMongoClient | None = None

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


async def upsert_chunks(collection_name: str, records: list[dict]) -> int:
    """Bulk-write chunk documents (each already carrying its embedding).

    Each record is a plain dict that must already contain `EMBEDDING_FIELD`
    (a list of `config.EMBEDDING_DIMENSION` floats from
    `embeddings.embed_documents`) alongside whatever metadata that collection
    uses — `text`, `source`, `chunk_index` and `timestamp` for `documents`;
    `text`, `user_text`, `assistant_text`, `session_id` and `timestamp` for
    `memory` (see the projected field list on `_build_search_pipeline`).

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


async def delete_chunks_by_source(collection_name: str, source: str) -> int:
    """Remove every chunk that was ingested from one source file.

    Exists to make re-ingestion idempotent. `app/rag/ingest.py` calls this
    immediately before inserting a file's new chunks, so running the ingest
    CLI twice replaces a document's chunks rather than doubling them.
    Duplicates are not merely wasted storage: identical chunks compete for the
    same top-k slots, so the LLM receives one paragraph four times instead of
    four different ones, and retrieval quality degrades with every re-run.

    Delete-then-insert rather than a keyed upsert, because re-chunking a file
    after a `CHUNK_SIZE` change renumbers everything — a file that produced 40
    chunks and now produces 30 would leave chunks 30-39 of the previous run
    orphaned in the collection, matching queries with text that no longer
    reflects the source.

    Uses `delete_many` (a genuine coroutine on `AsyncCollection`, verified
    with `inspect.iscoroutinefunction`) so the whole removal is one round trip
    to Atlas, for the same reason `upsert_chunks` batches its writes.

    Deleting nothing is a normal, successful outcome — it is what the first
    ingestion of a new file does — so a zero return is not an error and this
    does not raise for it.

    Args:
        collection_name: Usually `config.DOCUMENTS_COLLECTION`.
        source: The exact value stored in the `source` field, i.e. the
            filename as written by `ingest.py`.

    Returns:
        How many documents were deleted.

    Raises:
        ValueError: if `source` is empty. An empty filter would match and
            delete the ENTIRE collection, which is never what a caller that
            forgot to pass a filename meant.
        RuntimeError: if MONGODB_URI is unset.
        VectorStoreError: if the delete fails.
    """
    if not source:
        raise ValueError(
            "delete_chunks_by_source() requires a non-empty source — an empty "
            "value would match every document in the collection."
        )

    collection = _get_collection(collection_name)

    try:
        result = await collection.delete_many({"source": source})
    except PyMongoError as exc:
        raise VectorStoreError(
            f"Deleting existing chunks for source {source!r} from "
            f"{config.MONGODB_DB_NAME}.{collection_name} failed: {exc}"
        ) from exc

    if result.deleted_count:
        logger.info(
            "Deleted %d existing chunks for source %r from %s.%s.",
            result.deleted_count,
            source,
            config.MONGODB_DB_NAME,
            collection_name,
        )
    return result.deleted_count


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

    --- The projected field list ---
    One whitelist serves both collections, because both are read back through
    this one function. `$project` simply omits a listed field that a document
    does not have, so a `documents` hit comes back without `session_id` and a
    `memory` hit without `source`, with no error either way.

      * `text`         — both. The chunk / the exchange; what reaches the LLM.
      * `source`       — `documents`. The filename, used for citation.
      * `chunk_index`  — `documents`. Position in the file, for ordering and
                         provenance.
      * `session_id`   — `memory`. The scope the pre-filter matched on.
      * `user_text`,
        `assistant_text` — `memory`. The halves of the exchange, kept apart
                         from the combined `text` so a caller can display or
                         replay them without re-parsing the combined string.
      * `timestamp`    — both, deliberately one name for one concept
                         (`ingest.py` and `memory.py` both write it), so this
                         list does not have to carry two spellings of "when".

    `role` is NOT projected because it is not stored: `memory.py` writes one
    document per completed exchange rather than one per message — an
    assistant answer retrieved without its question is often uninterpretable
    ("Yes, about three weeks") — and a document containing both halves has no
    single role. The field list here matches `ingest._build_records` and
    `memory.add_turn` exactly; if either schema changes, this changes with it,
    or retrieval starts dropping fields without saying so.

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

    if pre_filter:
        search_stage["filter"] = pre_filter

    return [
        {"$vectorSearch": search_stage},
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "source": 1,
                "chunk_index": 1,
                "session_id": 1,
                "user_text": 1,
                "assistant_text": 1,
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
        raise ValueError(
            f"vector_search() received a {len(query_vector)}-dimensional query "
            f"vector, but config.EMBEDDING_DIMENSION is "
            f"{config.EMBEDDING_DIMENSION}. Atlas will reject this."
        )

    collection = _get_collection(collection_name)
    pipeline = _build_search_pipeline(query_vector, limit, pre_filter)

    try:
        cursor = await collection.aggregate(pipeline)
        results = await cursor.to_list()
    except OperationFailure as exc:
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

    try:
        cursor = await collection.list_search_indexes(config.VECTOR_INDEX_NAME)
        existing = await cursor.to_list()
    except PyMongoError as exc:
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
                type="vectorSearch",
            )
        )
    except PyMongoError as exc:
        _log_manual_index_instructions(collection_name, str(exc))
        return False

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
    import json

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
