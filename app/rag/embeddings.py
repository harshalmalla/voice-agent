"""
Text -> vector embeddings, backed by the Gemini embedding API.

This is the layer that turns human language into the numbers MongoDB Atlas
Vector Search can actually compare. Everything downstream depends on it:
`app/rag/ingest.py` embeds document chunks on the way *in*, and
`app/rag/store.py` embeds the user's question on the way *out* so Atlas can
run `$vectorSearch` and hand back the nearest chunks.

API confirmed against the actually-installed package (not from memory):
`google-genai` 2.22.0, source at
`.venv/lib/python3.12/site-packages/google/genai/` (`client.py`, `models.py`,
`types.py`, `_transformers.py`).

  * The SDK is **client-based**, not module-global. You build
    `genai.Client(*, api_key=None, vertexai=None, credentials=None,
    project=None, location=None, http_options=None, debug_config=None,
    enterprise=None)` — all keyword-only — and hang every call off that
    instance. There is no `configure()` to call any more; the "configuration"
    IS the client object, which is why the lazy singleton below caches a
    `Client` rather than flipping a boolean.
  * Sync call:  `client.models.embed_content(*, model, contents, config=None)`.
    Async call: `client.aio.models.embed_content(...)` — same keyword-only
    signature, and it is a genuine coroutine
    (`inspect.iscoroutinefunction(...)` is True) driven by an
    `httpx.AsyncClient` under the hood (see `_api_client.AsyncHttpxClient`),
    not a thread-pool wrapper. That is what makes `async def` here honest.
  * Options are no longer loose kwargs — they live in
    `google.genai.types.EmbedContentConfig`, a pydantic model whose fields are:
    `task_type`, `title`, `output_dimensionality`, `mime_type`,
    `auto_truncate`, `document_ocr`, `audio_track_extraction`, `http_options`.
    Note the snake_case spelling of `task_type` and `output_dimensionality`
    (they are serialized to `taskType` / `outputDimensionality` on the wire).
  * `task_type` is typed `Optional[str]` and is passed through to the request
    **verbatim** — the SDK does no enum coercion and no case-folding.
    This is a real behaviour change from `google-generativeai`, which accepted
    the enum, its int, or a case-insensitive string. Here the string must
    match the REST `TaskType` enum EXACTLY, in UPPER_SNAKE_CASE. Verified by
    building the request dict offline and reading the serialized value back:
    `"retrieval_document"` is transmitted as-is and is not the enum name, so
    lowercase is now a server-side error rather than a silent success. The
    valid values (https://ai.google.dev/api/embeddings) are:
    TASK_TYPE_UNSPECIFIED, RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT,
    SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING, QUESTION_ANSWERING,
    FACT_VERIFICATION, CODE_RETRIEVAL_QUERY.
  * `contents` is **plural** and accepts either a single item or a list. For
    the Gemini Developer API the SDK always posts to `{model}:batchEmbedContents`
    with a `requests[]` array, so a lone string is normalized into a
    one-element batch rather than taking a different code path.
  * The return value is an **object, not a dict**:
    `types.EmbedContentResponse` with fields `embeddings`, `metadata` and
    `sdk_http_response`. `response.embeddings` is a
    `list[types.ContentEmbedding]`, and each `ContentEmbedding` carries the
    floats on `.values` (a `list[float]`). Crucially the shape is now
    **uniform**: one input yields a one-element list, N inputs yield an
    N-element list. The old SDK's "same `"embedding"` key, two different
    nestings" trap is gone, so the unpacking below is a single
    `[e.values for e in response.embeddings]` in both cases. The two public
    entry points stay separate anyway — for the task_type reason below, not
    for the shape.
  * BATCHING IS NO LONGER CHUNKED FOR YOU. The old SDK silently split an
    iterable into `EMBEDDING_MAX_BATCH_SIZE = 100`-item HTTP calls. This one
    does not: `models.py` builds exactly one request containing every item you
    passed. So N inputs = 1 round trip, however large N is — which is great up
    to the server's per-request batch cap and a hard failure past it. See the
    note in `embed_documents` for what that means for callers.
  * Model naming: `_transformers.t_model()` prefixes a bare id with `models/`
    and leaves an already-prefixed name alone, so both
    `"gemini-embedding-001"` and `"models/gemini-embedding-001"` serialize to
    the same `models/gemini-embedding-001`. `config.GEMINI_EMBEDDING_MODEL`
    therefore needs no change.

MODEL / DIMENSION WARNING: Google retired `models/text-embedding-004` on
2026-01-14 in favour of `models/gemini-embedding-001`. Whichever you point
`GEMINI_EMBEDDING_MODEL` at, its output width must equal
`config.EMBEDDING_DIMENSION` and the `numDimensions` of the Atlas index.
"""

from __future__ import annotations

import logging
import math
import threading

from google import genai
from google.genai import types

from app import config

logger = logging.getLogger(__name__)


# --- Task types -------------------------------------------------------------
# ASYMMETRIC EMBEDDING — the single most important idea in this file.
#
# The naive mental model of embeddings is "similar texts get similar vectors".
# That model is wrong for retrieval, and believing it costs you recall.
#
# In a RAG system the two sides of a comparison are not the same kind of text.
# One side is a short, under-specified question ("what's our refund policy?").
# The other is a long, declarative passage that happens to *answer* it
# ("Customers may return unopened items within 30 days of delivery..."). As
# strings, those two are not very similar at all — different length, different
# vocabulary, different grammatical mood. A model trained purely on "are these
# two texts alike?" (that's what SEMANTIC_SIMILARITY optimises) would happily
# rank the question closer to *another question* about refunds than to the
# paragraph that actually answers it. Useless for retrieval.
#
# Retrieval-tuned embedding models are trained on a different objective:
# question-passage pairs, pushing a query and its correct answer passage
# together in vector space even though they don't look alike. `task_type` is
# how you tell the model which *role* the text you're sending plays, so it can
# project it into the right region of that shared space. Queries and documents
# effectively get two different (but co-trained) projections — hence
# "asymmetric".
#
# Both sides must be labelled consistently, and consistently over time:
#   * text being STORED/indexed  -> RETRIEVAL_DOCUMENT
#   * a question being SEARCHED  -> RETRIEVAL_QUERY
#
# What makes getting this wrong such a nasty bug is that nothing breaks. Pass
# the wrong task_type, or omit it entirely, and you still get a vector of the
# right length, Atlas still accepts it, `$vectorSearch` still returns exactly
# top-k results, and the agent still answers. The results are just measurably
# worse — the right chunk drops from rank 1 to rank 7 and falls off the end of
# a top-4 cutoff. There is no exception, no log line, no failing test; the
# only symptom is "the bot seems a bit dumb". That's why these are two
# separate named functions instead of one function with a task_type argument:
# a caller cannot accidentally leave it unset, and a code reviewer can see at
# the call site which side of the asymmetry they're on.
#
# One more consequence: if you ever re-embed the corpus with a different
# task_type (or a different model), you must re-embed *all* of it. A
# collection holding a mix of document-typed and similarity-typed vectors is
# comparing coordinates from two different spaces, which degrades quietly in
# the same invisible way.
#
# CASING IS LOAD-BEARING under google-genai. These constants used to be
# lowercase because the old SDK folded case for you; the new SDK forwards the
# string untouched (see the module docstring), so they must be the exact
# UPPER_SNAKE_CASE enum names or the API rejects the request.
_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
_TASK_QUERY = "RETRIEVAL_QUERY"


# --- Lazy, one-time client construction --------------------------------------
# Under `google-genai` there is no module-global `configure()`; credentials and
# transport live on a `genai.Client` instance, and every call is made through
# it. So the thing we build exactly once and reuse is the client itself. Reuse
# matters for more than tidiness: the client owns the underlying httpx
# connection pool, so building a fresh one per embed call would throw away
# keep-alive connections and pay a TLS handshake on every request.
#
# Why not build it at import time: `config.GOOGLE_API_KEY` may legitimately be
# empty. Importing this module must stay free of side effects and free of
# preconditions — a unit test that imports `app.rag.embeddings` to check a
# helper, a `--help` invocation, or an editor's autocomplete indexing the
# package should not explode because no .env exists. That's the whole point of
# the `config.require(...)` pattern documented in app/config.py: config stays
# lenient, and the module that actually *needs* a secret demands it at the
# point of use, so the error names the missing variable instead of surfacing
# as an opaque authentication failure from inside the HTTP layer three frames
# deeper.
#
# (`genai.Client()` would also happily pick GOOGLE_API_KEY up from the process
# environment on its own. We pass it explicitly through `config.require` anyway
# so the missing-key failure is ours, is early, and names the variable.)
_client: genai.Client | None = None

# Same reasoning as faster_whisper_stt.py's `_model_lock`: a threading.Lock,
# not an asyncio.Lock. Even though the public functions here are coroutines,
# an ingestion script may drive them from a worker thread, and nothing stops
# two OS threads from entering `_get_client()` at once. Only a threading.Lock
# actually blocks the second thread. Double-checked locking keeps the common
# case (client already built) free of lock acquisition.
_client_lock = threading.Lock()


def _get_client() -> genai.Client:
    """Build the Gemini client exactly once, on first real use."""
    global _client

    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            api_key = config.require(config.GOOGLE_API_KEY, "GOOGLE_API_KEY")
            _client = genai.Client(api_key=api_key)
            logger.info(
                "Gemini embedding client constructed (model=%s, dimension=%d).",
                config.GEMINI_EMBEDDING_MODEL,
                config.EMBEDDING_DIMENSION,
            )

    return _client


# --- Post-processing ---------------------------------------------------------
def _normalize(vector: list[float]) -> list[float]:
    """Scale a vector to unit length (L2 norm == 1).

    Why this is here rather than assumed: Google's embedding models return
    pre-normalized vectors only at their *native* output width.
    `gemini-embedding-001` is a Matryoshka-style model — asking for fewer than
    3072 dimensions truncates the vector, and truncating destroys the unit
    norm. Google's docs are explicit that you must re-normalize non-3072
    outputs yourself.

    That matters because Atlas Vector Search's `cosine` similarity, and the
    `dotProduct` option in particular, assume unit-length vectors; feeding it
    vectors of varying magnitude skews ranking toward whichever chunks happen
    to have larger norms. Normalizing an already-normalized vector is a no-op,
    so applying this unconditionally is safe regardless of which model or
    width is configured.
    """
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        # A genuinely all-zero embedding should never happen; if it does,
        # dividing would raise ZeroDivisionError deep in a list comprehension.
        # Fail loudly and specifically instead.
        raise RuntimeError(
            "Embedding API returned an all-zero vector — refusing to normalize it. "
            "This usually means the input text was empty or unusable."
        )
    return [component / norm for component in vector]


def _validate(vector: list[float]) -> list[float]:
    """Check the vector width against the configured/indexed dimension.

    Caught here, a mismatch says "model X returned 3072, config says 768".
    Left uncaught, it surfaces much later as an Atlas `$vectorSearch` error
    about the query vector — or worse, at *ingest* time it doesn't surface at
    all until someone finally runs a search against the finished index.
    """
    if len(vector) != config.EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Embedding model {config.GEMINI_EMBEDDING_MODEL!r} returned "
            f"{len(vector)} dimensions, but config.EMBEDDING_DIMENSION is "
            f"{config.EMBEDDING_DIMENSION}. These must match each other AND the "
            f"'numDimensions' of the Atlas vector index "
            f"{config.VECTOR_INDEX_NAME!r}."
        )
    return vector


# --- Internal call -----------------------------------------------------------
# Why async: this app is one asyncio event loop serving every live WebSocket
# session at once. Embedding is a *network*-bound call (an HTTPS round trip to
# Google, typically 100-400ms), not CPU-bound like
# faster_whisper_stt.transcribe(). For network work the right tool is a native
# async client, not `asyncio.to_thread` — awaiting it yields control back to
# the loop for the whole flight time, so every other session keeps being
# serviced, and it costs no worker thread. The installed SDK provides exactly
# that as `client.aio.models.embed_content`, a real coroutine over
# httpx.AsyncClient, so these functions are `async def` and must be awaited.
# (Contrast: transcribe() is sync-and-blocking, so callers there must wrap it
# in asyncio.to_thread. Different bottleneck, different tool.)
#
# For a synchronous context — e.g. a one-shot `python -m scripts.ingest_docs`
# with no loop running — wrap the call: `asyncio.run(embed_documents(chunks))`.
async def _embed(contents, task_type: str) -> list[list[float]]:
    """Call the SDK and return the raw vectors, always as a list of lists.

    `contents` may be a single string or a list of strings; the response shape
    is uniform either way (one `ContentEmbedding` per input), so this returns
    `list[list[float]]` in both cases and the callers below index into it.

    Deliberately does NOT catch exceptions. This is infrastructure, not an
    agent tool: an auth failure, a quota rejection or a network error must
    propagate so ingestion aborts loudly rather than writing a collection full
    of nulls, and so a failed retrieval surfaces instead of quietly answering
    from no context at all.
    """
    client = _get_client()

    response = await client.aio.models.embed_content(
        model=config.GEMINI_EMBEDDING_MODEL,
        contents=contents,
        config=types.EmbedContentConfig(
            task_type=task_type,
            # Pin the width explicitly rather than relying on the model default,
            # so the vectors always match the Atlas index. Models with a fixed
            # output size ignore/accept this; flexible ones honour it.
            output_dimensionality=config.EMBEDDING_DIMENSION,
        ),
    )

    # `embeddings` and `values` are both typed Optional on the response models,
    # so a malformed/empty reply would otherwise surface as a `NoneType is not
    # iterable` several lines later. Name the real problem instead.
    if not response.embeddings:
        raise RuntimeError(
            "Embedding API returned a response with no embeddings. "
            f"(model={config.GEMINI_EMBEDDING_MODEL!r}, task_type={task_type!r})"
        )

    vectors: list[list[float]] = []
    for position, embedding in enumerate(response.embeddings):
        if embedding.values is None:
            raise RuntimeError(
                f"Embedding API returned an embedding with no values at index {position}."
            )
        vectors.append(list(embedding.values))

    return vectors


# --- Public API --------------------------------------------------------------
async def embed_document(text: str) -> list[float]:
    """Embed one piece of text that is being STORED and indexed.

    Use for anything written into the `documents` or `memory` collections —
    a chunk of a source PDF, a note, a past conversation turn saved for
    semantic recall. Uses task_type RETRIEVAL_DOCUMENT; see the asymmetric
    embedding note at the top of this file for why that is not
    interchangeable with `embed_query`.

    Must be awaited from inside the event loop (or via `asyncio.run` from a
    plain script).

    Args:
        text: The passage to embed. Must be non-empty.

    Returns:
        A unit-length list of `config.EMBEDDING_DIMENSION` floats.

    Raises:
        ValueError: if `text` is empty or whitespace only.
        RuntimeError: if GOOGLE_API_KEY is unset, or the returned vector has
            an unexpected width.
    """
    if not text or not text.strip():
        raise ValueError("embed_document() received empty text — nothing to embed.")

    vectors = await _embed(text, _TASK_DOCUMENT)
    return _normalize(_validate(vectors[0]))


async def embed_query(text: str) -> list[float]:
    """Embed a user's question that is being SEARCHED with.

    Use for the vector handed to Atlas `$vectorSearch` — never for text you
    are about to store. Uses task_type RETRIEVAL_QUERY.

    Must be awaited. This one sits directly on the live voice latency path
    (user stops speaking -> transcribe -> embed_query -> $vectorSearch ->
    LLM), which is the concrete reason it is async rather than a blocking
    call parked in a thread.

    Args:
        text: The question to embed. Must be non-empty.

    Returns:
        A unit-length list of `config.EMBEDDING_DIMENSION` floats.

    Raises:
        ValueError: if `text` is empty or whitespace only.
        RuntimeError: if GOOGLE_API_KEY is unset, or the returned vector has
            an unexpected width.
    """
    if not text or not text.strip():
        raise ValueError("embed_query() received empty text — nothing to embed.")

    vectors = await _embed(text, _TASK_QUERY)
    return _normalize(_validate(vectors[0]))


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed many passages at once, for the ingestion path.

    This is still ONE HTTP round trip instead of `len(texts)` of them, which
    is the whole point: 200 sequential round trips is a minute of waiting, it
    burns the per-minute *request* quota (which you hit long before the token
    quota), and it leaves you 200 partial successes to reason about on failure
    instead of a single retryable call.

    What changed with google-genai (verified in the installed 2.22.0 source):
    the SDK no longer chunks for you. `google-generativeai` split an iterable
    into 100-item `BatchEmbedContentsRequest`s behind your back, so a
    250-chunk list quietly became 3 calls. `google-genai` builds exactly one
    `{model}:batchEmbedContents` request containing every item you passed. The
    Gemini API enforces a maximum batch size per request, so a corpus larger
    than that cap must now be chunked by the CALLER (in `app/rag/ingest.py`)
    rather than being handled invisibly here — a very large single call will
    fail with an explicit API error rather than silently succeeding.

    Order is preserved: result[i] is the embedding of texts[i], so callers can
    zip it straight back against their chunk list.

    Args:
        texts: Passages to embed. Must be non-empty, and no element may be
            blank — a blank string would produce a garbage/zero vector that
            then pollutes retrieval for everything else.

    Returns:
        A list of unit-length vectors, one per input, in the same order.

    Raises:
        ValueError: if `texts` is empty or contains a blank entry.
        RuntimeError: if GOOGLE_API_KEY is unset, or a returned vector has an
            unexpected width.
    """
    if not texts:
        raise ValueError("embed_documents() received an empty list — nothing to embed.")

    for index, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(f"embed_documents() received a blank text at index {index}.")

    logger.debug("Embedding %d documents in a single batch request.", len(texts))

    # Batch shape: one ContentEmbedding per input, in input order.
    vectors = await _embed(texts, _TASK_DOCUMENT)

    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Embedding API returned {len(vectors)} vectors for {len(texts)} inputs — "
            "refusing to continue, since results would be misaligned with their chunks."
        )

    return [_normalize(_validate(vector)) for vector in vectors]
