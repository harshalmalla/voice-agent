"""
Semantic conversation memory, scoped to a single session.

The agent already carries the last few turns in its prompt as ordinary chat
history. That window is short, and it is chronological: it remembers what was
said thirty seconds ago and nothing about the thing the user mentioned twenty
minutes and forty turns back. This module is the other kind of memory —
every completed exchange is embedded and written to
`config.MEMORY_COLLECTION`, and `recall()` retrieves the handful of PAST
exchanges that are semantically closest to what the user just asked,
regardless of how long ago they happened.

It is the same machinery as `app/rag/ingest.py` pointed at a different corpus:
`ingest.py` embeds a static pile of files once, this embeds the conversation
as it happens. The one behavioural difference is scoping, and it is the
important one — see `recall()`.

--- Why an exchange is ONE document, not two ------------------------------
The obvious schema is one document per message: a `role: "user"` row and a
`role: "assistant"` row. It retrieves badly, and it does so in a way that only
shows up with real conversations.

An assistant reply is frequently uninterpretable without the question that
produced it. "Yes, about three weeks" is a perfectly good answer and a
completely useless retrieval hit: as a standalone document its embedding is
near-meaningless, it will match almost nothing, and on the rare occasion it IS
retrieved it lands in the next prompt as a floating fragment that tells the
model nothing and may actively mislead it. The question half has the opposite
problem — it embeds well, matches a similar later question nicely, and then
supplies the model with a restatement of what was asked rather than what was
answered.

Storing the pair as one document with `text` holding both halves fixes both
ends at once: the question's vocabulary makes the document findable, and the
answer travels with it so a hit is actually informative. It also halves the
document count and the number of embedding calls, and it removes the
join-two-rows-back-together problem entirely.

The halves are kept alongside as `user_text` and `assistant_text` so a caller
that wants to display recalled memory — or to rebuild it as proper chat turns
for the model — never has to parse the combined string back apart. `role` is
deliberately absent from the schema: a document that contains both roles
cannot have one.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app import config
from app.rag import embeddings, store

logger = logging.getLogger(__name__)


def _format_exchange(user_text: str, assistant_text: str) -> str:
    """Render a question/answer pair as the single text that gets embedded.

    The `User:` / `Assistant:` labels are part of what is embedded on purpose.
    They cost a few tokens and they give the model — both the embedding model
    and, later, the one reading the recalled text in its prompt — an explicit
    marker of who said what, instead of two paragraphs run together with no
    indication that they came from different speakers.
    """
    return f"User: {user_text}\nAssistant: {assistant_text}"


async def add_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    """Store one completed exchange in this session's memory.

    Called after the agent has produced its answer, not before — a half-turn
    with no reply yet is exactly the uninterpretable fragment the module
    docstring argues against storing.

    The document written is:

      * `text` — both halves, via `_format_exchange`. This is the embedded
        field and the one `store.vector_search` projects back.
      * `user_text`, `assistant_text` — the halves kept separately for
        display and for reconstructing chat turns.
      * `session_id` — the scope. Declared in `store.FILTER_FIELDS`, which is
        what makes the pre-filter in `recall()` legal.
      * `store.EMBEDDING_FIELD` — the vector, from `embed_document` (this text
        is being stored, not searched with).
      * `timestamp` — timezone-aware UTC, the same field name the `documents`
        collection uses so one `$project` whitelist serves both.

    Blank input is a no-op rather than an error. An exchange where the user
    said nothing intelligible (an empty transcription from a stray click on
    the mic button) is a normal event on the voice path and should not raise
    into the WebSocket handler; there is simply nothing worth remembering.

    Args:
        session_id: The conversation this exchange belongs to. Required —
            an unscoped memory document could be recalled into a different
            user's context, which is the failure this whole design exists to
            prevent.
        user_text: What the user said.
        assistant_text: What the agent replied.

    Raises:
        ValueError: if `session_id` is empty.
        RuntimeError: if GOOGLE_API_KEY or MONGODB_URI is unset.
        store.VectorStoreError: if the write fails.
    """
    if not session_id:
        raise ValueError("add_turn() requires a session_id — memory must be scoped.")

    if not user_text.strip() or not assistant_text.strip():
        logger.debug(
            "Skipping memory write for session %s — one half of the exchange is empty.",
            session_id,
        )
        return

    text = _format_exchange(user_text.strip(), assistant_text.strip())
    vector = await embeddings.embed_document(text)

    record = {
        "text": text,
        "user_text": user_text.strip(),
        "assistant_text": assistant_text.strip(),
        "session_id": session_id,
        store.EMBEDDING_FIELD: vector,
        "timestamp": datetime.now(timezone.utc),
    }

    await store.upsert_chunks(config.MEMORY_COLLECTION, [record])
    logger.debug("Stored one memory exchange for session %s.", session_id)


async def recall(
    session_id: str,
    query: str,
    limit: int = config.RETRIEVAL_TOP_K,
) -> list[dict]:
    """Retrieve this session's past exchanges most relevant to `query`.

    Two details here are load-bearing and both are easy to get wrong silently.

    FIRST, the query is embedded with `embeddings.embed_query`, never
    `embed_document`. Gemini's retrieval embeddings are asymmetric: questions
    and passages are projected with different task types precisely so that a
    short question lands near the long passage that answers it. Embedding a
    question as though it were a document produces a vector that is perfectly
    valid, raises nothing, and quietly retrieves worse — the failure mode is a
    few points of recall, which no test catches unless it asserts on the call.
    See the task-type notes at the top of `embeddings.py`.

    SECOND, `session_id` is passed as a PRE-filter, inside the
    `$vectorSearch` stage, rather than as a `$match` afterwards. The `memory`
    collection holds every conversation the app has ever had. Post-filtering
    would let the vector search pick its global top-k first and only then
    discard the other sessions' turns, so the number of results that survive
    would depend on how this session's history compares to every stranger's —
    normally badly. You would ask for four and get one, or none, with nothing
    in the logs to say why, and it would get worse as the app gained users.
    `store._build_search_pipeline` documents this at length.

    Args:
        session_id: The conversation to search within. Empty means there is
            no session, so there is nothing to recall.
        query: What the user just asked, in natural language.
        limit: Maximum exchanges to return.

    Returns:
        Past exchanges, most similar first, each carrying `text`, `user_text`,
        `assistant_text`, `session_id`, `timestamp` and a `score` (0..1). An
        empty list when this session has no stored history yet, or when
        nothing matches — the first turn of every conversation takes this
        path, so callers must handle it as the ordinary case rather than an
        edge case.

    Raises:
        RuntimeError: if GOOGLE_API_KEY or MONGODB_URI is unset.
        store.VectorStoreError: if the search fails, e.g. because the Atlas
            vector index does not exist on the `memory` collection. Note that
            this is NOT the same condition as "no history yet": an empty
            collection with a live index returns `[]` normally, whereas a
            missing index is a setup error and must stay loud rather than
            being flattened into a silent empty result that would look like
            memory simply never working.
    """
    if not session_id or not query.strip():
        return []

    query_vector = await embeddings.embed_query(query)

    results = await store.vector_search(
        config.MEMORY_COLLECTION,
        query_vector,
        limit=limit,
        pre_filter={"session_id": session_id},
    )

    logger.debug(
        "Recalled %d past exchanges for session %s.", len(results), session_id
    )
    return results
