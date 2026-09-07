"""
The agent loop: retrieval, prompt construction, Gemini function calling, and
the memory write that makes the next turn possible.

This is the module every other module in the project points at. `stt.py` hands
it a transcript and a language; `embeddings.py` and `store.py` supply the
context; `tools.py` supplies the callables and the schema the model chooses
from; `session_manager.py` supplies the recent-turn window and the outbound
event queue; `memory.py` receives the finished exchange. `answer()` is the
only public entry point, and the WebSocket handler calls it once per user turn.

--- The function-calling API, confirmed against the installed SDK ------------
`google-genai` 2.22.0, source read at
`.venv/lib/python3.12/site-packages/google/genai/` (`models.py`, `types.py`,
`_extra_utils.py`). This is the current SDK; the retired
`google-generativeai` package, whose `genai.configure()` /
`GenerativeModel(...).start_chat()` shape most tutorials still show, is NOT
what is installed here and none of its API applies.

  * The call is `client.aio.models.generate_content(*, model, contents,
    config=None)` — all keyword-only, a genuine coroutine
    (`inspect.iscoroutinefunction(models.AsyncModels.generate_content)` is
    True), returning `types.GenerateContentResponse`.
  * Everything that is not the model or the contents lives in
    `types.GenerateContentConfig`, including `tools`, `system_instruction`,
    `tool_config` and `automatic_function_calling`.
  * Tools are declared as `types.Tool(function_declarations=[...])`. The
    declarations themselves may be handed over as plain dicts: pydantic
    coerces each into a `types.FunctionDeclaration` whose `parameters` becomes
    a `types.Schema`. Verified by constructing
    `types.Tool(function_declarations=TOOL_DECLARATIONS)` from `tools.py`'s
    literal dicts and reading the result back — the nested
    `{"type": "OBJECT", "properties": {...}, "required": [...]}` came out as
    `Schema(type=<Type.OBJECT>, properties={'query': Schema(type=<Type.STRING>,
    description=...)}, required=['query'])`. So `TOOL_DECLARATIONS` needs no
    translation layer and stays readable as data.

  * AUTOMATIC FUNCTION CALLING IS DISABLED HERE, DELIBERATELY. The SDK will,
    by default, execute Python callables you pass in `tools` and loop for you
    (`models.py` around the `remaining_remote_calls_afc` counter). That is the
    wrong shape for this app for three reasons: the dashboard would never see
    a `tool_call` or `tool_result` event, because the loop happens inside the
    SDK; our tools are session-bound closures from `build_tools(session_id)`
    rather than importable module-level functions the SDK can name; and
    `_extra_utils.get_function_map` explicitly raises
    `UnsupportedFunctionError` for `async def` tools, which is all of ours.
    The spelling is
    `types.AutomaticFunctionCallingConfig(disable=True)` — the field is
    `disable`, and `_extra_utils.should_disable_afc` reads exactly that.
    (Passing schema dicts rather than callables already leaves the SDK's
    function map empty, so AFC would find nothing to run; the flag is set
    anyway so the intent is stated in the request rather than implied by the
    absence of something.)

  * A function call comes back as `response.function_calls`, a property that
    collects `part.function_call` from `candidates[0].content.parts` and
    returns None when there are none. Each is a `types.FunctionCall` with
    `name: str | None` and `args: dict[str, Any] | None` (also `id`,
    `will_continue`, `partial_args`, unused here). A response can carry
    SEVERAL calls at once, so the loop below iterates rather than taking the
    first.

  * A result goes back as
    `types.Part.from_function_response(*, name: str, response: dict[str, Any])`
    — `response` must be a dict, not a string. The SDK's own AFC path wraps
    successes as `{"result": ...}` and failures as `{"error": ...}`
    (`_extra_utils.get_function_response_parts`), and this module follows that
    convention so the model sees the shape it was trained on.

  * MULTI-TURN CONTENTS. `contents` is a `list[types.Content]`, each with a
    `role` and `parts`. The roles are `"user"` and `"model"` — and the tool
    result goes back under role `"user"`, NOT `"function"` or `"tool"`.
    That is not a guess: `models.py` builds
    `types.Content(role='user', parts=func_response_parts)` in its own AFC
    loop. The model's own function-call turn must be appended first, verbatim
    (`response.candidates[0].content`), so the request stays a coherent
    call-then-result pair; dropping it leaves a response answering a call the
    transcript never contains.

--- Why the tool loop is bounded --------------------------------------------
`config.MAX_TOOL_ITERATIONS` (default 5) is a hard cap on how many times one
turn may go round the call-model / run-tools cycle, and it is the single most
important safety property in this file.

Nothing in the protocol says a model must eventually stop calling tools. A
model that has decided `web_search` did not quite answer the question will
happily call `web_search` again, with a near-identical query, forever — and
each pass is a paid API call whose prompt is LARGER than the last, because the
previous call and its result were appended to `contents`. Unbounded, that is
not a slow turn; it is an infinite loop that bills real money at an
accelerating rate, on a live user session, while the person waits in silence
for an answer that is never coming. The cap converts an unbounded failure into
a bounded, logged, mildly disappointing one.

Hitting the cap returns the best text available rather than raising: the user
gets something spoken back, and the incident is a WARNING in the logs rather
than a dead WebSocket turn.

--- Why a bad tool name is not an error -------------------------------------
The model can hallucinate a tool that does not exist, and it does so most
often when the conversation is going badly already. Dispatching straight
through `build_tools(session_id)[name]` would raise `KeyError` out of the loop
and kill the turn. Instead an unknown name comes back to the model as an error
string in the function response, which is text it can read and recover from —
usually by calling the right tool on the next pass. Same reasoning as
`tools.py`'s "a tool never raises" rule, extended to the dispatch itself.

--- Failure policy: infrastructure raises, policy decides -------------------
`app/audio/stt.py` sets out this split, and this module sits on the same side
of it as that one. `embeddings.py`, `store.py` and `memory.py` raise, because
none of them can know what a failure means. This module CAN know: it knows a
user is holding a microphone waiting for a reply. So retrieval failure
degrades to answering without context, and total model failure degrades to an
apology sentence. Nothing here propagates into the WebSocket handler, because
an exception there means the user hears nothing at all and cannot tell a
Mongo outage from a broken app.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google import genai
from google.genai import types

from app import config
from app.agent.tools import TOOL_DECLARATIONS, build_tools
from app.rag import embeddings, memory, store
from app.services.session_manager import get_session_manager

logger = logging.getLogger(__name__)


MODEL_RETRY_ATTEMPTS = 2

MODEL_RETRY_BACKOFF_SECONDS = 1.5

MEMORY_SOURCE_LABEL = "conversation memory"

FALLBACK_ANSWER = (
    "Sorry, I could not reach my language model just now, so I cannot answer that. "
    "Please try again in a moment."
)

TOOL_LIMIT_NOTE = (
    "I looked into that as far as I could, but I could not finish working it out. "
    "Could you ask me again, more specifically?"
)


_TOOLS: list[types.Tool] = [types.Tool(function_declarations=TOOL_DECLARATIONS)]

_AUTOMATIC_FUNCTION_CALLING_OFF = types.AutomaticFunctionCallingConfig(disable=True)


SYSTEM_INSTRUCTION = """You are a helpful voice assistant. Your replies are spoken aloud, so keep them short, natural and free of markdown, bullet points, code blocks and URLs.

Answer from the CONTEXT section below whenever it is relevant to the question, and say which source you took it from when the user would care. The context is retrieved automatically from the user's own documents and from earlier parts of this conversation; it is not always relevant, and you should ignore it when it is not.

If neither the context nor your tools give you the answer, say plainly that you do not know. Never invent a fact, a number, a date or a source. A wrong answer spoken confidently is worse than an admission of ignorance.

Use a tool when the question needs live information, arithmetic, the current time, the user's notes, or a reminder. Do not use a tool to answer something the context already answers.

Reply in {language_name}, the language the user is speaking. Write it in that language's own script, not transliterated, and do not translate or explain it in English as well."""


_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "od": "Odia",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
}


_client: genai.Client | None = None

_client_lock = asyncio.Lock()


async def _get_client() -> genai.Client:
    """Build the Gemini client exactly once, on first real use.

    A second client instance to the one in `embeddings.py` rather than a
    shared one, because the two modules are otherwise independent and neither
    should import the other's private singleton to make a call. The object is
    cheap — `genai.Client(api_key=...)` performs no I/O — so the duplication
    costs a few bytes and buys the two modules the freedom to be configured
    differently later (a different base URL, a Vertex backend for one of them).

    Guarded by an `asyncio.Lock` rather than `threading.Lock` because every
    caller of this module is on the one event loop, and an async lock yields
    to the loop while waiting instead of blocking the thread that is servicing
    every other live session.
    """
    global _client

    if _client is not None:
        return _client

    async with _client_lock:
        if _client is None:
            api_key = config.require(config.GOOGLE_API_KEY, "GOOGLE_API_KEY")
            _client = genai.Client(api_key=api_key)
            logger.info(
                "Gemini agent client constructed (primary=%s, fallback=%s).",
                config.GEMINI_MODEL_PRIMARY,
                config.GEMINI_MODEL_FALLBACK,
            )

    return _client


def _language_name(language: str) -> str:
    """Render a language code as a name the model can act on.

    The code arrives from faster-whisper via `app/audio/stt.py`, and "reply in
    ta" is a materially weaker instruction than "reply in Tamil": the two-letter
    codes are rare in training text and are ambiguous with ordinary words, so a
    model asked to answer "in or" can and does read that as English. An
    unrecognised code falls back to naming the code itself, which is still
    better than silently dropping the instruction.
    """
    return _LANGUAGE_NAMES.get(language, language or "English")


def _is_temporary_model_error(error: Exception) -> bool:
    """Decide whether a model failure is worth retrying.

    String matching on the error text, exactly as the reference project's
    `_invoke_with_model_fallback` does, and for the same reason: the failures
    that matter arrive through several different exception types depending on
    where they were raised (the SDK's own `errors.APIError`, an httpx
    transport error, a bare `RuntimeError` from a wrapper), and all of them
    render the HTTP status or the gRPC status name into the message.

    A retry is right for these and wrong for everything else. 429 and
    RESOURCE_EXHAUSTED mean "you are going too fast", 5xx and UNAVAILABLE mean
    "we are having a moment" — both pass on the second attempt often enough to
    be worth a second and a half of a user's patience. A 400 for a malformed
    request, or a 403 for a bad key, will fail identically forever; retrying
    those only adds latency to a turn that is already lost, so they propagate
    immediately to `answer()`, which apologises.
    """
    message = str(error).upper()
    return any(
        code in message
        for code in ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
    )


async def _generate(
    model: str,
    contents: list[types.Content],
    system_instruction: str,
) -> types.GenerateContentResponse:
    """Make one Gemini call. The only place in this module that touches the network.

    Kept as a thin, single-purpose seam on purpose. Everything above it — the
    retry/fallback ladder, the tool loop, the iteration cap, the degradation
    paths — is pure policy that must be testable without an API key, and this
    function is the one thing a test replaces to get there. Widening it (by
    folding the retry loop in here, say) would make the interesting logic
    untestable offline.

    Raises whatever the SDK raises. Retry policy belongs to the caller.
    """
    client = await _get_client()

    return await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=_TOOLS,
            automatic_function_calling=_AUTOMATIC_FUNCTION_CALLING_OFF,
        ),
    )


async def _generate_with_fallback(
    contents: list[types.Content],
    system_instruction: str,
) -> types.GenerateContentResponse:
    """Call the model, surviving a busy primary by retrying and then downgrading.

    The ladder, mirroring `_invoke_with_model_fallback` in
    `/Users/harry/Downloads/Projects/multiagent copy/pipeline.py`:

      primary, attempt 1 -> backoff -> primary, attempt 2
        -> fallback, attempt 1 -> backoff -> fallback, attempt 2 -> give up.

    Two models rather than one because the failure this actually defends
    against is a per-model quota, not an outage: the primary model's
    free-tier requests-per-minute allowance is the thing a demo runs into, and
    when it is exhausted the lighter `GEMINI_MODEL_FALLBACK` still has its own.
    A worse answer, spoken, beats a better answer the user never hears.

    The backoff is `asyncio.sleep`, not `time.sleep` as in the synchronous
    reference — the same second and a half of blocking would freeze audio for
    every other live session on the loop. It is short for the same reason: a
    person is standing there waiting, and a long exponential backoff on the
    voice path is indistinguishable from a hang.

    Raises:
        Whatever non-temporary error the SDK raised, unchanged — a malformed
        request or a bad API key is a bug or a misconfiguration, and burning
        four calls on it would only delay finding out.
        RuntimeError: if every attempt on both models failed temporarily.
    """
    last_error: Exception | None = None

    for model in (config.GEMINI_MODEL_PRIMARY, config.GEMINI_MODEL_FALLBACK):
        for attempt in range(MODEL_RETRY_ATTEMPTS):
            try:
                return await _generate(model, contents, system_instruction)
            except Exception as error:
                if not _is_temporary_model_error(error):
                    logger.error(
                        "Gemini call to %s failed permanently (%s: %s) — not retrying.",
                        model,
                        type(error).__name__,
                        error,
                    )
                    raise

                last_error = error
                logger.warning(
                    "Gemini call to %s failed temporarily on attempt %d/%d (%s: %s).",
                    model,
                    attempt + 1,
                    MODEL_RETRY_ATTEMPTS,
                    type(error).__name__,
                    error,
                )
                if attempt + 1 < MODEL_RETRY_ATTEMPTS:
                    await asyncio.sleep(MODEL_RETRY_BACKOFF_SECONDS)

        if model == config.GEMINI_MODEL_PRIMARY:
            logger.warning(
                "Primary model %s is still unavailable — falling back to %s.",
                config.GEMINI_MODEL_PRIMARY,
                config.GEMINI_MODEL_FALLBACK,
            )

    raise RuntimeError(
        f"Gemini is temporarily unavailable: both {config.GEMINI_MODEL_PRIMARY!r} and "
        f"{config.GEMINI_MODEL_FALLBACK!r} failed {MODEL_RETRY_ATTEMPTS} attempts each."
    ) from last_error


def _to_source(record: dict, default_source: str) -> dict:
    """Reshape one retrieved document into the `sources` entry the dashboard draws.

    The three keys are not arbitrary: `frontend/app.js`'s `buildSourceList`
    reads `source.source`, `source.score` (and calls `.toFixed(3)` on it, so it
    must be a number, never a string) and `source.text`. Anything else in the
    record — `chunk_index`, `timestamp`, the `session_id` — is dropped here
    rather than sent, because the retrieved text can be long and this event
    goes down a WebSocket on the live path.

    `default_source` supplies the label for memory hits, which carry no
    `source` field at all: `memory.add_turn` stores `session_id` instead, and
    showing a raw session id in the activity feed would tell the viewer
    nothing.
    """
    score = record.get("score")
    return {
        "text": record.get("text", ""),
        "source": record.get("source") or default_source,
        "score": float(score) if isinstance(score, (int, float)) else 0.0,
    }


async def _retrieve(session_id: str, user_text: str) -> list[dict]:
    """Gather document chunks and past exchanges relevant to this question.

    THE QUERY IS EMBEDDED ONCE AND THE TWO SEARCHES RUN CONCURRENTLY. Both
    details are about latency, and both are visible to the user as silence
    after they stop speaking.

    The single embedding call is possible because both collections are indexed
    with the same model at the same width (see `store.VECTOR_INDEX_DEFINITION`)
    and both are being searched with a question rather than a passage, so one
    RETRIEVAL_QUERY vector serves both. Embedding twice would double a network
    round trip to buy an identical vector.

    The `asyncio.gather` is the more important half. The document search and
    the memory recall are independent round trips to Atlas — neither reads the
    other's output — so running them one after the other spends the sum of two
    network waits where the loop could have spent the maximum. On a path that
    already has to fit a transcription, an embedding, a model call and a
    speech synthesis inside a conversational pause, that saved round trip is
    the difference between an assistant that feels responsive and one that
    feels broken.

    `return_exceptions=True` is what keeps the two failures independent: a
    missing vector index on `memory` must not also cost the user the document
    context that was retrieved successfully alongside it.

    Never raises. A total retrieval failure returns `[]`, and `answer()` then
    prompts the model with no context at all — degraded, but still a turn.
    """
    try:
        query_vector = await embeddings.embed_query(user_text)
    except Exception as error:
        logger.warning(
            "Retrieval skipped for session %s — embedding the query failed (%s: %s). "
            "Answering without context.",
            session_id,
            type(error).__name__,
            error,
            exc_info=True,
        )
        return []

    documents_result, memory_result = await asyncio.gather(
        store.vector_search(
            config.DOCUMENTS_COLLECTION,
            query_vector,
            limit=config.RETRIEVAL_TOP_K,
        ),
        memory.recall(session_id, user_text, limit=config.RETRIEVAL_TOP_K),
        return_exceptions=True,
    )

    sources: list[dict] = []

    if isinstance(documents_result, BaseException):
        logger.warning(
            "Document search failed for session %s (%s: %s) — continuing without it.",
            session_id,
            type(documents_result).__name__,
            documents_result,
        )
    else:
        sources.extend(_to_source(record, "unknown document") for record in documents_result)

    if isinstance(memory_result, BaseException):
        logger.warning(
            "Memory recall failed for session %s (%s: %s) — continuing without it.",
            session_id,
            type(memory_result).__name__,
            memory_result,
        )
    else:
        sources.extend(_to_source(record, MEMORY_SOURCE_LABEL) for record in memory_result)

    logger.info(
        "Retrieved %d source(s) for session %s.",
        len(sources),
        session_id,
    )
    return sources


def _format_context(sources: list[dict]) -> str:
    """Render retrieved chunks as a delimited, labelled block for the prompt.

    Delimiters and labels, rather than pasting the chunks in end to end, for
    two separate reasons.

    The first is citation: the model cannot say "according to handbook.pdf" if
    the filename never reached it. Each chunk is introduced by its source, so
    attribution is available rather than invented.

    The second is boundaries. Four chunks concatenated with blank lines read
    as one continuous passage, and a model asked to summarise them will
    cheerfully join a sentence from chunk 1 to a sentence from chunk 3 into a
    claim neither document makes. Explicit `[1] source=...` markers with a rule
    between them keep the chunks legible as separate, unrelated things — and
    they also blunt (they do not defeat) injected text inside a chunk that
    tries to pass itself off as instructions from the system, since the model
    can see exactly where the quoted material starts and stops.
    """
    if not sources:
        return "CONTEXT: (nothing relevant was retrieved for this question)"

    blocks = [
        f"[{index}] source={item['source']} (similarity {item['score']:.3f})\n{item['text']}"
        for index, item in enumerate(sources, start=1)
    ]
    return "CONTEXT:\n" + "\n---\n".join(blocks)


def _build_contents(
    user_text: str,
    sources: list[dict],
    recent_turns: list[dict],
) -> list[types.Content]:
    """Assemble the conversation to send, oldest first.

    The recent window from `SessionState.recent_turns()` is replayed as real
    `user` / `model` turns rather than being flattened into a transcript
    pasted inside the final prompt. That is what the model was trained on: a
    turn under role `"model"` is understood as something it said and is bound
    by, whereas the same text quoted inside a user message is merely something
    the user claims it said — a distinction that shows up as the model
    contradicting itself, or agreeing that it promised something it did not.

    The retrieved context rides on the FINAL user message, immediately above
    the question, rather than in the system instruction. Context is per-turn
    and the system instruction is not; putting today's chunks in a standing
    instruction invites the model to treat last turn's retrieved passage as a
    permanent fact about the world.
    """
    contents: list[types.Content] = []

    for turn in recent_turns:
        contents.append(
            types.Content(role="user", parts=[types.Part(text=turn.get("user", ""))])
        )
        contents.append(
            types.Content(role="model", parts=[types.Part(text=turn.get("assistant", ""))])
        )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=f"{_format_context(sources)}\n\nQUESTION: {user_text}")],
        )
    )

    return contents


def _extract_text(response: types.GenerateContentResponse) -> str:
    """Pull the model's prose out of a response, ignoring its function-call parts.

    Walks `candidates[0].content.parts` and joins the `text` of each, rather
    than reading `response.text`. One response can legitimately hold both a
    sentence and a function call — "let me look that up" plus the `web_search`
    call — and this has to be able to keep the sentence while the loop
    separately dispatches the call.

    Returns an empty string for a response that is only function calls, or one
    with no candidate at all (which a safety block produces), so every caller
    can treat "no answer yet" uniformly rather than testing for None.
    """
    if not response.candidates:
        return ""

    content = response.candidates[0].content
    if content is None or not content.parts:
        return ""

    return "".join(part.text for part in content.parts if part.text).strip()


async def _run_tool(tools: dict[str, Any], name: str, args: dict[str, Any]) -> Any:
    """Dispatch one model-requested tool call, converting every failure into text.

    Three things can go wrong, and all three come back as a string the model
    can read instead of an exception that ends the turn:

      * THE NAME DOES NOT EXIST. Models hallucinate tool names, particularly
        plausible ones (`search_web`, `get_time`). The reply names the tools
        that DO exist, which is usually enough for the model to correct itself
        on the next pass.
      * THE ARGUMENTS DO NOT FIT. The model fills these in from a schema and
        can omit a required one or invent an extra, which surfaces here as a
        `TypeError` from the call itself. Reported as text, again so the model
        can retry with the right shape.
      * THE TOOL RAISED. It should not — `tools.py`'s first rule is that no
        tool raises — but this is the boundary where that convention would be
        violated, and a broken tool must not be able to take the whole turn
        down with it.
    """
    tool = tools.get(name)
    if tool is None:
        logger.warning(
            "Model requested unknown tool %r — available: %s.", name, sorted(tools)
        )
        return (
            f"Error: there is no tool named {name!r}. The available tools are: "
            f"{', '.join(sorted(tools))}. Call one of those, or answer without a tool."
        )

    try:
        return await tool(**args)
    except TypeError as error:
        logger.warning("Tool %r rejected arguments %s: %s", name, args, error)
        return f"Error: the arguments given to {name} were not valid: {error}"
    except Exception as error:
        logger.error("Tool %r raised unexpectedly: %s", name, error, exc_info=True)
        return f"Error: the tool {name} failed: {type(error).__name__}: {error}"


async def _remember(session_id: str, user_text: str, answer_text: str) -> None:
    """Write the finished exchange to both halves of the memory design.

    `memory.add_turn` embeds the pair and stores it in Atlas; `record_turn`
    appends it to the session's bounded recent window. Both, always, and in
    that order — `session_manager.py`'s module docstring explains why either
    alone is broken: the deque is the exact chronological window that makes
    pronouns and "the other one" resolvable, and the vector store is the
    unbounded half that lets a turn evicted from the deque still be found by
    meaning forty turns later.

    THIS IS WHAT MAKES THE NEXT TURN'S RECALL WORK, and it is why the cost is
    accepted. The write is not free: `add_turn` calls the embedding API, so
    every answer carries one extra round trip. It is spent AFTER the answer
    has been produced and emitted, so it lands in the gap while the user is
    listening rather than while they are waiting — but it is a real cost, paid
    on every turn, to buy retrieval on a later turn that may never come. The
    alternative is an agent that cannot answer "what did I say about the
    invoice earlier?", which is most of the point of the project.

    Failure is logged and swallowed. A memory write that fails costs the user
    nothing they can perceive right now; raising here would cost them the
    answer they have already been given.
    """
    try:
        await memory.add_turn(session_id, user_text, answer_text)
    except Exception as error:
        logger.warning(
            "Could not store the exchange for session %s in semantic memory (%s: %s) — "
            "this turn will not be recallable later.",
            session_id,
            type(error).__name__,
            error,
            exc_info=True,
        )

    manager = get_session_manager()
    state = manager.get(session_id)
    if state is not None:
        state.record_turn(user_text, answer_text)


async def answer(session_id: str, user_text: str, language: str = "en") -> str:
    """Produce one spoken reply to one user turn. Never raises.

    The whole turn, in order: retrieve context, build the prompt, run the
    bounded function-calling loop, emit the answer, persist the exchange.
    Every event the dashboard needs is emitted through
    `SessionManager.emit` as it happens rather than being batched at the end,
    so the activity feed fills in while the model is still working — the
    reason `tool_call` is emitted BEFORE the tool runs and `tool_result` after
    it, instead of one combined event afterwards.

    Events emitted, with the shapes `frontend/app.js` parses:
      * `retrieval`    — `sources: [{text, source, score}]`
      * `tool_call`    — `{name, args}`
      * `tool_result`  — `{name, result}`
      * `agent_answer` — `{text, language}`

    `agent_answer` is emitted here, not by the caller. The WebSocket handler's
    remaining job for this turn is the audio: for a non-English `language` the
    frontend sets `awaitingRegionalAudio` on receiving `agent_answer` and waits
    for a `tts_audio` event, so the handler must emit that AFTER this returns
    and must not emit a second `agent_answer` of its own.

    Args:
        session_id: The live session. Scopes memory recall and binds the
            reminder tools; see `tools.build_tools`.
        user_text: The transcript of what the user just said.
        language: The language code detected by `app/audio/stt.py`. Carried
            into the system instruction so the reply comes back in the same
            language, which is what makes the Sarvam TTS path produce regional
            speech rather than an English sentence read with an accent.

    Returns:
        The answer to speak. Always a non-empty string — an apology if the
        model could not be reached at all, a note if the tool loop hit
        `config.MAX_TOOL_ITERATIONS`.
    """
    manager = get_session_manager()

    if not user_text or not user_text.strip():
        logger.info("Empty transcript for session %s — nothing to answer.", session_id)
        return ""

    user_text = user_text.strip()

    sources = await _retrieve(session_id, user_text)
    manager.emit(session_id, {"type": "retrieval", "sources": sources})

    state = manager.get(session_id)
    recent_turns = state.recent_turns() if state is not None else []

    system_instruction = SYSTEM_INSTRUCTION.format(language_name=_language_name(language))
    contents = _build_contents(user_text, sources, recent_turns)
    tools = build_tools(session_id)

    answer_text = ""

    try:
        for iteration in range(config.MAX_TOOL_ITERATIONS):
            response = await _generate_with_fallback(contents, system_instruction)

            text = _extract_text(response)
            if text:
                answer_text = text

            function_calls = response.function_calls
            if not function_calls:
                break

            logger.info(
                "Session %s: model requested %d tool call(s) on iteration %d/%d.",
                session_id,
                len(function_calls),
                iteration + 1,
                config.MAX_TOOL_ITERATIONS,
            )

            response_parts: list[types.Part] = []
            for call in function_calls:
                name = call.name or ""
                args = dict(call.args or {})

                manager.emit(session_id, {"type": "tool_call", "name": name, "args": args})

                result = await _run_tool(tools, name, args)

                manager.emit(
                    session_id, {"type": "tool_result", "name": name, "result": result}
                )

                response_parts.append(
                    types.Part.from_function_response(name=name, response={"result": result})
                )

            contents.append(response.candidates[0].content)
            contents.append(types.Content(role="user", parts=response_parts))
        else:
            logger.warning(
                "Session %s hit MAX_TOOL_ITERATIONS (%d) without a final answer — "
                "stopping the loop and replying with what is available. The model kept "
                "requesting tools; continuing would be an unbounded, billed loop.",
                session_id,
                config.MAX_TOOL_ITERATIONS,
            )
            if not answer_text:
                answer_text = TOOL_LIMIT_NOTE
    except Exception as error:
        logger.error(
            "The agent turn for session %s failed entirely (%s: %s) — replying with an "
            "apology rather than raising into the WebSocket handler.",
            session_id,
            type(error).__name__,
            error,
            exc_info=True,
        )
        answer_text = FALLBACK_ANSWER

    if not answer_text:
        logger.warning(
            "Session %s: the model returned no text at all (a safety block or an empty "
            "candidate) — substituting the apology.",
            session_id,
        )
        answer_text = FALLBACK_ANSWER

    manager.emit(
        session_id, {"type": "agent_answer", "text": answer_text, "language": language}
    )

    await _remember(session_id, user_text, answer_text)

    return answer_text
