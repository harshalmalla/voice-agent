"""
The FastAPI application: process lifecycle, the WebSocket voice endpoint, and
the static frontend.

This is the outermost layer of the app and the only module that knows a network
exists. Everything below it — `app/audio/stt.py`, `app/agent/gemini_agent.py`,
`app/rag/*`, `app/reminders/scheduler.py` — is callable from a test or a script
with no socket in sight; this file is what wires those pieces to a browser.

It owns three things and deliberately nothing else:

  * PROCESS LIFECYCLE. Logging configured once, the vector indexes ensured
    once at boot, the Mongo client closed once on the way out (see `lifespan`).
  * ONE TURN'S CHOREOGRAPHY. Which module runs in which order for a single
    user utterance, and what happens when one of them fails (see
    `_process_audio_turn` / `_process_text_turn`).
  * THE SOCKET ITSELF. Two concurrent loops per connection, and the teardown
    that guarantees neither outlives the other (see `voice_session`).

--- The wire protocol, which is fixed by the client ------------------------
`frontend/app.js` is already written, so this file matches it rather than the
other way round. The client connects to `ws://<host>/ws/{session_id}` and:

  * sends a recorded utterance as a single BINARY frame (a MediaRecorder blob,
    typically webm/opus — nothing here decodes it, it goes straight to STT);
  * sends `{"type": "text_query", "text": ...}` as a JSON TEXT frame;
  * dispatches inbound JSON on a `type` field, through its `eventHandlers`
    map: `transcript` `{text, language, provider}`, `retrieval`
    `{sources: [{text, source, score}]}`, `tool_call` `{name, args}`,
    `tool_result` `{name, result}`, `agent_answer` `{text, language}`,
    `tts_audio` `{audio_base64}`, `reminder_fired` `{message}`, `error`
    `{message}`. An unknown `type` is logged and dropped by the client, so a
    typo here is silent — hence the shapes are stated once, here, and produced
    nowhere else.

Note which of those events this module does NOT emit. `retrieval`,
`tool_call`, `tool_result` and `agent_answer` are emitted by
`gemini_agent.answer()` while the turn is still running; `reminder_fired` is
emitted by a background task in `app/reminders/scheduler.py`, possibly hours
later. This module emits only `transcript`, `tts_audio` and `error`. That
split is the whole reason for the two-loop design below.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config
from app.agent import gemini_agent
from app.audio import stt
from app.audio.providers import sarvam
from app.audio.providers.sarvam import SarvamAPIError
from app.logging_config import setup_logging
from app.rag import store
from app.services.session_manager import SessionState, get_session_manager

logger = logging.getLogger(__name__)


FRONTEND_DIR = config.BASE_DIR / "frontend"

ENGLISH = "en"

TYPED_PROVIDER = "typed"

EMPTY_TRANSCRIPT_MESSAGE = (
    "I did not catch any speech in that recording. Hold the button while you talk, "
    "then release it when you are done."
)

TURN_FAILED_MESSAGE = (
    "Something went wrong while handling that. The connection is still open, so please try again."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop the process-wide resources exactly once.

    The `lifespan` context manager passed to `FastAPI(lifespan=...)` is the
    current API, verified against the installed FastAPI 0.141.1 / Starlette
    1.6.0: `FastAPI.__init__` accepts `lifespan: Callable[[AppType],
    AbstractAsyncContextManager[None]] | ...`, while `FastAPI.on_event` is
    decorated with `@deprecated("on_event is deprecated, use lifespan event
    handlers instead")`. Everything before the `yield` runs before the server
    accepts its first connection; everything after runs once the last one has
    drained.

    STARTUP does two things. `setup_logging()` first, so that anything the
    index check logs is already formatted and levelled rather than lost.
    Then `store.ensure_vector_index` for both collections — the documents
    corpus and the conversation memory — because a missing Atlas vector index
    is the single most common reason retrieval silently returns nothing, and
    finding out at boot beats finding out mid-conversation. That call is
    best-effort by contract: on a cluster tier that forbids programmatic index
    management it logs the JSON to paste into the Atlas UI and returns False.
    It is wrapped anyway, because it DOES raise when `MONGODB_URI` is unset,
    and a developer who has not filled in `.env` yet should still get a server
    that boots and serves the frontend rather than a stack trace on startup.

    SHUTDOWN closes the Mongo client. `AsyncMongoClient.close()` is a
    coroutine in the installed pymongo and stops the topology-monitor task;
    skipping it is what produces "Task was destroyed but it is pending" on
    exit, and leaks the connection pool's sockets under a reloading dev server.
    """
    setup_logging()
    logger.info("Live Voice RAG Agent starting up.")

    for collection_name in (config.DOCUMENTS_COLLECTION, config.MEMORY_COLLECTION):
        try:
            await store.ensure_vector_index(collection_name)
        except Exception as error:
            logger.warning(
                "Could not ensure the vector index on %r at startup (%s: %s). The server will "
                "still run; retrieval against that collection will fail until it exists.",
                collection_name,
                type(error).__name__,
                error,
            )

    yield

    logger.info("Live Voice RAG Agent shutting down.")
    try:
        await store.close_client()
    except Exception as error:
        logger.warning(
            "Closing the MongoDB client failed (%s: %s) — the process is exiting anyway.",
            type(error).__name__,
            error,
        )


app = FastAPI(
    title="Live Voice RAG Agent",
    description="A real-time voice assistant with retrieval, tools and reminders.",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""Permissive CORS, for local development only.

The frontend is served by this same app from the same origin, so in the normal
setup CORS is not exercised at all. It is opened up for the case that actually
comes up while building: running the page from a file, from a Live Server on
port 5500, or from a separate dev server, while the API stays on 8000.

TIGHTEN THIS BEFORE DEPLOYING. `allow_origins=["*"]` means any website a user
visits can call this API with their browser. `allow_credentials` is False here,
which limits the damage (cookies are not sent, and the two settings are
mutually exclusive in the CORS spec anyway), but the endpoints are still open
to any origin. Replace the list with the real deployed origin, e.g.
`allow_origins=["https://voice-agent.example.com"]`. Note that this does NOT
protect the WebSocket: browsers do not apply CORS to WebSocket handshakes, so
an origin check for `/ws/...` has to be written by hand if it is needed.
"""


@app.get("/health")
async def health() -> dict:
    """Liveness probe.

    Trivial by design and deliberately dependency-free: it does not touch
    Atlas, Gemini or Sarvam. A health check that fails when a downstream
    dependency is slow tells a deploy platform to kill and restart a process
    that is perfectly capable of serving the frontend and reporting the
    problem — turning a degraded service into a crash loop. This answers only
    "is the process up and serving HTTP", which is the question the platform
    is actually asking.
    """
    manager = get_session_manager()
    return {"status": "ok", "live_sessions": len(manager.live_session_ids())}


async def _speak(session_id: str, text: str, language: str) -> None:
    """Synthesize `text` through Sarvam and emit it as a `tts_audio` event.

    Only called for non-English answers. English is spoken by the browser's own
    `speechSynthesis` — `frontend/app.js`'s `agent_answer` handler calls
    `speak()` immediately for `language === "en"` and sets
    `awaitingRegionalAudio` otherwise — so paying for a Sarvam call on the
    common case would buy nothing but latency and cost.

    A `SarvamAPIError` HERE IS NOT A FAILED TURN, and that is the whole point
    of this function existing separately from the turn handlers. By the time
    synthesis runs, `gemini_agent.answer()` has already emitted `agent_answer`
    and the user is reading the reply on screen. Losing the audio degrades the
    experience; raising would additionally throw away an answer that was
    already delivered, and surface an `error` banner about a turn that
    demonstrably worked. Same policy split as `app/audio/stt.py`: the provider
    raises because it cannot know what a failure means, and this layer, which
    knows a person is mid-conversation, decides to degrade instead.

    The event carries base64 rather than a binary frame because the client
    expects `payload.audio_base64` and `atob`s it — and because the send loop
    writes JSON exclusively, so an interleaved binary frame would have to
    bypass the queue that guarantees one writer per socket.
    """
    manager = get_session_manager()

    try:
        audio = await sarvam.synthesize_speech(text, language)
    except SarvamAPIError as error:
        logger.warning(
            "Sarvam speech synthesis failed for session %s (language=%r): %s. The text answer "
            "was already delivered, so this turn continues without audio; the browser will not "
            "speak it, because a regional answer read by an English voice is worse than silence.",
            session_id,
            language,
            error,
        )
        return
    except Exception as error:
        logger.warning(
            "Unexpected failure synthesizing speech for session %s (%s: %s) — continuing "
            "without audio.",
            session_id,
            type(error).__name__,
            error,
            exc_info=True,
        )
        return

    manager.emit(
        session_id,
        {"type": "tts_audio", "audio_base64": base64.b64encode(audio).decode("ascii")},
    )
    logger.info(
        "Emitted %d bytes of %s speech audio for session %s.", len(audio), language, session_id
    )


async def _answer_and_speak(session_id: str, user_text: str, language: str) -> None:
    """Run the agent for one turn, then add audio if the answer is not English.

    The shared tail of both turn handlers, kept in one place because the order
    of the two steps is a contract rather than a preference.

    `gemini_agent.answer()` emits its own `agent_answer` event before it
    returns — its docstring is explicit that the handler must not emit a second
    one, and that the handler's remaining job is the audio. So this function
    deliberately ignores the returned string for eventing purposes and uses it
    only as the text to synthesize. Emitting `agent_answer` again here would
    resolve the client's pending turn bubble twice and re-trigger browser
    speech on the English path.

    An empty return means the agent had nothing to say (it guards its own empty
    input), which is not worth a synthesis call.
    """
    answer_text = await gemini_agent.answer(session_id, user_text, language)

    if answer_text and _server_speaks(language):
        await _speak(session_id, answer_text, language)


def _server_speaks(language: str) -> bool:
    """Whether this turn's reply is synthesized here rather than in the browser.

    Regional languages always are: the browser's speechSynthesis has patchy
    and inconsistent Indian-language voice support, while Sarvam's is good.
    English is the toggle. The browser path is free, instant and costs no
    quota, but its default voices sound noticeably synthetic; setting
    SARVAM_TTS_FOR_ENGLISH routes English through Sarvam too, trading an API
    call and some latency per reply for a uniformly better voice.

    The client is told which mode is in force when it connects, because it
    cannot infer it: on receiving an answer it must either speak the text
    itself or wait for the audio, and doing both would talk over itself.
    """
    return language != ENGLISH or config.SARVAM_TTS_FOR_ENGLISH


async def _process_audio_turn(session_id: str, audio: bytes) -> None:
    """Handle one recorded utterance: transcribe, answer, speak.

    `stt.transcribe` is the routing layer, not a single engine: it runs
    faster-whisper on every chunk for the transcript AND the language
    detection, then re-transcribes through Sarvam only when the detected
    language is regional and confident. Which engine won comes back on
    `result.provider`, and is forwarded verbatim in the `transcript` event so
    the dashboard shows the route that was actually taken.

    THE EMPTY-TRANSCRIPT GUARD is not a formality. A push-to-talk button
    releases on silence all the time — a mis-click, a mic that never got
    permission, a user who let go early — and the result is a valid audio blob
    that transcribes to "". Passing that to the agent would spend a retrieval
    round trip and a model call on nothing, and `answer()` would return an
    empty string, leaving the client's "Thinking" bubble hanging forever
    because no `agent_answer` ever arrives. An `error` event instead both
    clears that bubble (the client's `error` handler calls
    `resolvePendingAgentTurn(null)`) and tells the user what to do differently.
    """
    manager = get_session_manager()

    logger.info("Session %s: received %d bytes of audio.", session_id, len(audio))

    result = await stt.transcribe(audio)
    text = result.text.strip()

    if not text:
        logger.info(
            "Session %s: transcript was empty (provider=%s, language=%r) — treating as silence.",
            session_id,
            result.provider,
            result.language,
        )
        manager.emit(session_id, {"type": "error", "message": EMPTY_TRANSCRIPT_MESSAGE})
        return

    manager.emit(
        session_id,
        {
            "type": "transcript",
            "text": text,
            "language": result.language,
            "provider": result.provider,
        },
    )

    await _answer_and_speak(session_id, text, result.language)


async def _process_text_turn(session_id: str, text: str) -> None:
    """Handle one typed question: the same turn, minus the transcription.

    Typed input skips STT entirely — there is no audio to transcribe and no
    language to detect — so the language is fixed to English, which also means
    the answer is spoken by the browser rather than by Sarvam.

    A `transcript` event is still emitted, with `provider` of `"typed"`. That
    string is not invented here: `frontend/app.js`'s submit handler renders its
    own optimistic turn with `addUserTurn(text, null, "typed")`, so using the
    same label keeps the server's version of the turn indistinguishable from
    the client's. Emitting the event at all is what keeps the two input paths
    symmetric: the client's `transcript` handler is also what queues the
    "Thinking" bubble the agent's answer later resolves, and every downstream
    consumer of the event stream sees a typed turn exactly as it sees a spoken
    one.
    """
    manager = get_session_manager()

    logger.info("Session %s: received a typed query (%d chars).", session_id, len(text))

    manager.emit(
        session_id,
        {
            "type": "transcript",
            "text": text,
            "language": ENGLISH,
            "provider": TYPED_PROVIDER,
        },
    )

    await _answer_and_speak(session_id, text, ENGLISH)


async def _handle_message(session_id: str, message: dict) -> None:
    """Route one raw ASGI WebSocket message to the turn handler it belongs to.

    Reading the raw message rather than calling `receive_text()` /
    `receive_bytes()` is what makes a single loop able to accept both frame
    kinds: those helpers each assert the frame's type and error on the other,
    but this connection genuinely carries both — binary audio blobs and JSON
    text — and cannot know which is coming next.

    Anything unrecognised is logged and ignored rather than raising. A stray
    frame from a future client version, or a ping payload, must not end a live
    conversation.
    """
    if message.get("bytes") is not None:
        await _process_audio_turn(session_id, message["bytes"])
        return

    raw = message.get("text")
    if raw is None:
        logger.warning("Session %s: received a frame with neither bytes nor text.", session_id)
        return

    try:
        payload = json.loads(raw)
    except ValueError:
        logger.warning("Session %s: ignoring a text frame that is not JSON.", session_id)
        return

    if not isinstance(payload, dict):
        logger.warning("Session %s: ignoring a JSON frame that is not an object.", session_id)
        return

    message_type = payload.get("type")

    if message_type == "text_query":
        text = str(payload.get("text") or "").strip()
        if not text:
            logger.info("Session %s: ignoring an empty text_query.", session_id)
            return
        await _process_text_turn(session_id, text)
        return

    logger.warning("Session %s: ignoring unknown message type %r.", session_id, message_type)


async def _receive_loop(websocket: WebSocket, session_id: str) -> None:
    """Read client frames and process one turn at a time, until the socket ends.

    Turns are processed sequentially — the `await` on the handler blocks the
    next `receive()` — and that is intentional. Two overlapping turns on one
    session would race on the conversation history and interleave their
    `retrieval` / `tool_call` events in the activity feed, so a user cannot tell
    which question produced which. A person cannot talk over themselves anyway.

    EVERY UNEXPECTED EXCEPTION IN A TURN IS CAUGHT HERE, and this is the last
    place it can be. Below this line the codebase already degrades rather than
    raises — the agent apologises instead of failing, Sarvam TTS failures are
    swallowed, retrieval failures answer without context — so an exception
    reaching this point is by definition something nobody anticipated. If it
    propagated, the WebSocket would close, the client would enter its
    exponential-backoff reconnect, and the user would see a connection drop
    with no explanation for what was, most likely, one bad turn. Catching it
    keeps the socket open, logs the traceback for whoever has to fix it, and
    sends the user an `error` event that also clears their pending "Thinking"
    bubble.

    `WebSocketDisconnect` is re-raised rather than caught, because a closed
    socket is not a bad turn: there is nobody left to apologise to, and the
    loop must end so the connection can be torn down.
    """
    manager = get_session_manager()

    while True:
        message = await websocket.receive()

        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(code=message.get("code", 1000))

        try:
            await _handle_message(session_id, message)
        except WebSocketDisconnect:
            raise
        except Exception as error:
            logger.error(
                "Unhandled error processing a turn for session %s (%s: %s) — the connection "
                "stays open for the next one.",
                session_id,
                type(error).__name__,
                error,
                exc_info=True,
            )
            manager.emit(session_id, {"type": "error", "message": TURN_FAILED_MESSAGE})


async def _send_loop(websocket: WebSocket, state: SessionState) -> None:
    """Drain this session's event queue onto the socket, forever.

    The single writer. `SessionManager.emit` is a non-blocking `put_nowait`
    from anywhere in the process, and this is the only coroutine that ever
    calls `send_json` on this socket — which is what keeps concurrent producers
    from interleaving frames, and keeps a slow client from applying
    backpressure to an agent turn or a reminder task (the queue is bounded and
    drops instead; see `session_manager.emit`).
    """
    while True:
        event = await state.queue.get()
        await websocket.send_json(event)


@app.websocket("/ws/{session_id}")
async def voice_session(websocket: WebSocket, session_id: str) -> None:
    """One live conversation, for as long as the browser stays connected.

    --- Why two concurrent loops rather than receive-then-reply -------------
    The obvious shape for a WebSocket handler is a single loop: read a message,
    do the work, write the reply, repeat. It cannot express this application,
    because most of what this app sends is not a reply to anything.

      * A REMINDER FIRES ON ITS OWN CLOCK. `app/reminders/scheduler.py` sleeps
        in a background task and emits `reminder_fired` twenty minutes later,
        while the user is silent. In a receive-then-reply loop that event has
        nowhere to go: the loop is parked in `receive()`, and the only code
        path that writes to the socket runs after a message arrives. The
        reminder would be delivered whenever the user next happened to speak,
        which for a reminder is the same as not delivering it.
      * THE AGENT REPORTS PROGRESS MID-TURN. `gemini_agent.answer()` emits
        `retrieval`, then `tool_call` before each tool runs and `tool_result`
        after it, so the dashboard fills in while the model is still working.
        All of that happens INSIDE the single `await` a receive-then-reply loop
        would be blocked on. Batching those events until the turn finished
        would defeat the point of emitting them separately, which is to show
        the work as it happens.

    So: a receive loop that consumes input and drives turns, and a send loop
    that drains `state.queue` and writes whatever anyone has produced —
    the current turn, a background task, or a reminder from an hour ago. The
    queue is the seam between them, and `session_manager.py`'s docstring covers
    why producers push to it instead of writing to the socket directly.

    --- Why either loop ending must kill the other -------------------------
    The two loops fail independently and neither failure is survivable alone. A
    dead receive loop with a live sender is a connection that can talk but not
    listen; a dead send loop with a live receiver accepts audio, spends money
    transcribing and answering it, and delivers nothing. Worse, each is a leak:
    the surviving task holds the socket, the session state and its reminder
    tasks alive indefinitely, and nothing will ever wake it to notice.

    `asyncio.wait(..., FIRST_COMPLETED)` therefore returns the moment EITHER
    finishes for any reason, the survivor is cancelled, and the cancellation is
    then awaited via `gather` — `cancel()` only requests cancellation, and
    walking away at that point is what produces "Task was destroyed but it is
    pending". The first task's exception is re-raised afterwards so a real bug
    is not silently swallowed by the teardown.

    --- Why the cleanup is in `finally` ------------------------------------
    `manager.close(session_id)` cancels the session's pending reminder tasks and
    drops its state. It must run on every exit path, including the ordinary one
    — the browser closing a tab is the common case, not an error — because
    those reminder tasks are the one thing that outlives the connection on its
    own. `WebSocketDisconnect` is caught and logged at INFO for exactly that
    reason: a disconnect is a normal end to a session, and logging it as an
    error trains people to ignore errors.

    Note that `get_or_create` is idempotent: the browser reconnects with the
    same `session_id` after a dropped socket, and re-entering here must resume
    that session rather than replace it.
    """
    await websocket.accept()

    manager = get_session_manager()
    state = manager.get_or_create(session_id)

    logger.info("WebSocket session %s connected.", session_id)

    manager.emit(
        session_id,
        {
            "type": "session_config",
            "server_tts_for_english": config.SARVAM_TTS_FOR_ENGLISH,
        },
    )

    receive_task = asyncio.create_task(
        _receive_loop(websocket, session_id), name=f"receive-{session_id}"
    )
    send_task = asyncio.create_task(_send_loop(websocket, state), name=f"send-{session_id}")

    try:
        done, pending = await asyncio.wait(
            {receive_task, send_task}, return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        for task in done:
            task.result()
    except WebSocketDisconnect:
        logger.info("WebSocket session %s disconnected by the client.", session_id)
    except Exception as error:
        logger.error(
            "WebSocket session %s ended with an unexpected error (%s: %s).",
            session_id,
            type(error).__name__,
            error,
            exc_info=True,
        )
    finally:
        await manager.close(session_id)
        logger.info("WebSocket session %s torn down.", session_id)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
"""Serve `frontend/` from this same process, at the root.

One process, one port, one origin: the browser loads the page and opens its
WebSocket against the same host, so there is no second server to start, no CORS
preflight in the normal path, and `frontend/app.js`'s
`` `${protocol}//${location.host}/ws/${sessionId()}` `` resolves correctly with
no configuration. `html=True` is what makes `/` serve `index.html` (and gives
the relative `styles.css` / `app.js` references in that file a home).

MOUNTING AT `/` LAST IS LOAD-BEARING, NOT STYLISTIC. Starlette matches routes in
the order they were added and stops at the first match, and a mount at `/`
matches EVERY path. Declared above `/health` and `/ws/{session_id}`, it would
shadow both: health checks would get a 404 from the static handler, and the
WebSocket upgrade would never reach `voice_session` — which surfaces as a
client that connects, immediately disconnects, and retries forever with no
server-side error to explain it. Because this statement sits below both route
declarations, those are matched first and only unmatched paths fall through to
the static files.
"""
