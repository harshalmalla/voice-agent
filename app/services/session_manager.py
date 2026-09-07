"""
Per-session state for concurrent WebSocket voice sessions.

One process serves many simultaneous conversations, and every one of them
needs three things that must not be shared with any other: a place to queue
outgoing events, a short conversation history to put in front of the model,
and a handle on the background reminder tasks it has spawned. This module owns
all three, keyed by `session_id`, and is the only place that knows a session
"exists".

--- The event queue, and why events are queued rather than sent -------------
The WebSocket endpoint runs two things at once per connection: a receive loop
(audio frames in) and a send loop (JSON events out). Anything that wants to
tell the client something — the agent mid-turn, a reminder task firing twenty
minutes later — pushes a dict into that session's queue and returns
immediately; the send loop drains it. This is the same fan-out shape as
`multiagent copy/main.py`'s `listeners: set[asyncio.Queue]` per job, adapted
from SSE to WebSocket.

Writing to the socket directly from every producer instead would be worse in
two specific ways: concurrent `send()` calls on one WebSocket can interleave
frames and corrupt the stream, and a slow client would block whichever
coroutine happened to be doing the sending — including the agent turn itself.
A queue gives one writer, one ordering, and producers that never block.

--- The bounded history, and why truncation is safe HERE --------------------
`SessionState.history` is a `deque(maxlen=config.MAX_HISTORY_TURNS)`. Once it
is full, appending turn N+1 silently drops turn 1. Two reasons that bound
exists at all: an unbounded per-session list in a long-lived server is a
memory leak with a friendly face, and every extra turn is tokens paid for on
every subsequent request, so an hour-long conversation would grow its own
latency and cost without bound.

Truncation would normally mean the agent forgets. It doesn't here, and that is
the actual memory design of this project rather than a happy accident:

  * this deque is the RECENT window — chronological, cheap, exact. It carries
    the things that only make sense in sequence: pronouns, "that one", "no, the
    other one", the thread of the last minute of conversation.
  * `app/rag/memory.py::recall()` is the LONG-TERM half — semantic, unbounded,
    and searched rather than replayed. Every completed exchange is embedded and
    written to Atlas as it happens, so a turn evicted from this deque is not
    lost; it is merely no longer free. When the user says something related to
    it forty turns later, `recall()` retrieves that exact exchange by meaning
    and it re-enters the prompt.

So the window can be small precisely BECAUSE recall exists, and recall is
worth its embedding call precisely BECAUSE the window is small. Dropping
either half breaks the other: without the deque the agent loses conversational
flow (semantic search does not reliably retrieve "the last thing said"),
and without recall this deque's `maxlen` really would be amnesia.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from app import config

logger = logging.getLogger(__name__)


EVENT_QUEUE_MAXSIZE = 256


@dataclass
class SessionState:
    """Everything one live conversation owns.

    Attributes:
        session_id: The id from the `/ws/{session_id}` path. Also the scope key
            for `memory.recall()`, which is why it must be the same string in
            both places.
        queue: Outgoing events, each a JSON-serializable dict with a `type`
            field the frontend's `eventHandlers` map dispatches on (see
            `frontend/app.js`). Bounded: see `SessionManager.emit`.
        history: The recent-turn window described in the module docstring.
            Bounded by construction — `deque(maxlen=...)` evicts from the left
            on append, so nothing here ever has to remember to trim it.
        reminder_tasks: Strong references to this session's in-flight
            reminder tasks. Held here, and not merely by `asyncio`, because the
            event loop keeps only a WEAK reference to a running task — see the
            docstring of `app/reminders/scheduler.py`, which is the module that
            populates this set.
        reminders: The user-visible record of those same reminders, keyed by
            reminder id. Its dict shape is owned entirely by the scheduler and
            is opaque to this module; it lives on the session so that closing
            the session disposes of it along with everything else, rather than
            leaving a registry elsewhere to leak entries per disconnect.
    """

    session_id: str
    queue: asyncio.Queue[dict] = field(
        default_factory=lambda: asyncio.Queue(maxsize=EVENT_QUEUE_MAXSIZE)
    )
    history: deque[dict] = field(
        default_factory=lambda: deque(maxlen=config.MAX_HISTORY_TURNS)
    )
    reminder_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    reminders: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        """Append one completed exchange to the recent-turn window.

        Called with the pair, after the agent has answered — the same unit
        `memory.add_turn()` stores, deliberately, so the two halves of the
        memory design stay in step and a turn cannot be in one and not the
        other.

        Eviction of the oldest turn happens here, silently and by design; the
        module docstring explains why that is not data loss.
        """
        self.history.append({"user": user_text, "assistant": assistant_text})

    def recent_turns(self) -> list[dict]:
        """Return the window as a plain list, oldest first.

        A copy, not the live deque: the caller is usually building a prompt,
        and handing it a mutable view of session state that another coroutine
        may append to mid-iteration is a race waiting to be written.
        """
        return list(self.history)


class SessionManager:
    """The registry of live sessions.

    Deliberately a plain dict with no locking. Every method here is called from
    the single asyncio event loop that serves every WebSocket, and none of them
    awaits between reading and writing `_sessions` — so each is atomic with
    respect to the other coroutines by construction. This is the opposite of
    the choice in `app/rag/store.py`, whose client singleton IS lock-guarded
    because it can also be built from a synchronous ingest script on another
    thread. Nothing constructs sessions off-loop.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        """Return this session's state, creating it on first sight.

        Idempotent: calling it twice with the same id returns the identical
        object, never a second one. That matters because the browser
        reconnects with the same `session_id` after a dropped socket (see
        `scheduleReconnect` in `frontend/app.js`), and a reconnect that
        silently replaced the state would drop the conversation window and
        orphan the pending reminder tasks — the reminders would still fire,
        into a queue nobody was reading.

        Raises:
            ValueError: if `session_id` is empty. An unkeyed session would be
                indistinguishable from every other unkeyed session, and its
                memory would be recalled into strangers' conversations.
        """
        if not session_id:
            raise ValueError("get_or_create() requires a non-empty session_id.")

        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing

        state = SessionState(session_id=session_id)
        self._sessions[session_id] = state
        logger.info(
            "Session %s opened (%d live session(s), history window=%d turns).",
            session_id,
            len(self._sessions),
            config.MAX_HISTORY_TURNS,
        )
        return state

    def get(self, session_id: str) -> SessionState | None:
        """Return this session's state, or None if it is not (or no longer) live.

        The None return is the whole point: background work started during a
        session routinely finishes after the user has gone, and asking "is this
        session still here?" must be answerable without an exception.
        """
        return self._sessions.get(session_id)

    def emit(self, session_id: str, event: dict) -> bool:
        """Queue one outgoing event for a session. Never raises.

        This is the seam every background producer pushes through, and it is
        written to be un-losable rather than correct-or-loud, because both of
        its failure modes are ordinary rather than exceptional:

          * THE SESSION IS GONE. A reminder scheduled for twenty minutes'
            time fires ten minutes after the user closed the tab. That is a
            race, not a bug, and there is nothing to do about it and nobody to
            tell — so it is logged at debug and dropped. Raising instead would
            turn a normal disconnect into a traceback inside an
            `asyncio.Task` that nobody awaits, which surfaces as
            "Task exception was never retrieved" and looks like a crash.
          * THE QUEUE IS FULL. The queue is bounded (`EVENT_QUEUE_MAXSIZE`)
            because an unbounded one turns a stalled client into unbounded
            server memory. `put_nowait` + drop is chosen over `await put()`
            precisely so that a wedged client cannot apply backpressure to the
            agent turn or to a reminder task; a dropped event is logged at
            warning because, unlike a dead session, it means something is
            actually wrong.

        Args:
            session_id: The session to notify.
            event: A JSON-serializable dict carrying a `type` the frontend
                knows — `transcript`, `retrieval`, `tool_call`, `tool_result`,
                `agent_answer`, `tts_audio`, `reminder_fired`, `error`.

        Returns:
            True if the event was queued; False if it was dropped.
        """
        state = self._sessions.get(session_id)
        if state is None:
            logger.debug(
                "Dropping %r event for session %s — session is not live.",
                event.get("type"),
                session_id,
            )
            return False

        try:
            state.queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "Dropping %r event for session %s — outgoing queue is full at %d "
                "events. The client is not draining it (slow or wedged connection).",
                event.get("type"),
                session_id,
                EVENT_QUEUE_MAXSIZE,
            )
            return False

        return True

    async def close(self, session_id: str) -> None:
        """Retire a session and everything still running on its behalf.

        Called from the WebSocket handler's `finally` block, so it must
        tolerate being called for a session that was never created or has
        already been closed — a connection that failed during handshake takes
        exactly that path.

        Order matters. The session is removed from the registry FIRST, so that
        any reminder task which wins the race and fires during teardown finds
        no session in `emit()` and drops its event harmlessly instead of
        appending to a queue that is being discarded.

        Cancelled tasks are then awaited rather than merely cancelled.
        `task.cancel()` only *requests* cancellation: it schedules a
        `CancelledError` to be raised at the task's next suspension point and
        returns immediately. Walking away at that moment leaves tasks that are
        still nominally running as the loop shuts down, which is the usual
        source of "Task was destroyed but it is pending" on exit. Gathering
        with `return_exceptions=True` waits for each to actually unwind and
        keeps one task's failure from hiding the rest.

        Draining the queue afterwards is not about freeing the dicts — dropping
        the last reference to the queue does that — but about not leaving a
        half-consumed queue reachable from a task that is still finishing.
        """
        state = self._sessions.pop(session_id, None)
        if state is None:
            logger.debug("close() called for unknown session %s — nothing to do.", session_id)
            return

        pending = [task for task in state.reminder_tasks if not task.done()]
        for task in pending:
            task.cancel()

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        state.reminder_tasks.clear()
        state.reminders.clear()

        drained = 0
        while not state.queue.empty():
            state.queue.get_nowait()
            drained += 1

        logger.info(
            "Session %s closed (%d reminder task(s) cancelled, %d undelivered event(s) "
            "discarded, %d session(s) still live).",
            session_id,
            len(pending),
            drained,
            len(self._sessions),
        )

    def live_session_ids(self) -> list[str]:
        """Snapshot of the currently live session ids, for logging and health checks."""
        return list(self._sessions)


_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Return the process-wide session registry, building it on first use.

    A module-level singleton because "which sessions are live" is genuinely
    process-global state: the WebSocket handler, the agent's tools and the
    reminder tasks must all be looking at the same dict, and threading one
    manager instance through every call site would buy nothing but ceremony.

    Lazily constructed for the same reason `store._get_client()` is — importing
    a module should not build application state — though here the construction
    is free, so the laziness is mostly about keeping the pattern uniform and
    about letting a test reset the singleton between cases.
    """
    global _manager

    if _manager is None:
        _manager = SessionManager()

    return _manager
