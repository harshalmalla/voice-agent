"""
Live reminders: background asyncio tasks that push a `reminder_fired` event
into a session's outgoing queue when they come due.

This is what makes a reminder feel live rather than polled. Nothing on the
client asks "is anything due yet?"; the task sleeps for exactly the requested
delay and then emits, and the WebSocket send loop delivers it into a
conversation that has moved on to something else entirely. The frontend
handles it with `showBanner` + `speak` (see `reminder_fired` in
`frontend/app.js`), so the user hears it out loud.

--- THE STRONG REFERENCE, which is the whole reason this module is careful ---
`asyncio.create_task()` returns a Task, and it is tempting to ignore the return
value: the loop is running it, so surely it will finish. It may not. The event
loop keeps only a WEAK reference to a scheduled task. If the only strong
reference is the local variable inside the function that created it, that
reference dies when the function returns, and the task becomes eligible for
garbage collection. A task collected mid-`await` is simply destroyed — it never
resumes, never fires, and never errors. You get a reminder that silently does
not happen, at a rate that depends on when the GC happens to run, which is to
say: not in any test you write, and reliably in front of an audience.

This is a known asyncio footgun, called out in CPython's own documentation for
`asyncio.create_task`: keep a reference to the task for the duration of its
life. So every task created here is inserted into
`SessionState.reminder_tasks` (the strong reference), and removed again by an
`add_done_callback` once it has finished, so the set does not accumulate
completed tasks for the life of the session. Storing the task on the session
rather than in a module-level set has a second payoff: closing a session
already cancels exactly the tasks belonging to it, with no per-session
bookkeeping in this module at all.

--- Cancellation policy ---
`asyncio.CancelledError` is re-raised, never swallowed. It inherits from
`BaseException` rather than `Exception` precisely so that a blanket
`except Exception` does not eat it, and a task that catches it and returns
normally reports itself as *completed* to whoever is awaiting its
cancellation — so `SessionManager.close()` would believe teardown succeeded
while the task carried on. Catching it to log and then re-raising, as
`_run_reminder` does, is the one legitimate pattern.

--- Known limitation: reminders do not survive a restart ---
Everything here lives in process memory. Restart the server and every pending
reminder is gone, with no error and no notice to the user who set it. This is
the accepted v1 tradeoff recorded in PLAN.md (verification step 9: "in-memory
reminders are lost on restart — note this as a known v1 limitation, not a bug
to fix now"). Making them durable is not a small change dressed up as a big
one: it needs the reminder persisted at schedule time, a rehydration pass on
startup that re-arms every future reminder and decides what to do with the ones
that came due while the process was down, and — since a session's WebSocket
does not survive a restart either — somewhere for a fired reminder to go when
its session no longer exists. That is a feature, not a fix, and pretending
otherwise in a docstring would be the worse sin.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app import config
from app.services.session_manager import get_session_manager

logger = logging.getLogger(__name__)


REMINDER_EVENT_TYPE = "reminder_fired"


def schedule_reminder(session_id: str, delay_seconds: float, message: str) -> str:
    """Arrange for `message` to be pushed to `session_id` in `delay_seconds`.

    Returns as soon as the task is created — it does not wait for the reminder
    to come due, which is the point: the agent turn that set the reminder
    continues and answers the user immediately.

    Synchronous rather than a coroutine because it performs no I/O and does not
    await; it only registers a task. It does, however, require a RUNNING event
    loop, since `asyncio.create_task` has nowhere to schedule otherwise. Inside
    the WebSocket handler and the agent's tools that is always true.

    Args:
        session_id: The session that will receive the event. Must be live; a
            reminder for a session that does not exist could never be
            delivered, so it is rejected up front rather than created and
            silently dropped later.
        delay_seconds: How long to wait, in seconds.
        message: What to say when it fires. Reaches the user as spoken text.

    Returns:
        The reminder id, for `cancel_reminder` and for correlating the fired
        event.

    Raises:
        ValueError: if the session is unknown, the message is blank, or the
            delay is negative, non-finite, or beyond
            `config.MAX_REMINDER_DELAY_SECONDS`. This module is
            infrastructure and follows the codebase's raise-don't-swallow
            policy; `app/agent/tools.py` is the layer that turns these into
            sentences the model can read.
        RuntimeError: if there is no running event loop.
    """
    state = get_session_manager().get(session_id)
    if state is None:
        raise ValueError(
            f"Cannot schedule a reminder for session {session_id!r} — no such live "
            "session, so the event could never be delivered."
        )

    if not message or not message.strip():
        raise ValueError("A reminder needs a non-empty message — there would be nothing to say.")

    delay = _validate_delay(delay_seconds)
    reminder_id = uuid.uuid4().hex[:8]
    due_at = datetime.now(timezone.utc) + timedelta(seconds=delay)

    task = asyncio.create_task(
        _run_reminder(session_id, reminder_id, delay, message.strip()),
        name=f"reminder-{session_id}-{reminder_id}",
    )

    state.reminder_tasks.add(task)
    state.reminders[reminder_id] = {
        "id": reminder_id,
        "message": message.strip(),
        "due_at": due_at.isoformat(),
        "delay_seconds": delay,
        "task": task,
    }

    task.add_done_callback(
        lambda finished: _forget_reminder(session_id, reminder_id, finished)
    )

    logger.info(
        "Scheduled reminder %s for session %s in %.1fs: %r",
        reminder_id,
        session_id,
        delay,
        message.strip(),
    )
    return reminder_id


def _validate_delay(delay_seconds: float) -> float:
    """Coerce and sanity-check a delay, or explain what is wrong with it.

    Three separate rejections, because they are three different mistakes:

      * NOT A NUMBER. The argument arrives from an LLM's function call, where
        "in five minutes" can come back as the string "300" or as "five".
        `float()` handles the first; the second must be refused clearly enough
        that the model can ask the user or retry with a number.
      * NEGATIVE, NaN or infinite. `asyncio.sleep` treats a negative delay as
        zero, so a reminder for "-60 seconds" would fire instantly instead of
        failing — a wrong answer delivered confidently. NaN comparisons are
        all false, so it would slip past a naive upper-bound check and then
        make `sleep` raise from inside the task where nobody sees it.
      * ABSURDLY LARGE. A reminder further out than
        `config.MAX_REMINDER_DELAY_SECONDS` is a task that will sit in memory
        for that long and, given the in-memory limitation documented at the
        top of this module, will not survive to fire anyway. Refusing it is
        more honest than accepting it and losing it.
    """
    try:
        delay = float(delay_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Reminder delay must be a number of seconds, got {delay_seconds!r}."
        ) from exc

    if delay != delay or delay in (float("inf"), float("-inf")):
        raise ValueError(f"Reminder delay must be a finite number of seconds, got {delay!r}.")

    if delay < 0:
        raise ValueError(
            f"Reminder delay must not be negative, got {delay}. A reminder cannot be "
            "set in the past."
        )

    if delay > config.MAX_REMINDER_DELAY_SECONDS:
        raise ValueError(
            f"Reminder delay of {delay:.0f}s exceeds the maximum of "
            f"{config.MAX_REMINDER_DELAY_SECONDS}s."
        )

    return delay


async def _run_reminder(
    session_id: str, reminder_id: str, delay_seconds: float, message: str
) -> None:
    """Sleep for the delay, then emit the reminder into the session's queue.

    The body of every scheduled task. It is intentionally almost empty: the
    less that happens between the sleep and the emit, the fewer ways a reminder
    has to not arrive.

    `SessionManager.emit` is used rather than any direct socket access, which
    makes the disconnected-session case a no-op by construction — the session
    is gone from the registry, `emit` logs and returns False, and this task
    ends normally. Nothing here has to check whether the user is still
    connected, and no traceback escapes into an unawaited task.

    `CancelledError` is caught only to record that it happened, and re-raised
    immediately. See the cancellation note in the module docstring for why
    returning normally here would quietly corrupt `SessionManager.close()`.
    """
    try:
        await asyncio.sleep(delay_seconds)
    except asyncio.CancelledError:
        logger.info("Reminder %s for session %s cancelled before firing.", reminder_id, session_id)
        raise

    delivered = get_session_manager().emit(
        session_id,
        {"type": REMINDER_EVENT_TYPE, "message": message, "reminder_id": reminder_id},
    )

    if delivered:
        logger.info("Reminder %s fired for session %s: %r", reminder_id, session_id, message)
    else:
        logger.info(
            "Reminder %s came due for session %s but could not be delivered — the session "
            "is no longer live. Dropping it.",
            reminder_id,
            session_id,
        )


def _forget_reminder(session_id: str, reminder_id: str, task: asyncio.Task[Any]) -> None:
    """Remove a finished reminder from the session, whatever its outcome.

    Registered with `task.add_done_callback`, so it runs for a task that fired,
    one that was cancelled, and one that raised — all three are "no longer
    pending", and leaving any of them in `reminder_tasks` would make
    `list_reminders` lie and make the set grow for the life of the session.

    A done callback runs on the event loop after the task completes, and must
    not raise: an exception here is reported by the loop's exception handler
    and swallowed, so it would be invisible. Hence the defensive lookup — by
    the time this runs, `SessionManager.close()` may already have popped the
    session out from under it, which is a normal shutdown ordering, not an
    error.

    An unexpected exception inside a reminder task is retrieved and logged
    here. Without that, the task's exception is never consumed and asyncio
    prints "Task exception was never retrieved" at some arbitrary later moment,
    usually with no useful context about which reminder it was.
    """
    state = get_session_manager().get(session_id)
    if state is not None:
        state.reminder_tasks.discard(task)
        state.reminders.pop(reminder_id, None)

    if task.cancelled():
        return

    error = task.exception()
    if error is not None:
        logger.error(
            "Reminder %s for session %s failed: %s: %s",
            reminder_id,
            session_id,
            type(error).__name__,
            error,
            exc_info=error,
        )


def list_reminders(session_id: str) -> list[dict]:
    """Return this session's still-pending reminders, soonest first.

    Reads the registry that `_forget_reminder` prunes, so a reminder that has
    already fired or been cancelled is absent — the list is what is still
    *going to happen*, which is the only question a user asking "what
    reminders do I have?" is asking.

    An unknown session yields an empty list rather than an error: from the
    caller's point of view "you have no reminders" and "you have no session"
    are the same answer, and this is read-only.

    Each record is projected through `_public_fields`, which drops the stored
    `asyncio.Task`. The result of this function travels into a `tool_result`
    WebSocket event and into the model's context, and a Task is neither
    JSON-serializable nor anything a language model can do something with.
    """
    state = get_session_manager().get(session_id)
    if state is None:
        return []

    return [
        _public_fields(reminder)
        for reminder in sorted(state.reminders.values(), key=lambda item: item["due_at"])
    ]


def _public_fields(reminder: dict[str, Any]) -> dict[str, Any]:
    """Strip the internal task handle from a stored reminder record."""
    return {key: value for key, value in reminder.items() if key != "task"}


def cancel_reminder(session_id: str, reminder_id: str) -> bool:
    """Cancel one pending reminder.

    Returns True if a pending reminder was found and cancellation requested,
    False if there was no such reminder — an already-fired id, a typo, or a
    dead session. False is a legitimate answer, not an exception, because the
    user is allowed to ask to cancel something that is not there.

    Note the asymmetry with the return value: `task.cancel()` requests
    cancellation and returns immediately, so True means "asked", not "already
    stopped". The task's own `CancelledError` handling and `_forget_reminder`
    complete the removal a moment later on the event loop. Callers that need
    the task to be genuinely finished must await it — that is what
    `SessionManager.close()` does, and it is the reason it gathers.
    """
    state = get_session_manager().get(session_id)
    if state is None:
        return False

    reminder = state.reminders.get(reminder_id)
    if reminder is None:
        return False

    task = reminder["task"]
    if task.done():
        return False

    task.cancel()
    logger.info("Cancellation requested for reminder %s (session %s).", reminder_id, session_id)
    return True
