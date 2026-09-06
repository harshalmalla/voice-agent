"""
Logging setup for the Live Voice RAG Agent.

The two reference projects this app is modeled on (Ecoloop, multiagent) rely
on bare `print()` statements for visibility into what's happening. That's
fine for a short-lived script you watch run once, but this is a long-lived
real-time WebSocket service: multiple client connections, background tasks
(STT, RAG retrieval, TTS) all interleaving, and — unlike a script — nobody
is necessarily staring at the terminal when something goes wrong. `print()`
gives you no timestamp (so you can't tell how long a step took or correlate
it with an external log), no severity (so you can't grep for just the
errors), and no indication of which module produced it. The stdlib
`logging` module solves all three for free, so we use it instead.
"""

from __future__ import annotations

import logging

from app.config import LOG_LEVEL

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def setup_logging() -> None:
    """Configure the root logger once, for the whole process.

    We configure the *root* logger rather than reaching into every module
    and configuring its logger individually, because of how the logging
    module's propagation works: every logger created elsewhere in this app
    via `logging.getLogger(__name__)` has no handlers of its own by
    default, so any record it emits walks up the logger hierarchy to the
    root logger and is handled there. Attach one handler to the root once,
    here, and every module's logger — present and future — automatically
    inherits console output, the level, and the format, with zero setup
    required in that module beyond `logging.getLogger(__name__)`. That's
    much less error-prone than remembering to configure each new module.

    Console-only for now (a single StreamHandler to stdout/stderr) — no
    RotatingFileHandler or `logs/` directory. This is a teaching-sized
    project run locally / in a container where stdout is already captured
    by the platform (or your terminal); a rotating file handler can be
    added later here if the app ever needs on-disk logs it manages itself.
    """
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger()

    numeric_level = getattr(logging, LOG_LEVEL.upper(), None)
    if not isinstance(numeric_level, int):
        root_logger.warning(
            "Invalid LOG_LEVEL '%s'; falling back to INFO. "
            "Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
            LOG_LEVEL,
        )
        numeric_level = logging.INFO

    root_logger.setLevel(numeric_level)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root_logger.addHandler(handler)

    _configured = True
