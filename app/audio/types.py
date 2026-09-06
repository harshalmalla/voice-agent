"""
Shared types for the `app.audio.providers` package.

`TranscriptionResult` is the common contract every STT provider returns —
today `faster_whisper_stt.py` (local, synchronous), soon `sarvam.py`
(remote, async) too. Defining it once here, instead of inside one
provider's module, is what lets the orchestrator (`app/audio/stt.py`) treat
every provider as interchangeable: it can call whichever provider it chose
for a given chunk and get back the exact same shape either way, without
caring which vendor actually produced it (Liskov substitution — any
provider implementing this contract can stand in for any other).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """The piece of a transcription result this app actually needs.

    Each provider's own SDK/API response carries a lot more than this (e.g.
    faster-whisper's per-segment timestamps and token ids, Sarvam's request
    metadata, ...) that nothing here uses yet. Wrapping the bits every
    provider agrees on in this one small dataclass keeps the rest of the
    app decoupled from any single vendor's response type — callers import
    `TranscriptionResult` from here, not from `faster_whisper` or from
    Sarvam's response model.
    """

    text: str
    language: str
    language_probability: float

    # Which engine actually produced this transcript — "faster-whisper" or
    # "sarvam" today. The orchestrator (app/audio/stt.py) can route the same
    # audio to either provider depending on the detected language, so by the
    # time a result reaches the WebSocket handler there is otherwise no way
    # to tell which one it came from. Carrying it here makes the routing
    # decision *observable*: the live dashboard can label a transcript with
    # its engine, and a bug report ("this Hindi came back as gibberish") can
    # be answered from the payload rather than by re-reading server logs.
    #
    # Deliberately REQUIRED — no default. A default like "unknown" would let
    # a future provider be added, forget to identify itself, and silently
    # emit mislabelled results that nobody notices until someone is trying
    # to debug quality differences between engines. With no default, Python
    # raises a TypeError at the construction site the moment a new provider
    # is written, which is exactly where the fix belongs.
    provider: str
