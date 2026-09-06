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

    provider: str
