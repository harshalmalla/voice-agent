"""
Local speech-to-text provider, backed by faster-whisper.

This module wraps a single `faster_whisper.WhisperModel` and exposes one
function, `transcribe()`. It is a *provider* — the future orchestrator
(`app/audio/stt.py`) calls this first for every incoming audio chunk, for
two reasons at once:

1. It's the primary transcript source for English audio (fast, free, local).
2. Its `language`/`language_probability` output is the signal that decides
   whether to route the *same* audio to Sarvam for a regional-language
   retranscription (see PLAN.md's "Regional language routing" section).
   faster-whisper's non-English transcription quality is weak, but its
   language *detection* is good — so we always trust the language it
   reports even when we throw away its non-English transcript text.

API confirmed against the actually-installed package (not from memory):
`WhisperModel.transcribe(audio, ...)` accepts a `str` path, a `BinaryIO`,
or a `numpy.ndarray` — never raw `bytes`. Under the hood it hands `audio`
straight to PyAV's `av.open(...)` (see faster_whisper/audio.py), which can
open a file-like object directly and demux/decode whatever container is
inside it (webm/opus, wav, mp3, ...) — there's no need to know the codec
up front or write anything to a temp file. So a browser MediaRecorder blob
(commonly a webm/opus container) just needs wrapping in `io.BytesIO` before
being passed in.
"""

from __future__ import annotations

import io
import logging
import threading

from faster_whisper import WhisperModel

from app.audio.types import TranscriptionResult
from app.config import WHISPER_COMPUTE_TYPE, WHISPER_MODEL_SIZE

logger = logging.getLogger(__name__)

PROVIDER_NAME = "faster-whisper"


_model: WhisperModel | None = None

_model_lock = threading.Lock()


def _get_model() -> WhisperModel:
    """Return the process-wide WhisperModel, creating it on first use.

    Double-checked locking: check once before taking the lock (so the
    overwhelmingly common case — model already loaded — never pays for lock
    acquisition at all), then, only if it looks unloaded, take the lock and
    check *again* before actually constructing it. That second check matters
    because two threads can both read `_model is None` as True before either
    has grabbed the lock; without re-checking inside the lock, both would
    proceed to build their own WhisperModel, wasting memory/load time and
    leaving whichever one finishes last as the "winner" arbitrarily.
    """
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            logger.info(
                "Loading faster-whisper model (size=%s, compute_type=%s)...",
                WHISPER_MODEL_SIZE,
                WHISPER_COMPUTE_TYPE,
            )
            _model = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            logger.info("faster-whisper model loaded.")

    return _model


def transcribe(audio_bytes: bytes) -> TranscriptionResult:
    """Transcribe raw audio bytes and detect their spoken language.

    This function is **synchronous and CPU-bound by design** — faster-whisper
    runs inference on the CPU (see `WHISPER_COMPUTE_TYPE=int8` in config.py),
    and that inference call blocks the calling thread for its full duration
    (anywhere from tens of milliseconds to a few seconds, depending on clip
    length and model size). Callers — the future `app/audio/stt.py`
    orchestrator — MUST invoke this via `asyncio.to_thread(transcribe,
    audio_bytes)`, never `await`ed directly and never called straight from an
    `async def` function's body. This app is a single-process asyncio event
    loop juggling every connected WebSocket session; calling a blocking
    function directly from a coroutine freezes that one event loop thread
    for the whole inference call, which means EVERY other session's
    WebSocket messages, timers, and reminders also stall until it returns.
    Handing it to `asyncio.to_thread` moves the blocking work to a separate
    worker thread so the event loop stays free to service everyone else.

    Args:
        audio_bytes: Raw audio file bytes, e.g. a browser MediaRecorder
            blob (typically a webm/opus container). Passed as bytes rather
            than a file path because this is a live WebSocket service — the
            audio never touches disk.

    Returns:
        A TranscriptionResult with the joined transcript text and the
        detected language + confidence.

    Raises:
        ValueError: if `audio_bytes` is empty.
    """
    if not audio_bytes:
        raise ValueError("transcribe() received empty audio_bytes — nothing to transcribe.")

    model = _get_model()

    segments, info = model.transcribe(io.BytesIO(audio_bytes))

    full_text = " ".join(segment.text.strip() for segment in segments)

    logger.debug(
        "Transcription detected language=%s (probability=%.2f)",
        info.language,
        info.language_probability,
    )

    return TranscriptionResult(
        text=full_text,
        language=info.language,
        language_probability=info.language_probability,
        provider=PROVIDER_NAME,
    )
