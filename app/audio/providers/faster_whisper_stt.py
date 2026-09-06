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

# The value this provider stamps onto every TranscriptionResult it returns
# (see `provider` in app/audio/types.py). Named here rather than inlined at
# the construction site so the orchestrator and any dashboard code can
# compare against the constant instead of re-typing the string literal —
# a typo in a literal comparison fails silently, a typo in an imported name
# fails at import.
PROVIDER_NAME = "faster-whisper"


# --- Lazy singleton model loader --------------------------------------------
# Loading a WhisperModel means reading model weights off disk (or downloading
# them on first run) and initializing a ctranslate2 inference session — that
# takes real time (seconds, not milliseconds). Doing that at *import* time
# would slow down importing this module even for code paths that never call
# transcribe() at all (e.g. running an unrelated test file that merely
# happens to import something that imports this). So we defer construction
# until the first real call, and cache the instance in this module-level
# variable for every call after that.
_model: WhisperModel | None = None

# Why `threading.Lock` and not `asyncio.Lock`: this module's `transcribe()`
# is synchronous and CPU-bound (see its docstring below), so the orchestrator
# is expected to call it via `asyncio.to_thread(transcribe, audio_bytes)`.
# `asyncio.to_thread` runs the call in a worker thread from the default
# executor's thread pool — with several WebSocket sessions transcribing
# concurrently, `_get_model()` can genuinely be entered by more than one
# *OS thread* at the same time, not just multiple coroutines interleaved on
# one thread. An `asyncio.Lock` only protects against the latter (it's not
# thread-safe and isn't even usable without a running event loop in the
# calling thread), so it would do nothing to stop two threads from both
# passing the "is it loaded yet?" check and each constructing their own
# WhisperModel. A `threading.Lock` is the one that actually blocks a second
# OS thread while the first is inside the critical section.
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

    # `audio` must be a path, a BinaryIO, or a numpy array — never raw bytes
    # (confirmed via inspect.signature(WhisperModel.transcribe) against the
    # installed package). io.BytesIO gives PyAV a seekable file-like object
    # to demux the container from, with no temp file needed.
    segments, info = model.transcribe(io.BytesIO(audio_bytes))

    # `segments` is a *generator*, not a list — faster-whisper decodes each
    # segment lazily, only as you iterate. That's a nice property for
    # streaming use cases (you can start acting on segment 1 before segment
    # 2 has even been decoded), but it also means `info.language` is not
    # fully reliable and no segment's text exists yet until the generator
    # has actually been driven to completion. Since this app wants the
    # whole transcript in one shot (not a live stream of partial segments),
    # we exhaust the generator here with a single list comprehension rather
    # than returning it to the caller half-consumed.
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
