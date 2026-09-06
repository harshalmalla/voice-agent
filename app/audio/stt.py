"""
Speech-to-text ORCHESTRATOR — the module that decides *which* STT provider
transcribes a given piece of audio.

The providers underneath this file (`app/audio/providers/faster_whisper_stt.py`
and `app/audio/providers/sarvam.py`) each know how to transcribe audio with
exactly one engine, and neither knows the other exists. This module is the
one place that holds the *policy*: given a chunk of audio from a live
WebSocket session, run the engines in the right order, on the right
conditions, and hand back a single `TranscriptionResult` (see
`app/audio/types.py`) that the rest of the app can consume without caring
which engine produced it.

--- The two-stage design, and why it isn't just "pick the better engine" ---
The whole shape of this file follows from one asymmetry in faster-whisper's
behaviour (documented in that provider's own module docstring):

  * its language DETECTION is reliable, across languages;
  * its non-English TRANSCRIPTION quality is weak.

Those two facts point in opposite directions, and that is precisely why
neither engine alone is the right answer:

  * "Just use faster-whisper" — free, local, no network hop, but Hindi comes
    back as approximately-phonetic gibberish.
  * "Just use Sarvam" — accurate on Indian languages, but every single
    English turn (the common case) would then cost money, burn a network
    round-trip of latency on the live voice path, and fail entirely whenever
    Sarvam is down or unconfigured.

So we run faster-whisper FIRST on every chunk, unconditionally, and use it
for two different jobs at once: it is the transcript for English audio, AND
it is the language *detector* that decides whether the audio deserves a
second, paid pass. For a regional language we keep its detection and throw
away its text. That "keep the signal, discard the payload" move is the key
idea of the file — it buys accurate regional transcription while paying the
Sarvam cost only on the turns that actually need it, and it costs nothing
extra, because language detection falls out of a normal faster-whisper
`transcribe()` call as a free side effect.

See PLAN.md's "Regional language routing" note for the product-level
statement of the same policy.
"""

from __future__ import annotations

import asyncio
import logging

from app import config
from app.audio.providers import faster_whisper_stt, sarvam
from app.audio.types import TranscriptionResult

logger = logging.getLogger(__name__)


_ENGLISH = "en"
SARVAM_ROUTED_LANGUAGES: frozenset[str] = frozenset(
    sarvam.WHISPER_TO_SARVAM_LANGUAGE
) - {_ENGLISH}


async def transcribe(audio_bytes: bytes) -> TranscriptionResult:
    """Transcribe audio with whichever engine suits the language it's in.

    The single entry point the WebSocket handler calls per user turn. Always
    returns a usable `TranscriptionResult` when faster-whisper succeeded —
    a Sarvam failure degrades the answer, it never removes it (see the
    graceful-degradation note in the body).

    Routing, in order:
      1. faster-whisper runs on every chunk (transcript + language detection).
      2. English  -> return faster-whisper's result unchanged. No network
         call, no cost, no added latency.
      3. Low detection confidence -> stay on faster-whisper; do not spend a
         Sarvam call on a guess.
      4. A Sarvam-supported regional language -> re-transcribe the SAME audio
         through Sarvam and return that instead.
      5. Anything else (e.g. French) -> faster-whisper's result, with a
         warning naming the language.

    Args:
        audio_bytes: Raw audio file bytes, e.g. a browser MediaRecorder blob
            (typically a webm/opus container). Passed straight through to
            whichever provider(s) run — nothing here needs to decode it.

    Returns:
        A TranscriptionResult whose `provider` field says which engine
        actually produced the text.

    Raises:
        ValueError: if `audio_bytes` is empty (propagated from the
            faster-whisper provider — an empty buffer is a caller bug, not a
            dependency failure, so unlike a Sarvam outage it is NOT
            swallowed here).
    """
    whisper_result = await asyncio.to_thread(faster_whisper_stt.transcribe, audio_bytes)

    language = whisper_result.language
    confidence = whisper_result.language_probability

    if language == _ENGLISH:
        logger.info(
            "STT route: using %s for English audio (confidence=%.2f) — no second pass needed.",
            whisper_result.provider,
            confidence,
        )
        return whisper_result

    if confidence < config.LANGUAGE_DETECTION_MIN_CONFIDENCE:
        logger.warning(
            "STT route: detected language=%r but confidence %.2f is below the "
            "LANGUAGE_DETECTION_MIN_CONFIDENCE threshold of %.2f — skipping the Sarvam "
            "pass and keeping the %s transcript.",
            language,
            confidence,
            config.LANGUAGE_DETECTION_MIN_CONFIDENCE,
            whisper_result.provider,
        )
        return whisper_result

    if language not in SARVAM_ROUTED_LANGUAGES:
        logger.warning(
            "STT route: detected language=%r (confidence=%.2f) is not supported by Sarvam — "
            "falling back to the %s transcript, whose non-English quality is weak.",
            language,
            confidence,
            whisper_result.provider,
        )
        return whisper_result

    logger.info(
        "STT route: detected regional language=%r (confidence=%.2f) — re-transcribing the same "
        "audio via Sarvam for accuracy, discarding the %s transcript.",
        language,
        confidence,
        whisper_result.provider,
    )

    try:
        sarvam_result = await sarvam.transcribe(audio_bytes)
    except Exception as error:
        logger.warning(
            "STT route: Sarvam transcription failed for language=%r (%s: %s) — degrading to the "
            "%s transcript rather than failing this turn.",
            language,
            type(error).__name__,
            error,
            whisper_result.provider,
            exc_info=True,
        )
        return whisper_result

    logger.info(
        "STT route: using %s transcript for language=%r (Sarvam reported language=%r).",
        sarvam_result.provider,
        language,
        sarvam_result.language,
    )
    return sarvam_result
