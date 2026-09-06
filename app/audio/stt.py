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


# --- Which languages are worth a second pass? -------------------------------
# Derived from Sarvam's own language map rather than restated as a second
# literal list here. Single source of truth: the day someone adds a language
# to `WHISPER_TO_SARVAM_LANGUAGE` in the provider (because Sarvam shipped
# support for it), routing for that language turns on here automatically,
# with no second edit to remember. A hand-copied list would eventually drift
# out of sync, and the failure mode of that drift is silent — a language
# Sarvam supports quietly keeps getting served weak faster-whisper text, and
# nothing errors.
#
# English is subtracted out even though Sarvam does support it ("en-IN"),
# because supporting it and it being *worth calling for* are different
# questions. faster-whisper's English transcription is already good, and it
# has already run by the time we get here — sending English to Sarvam would
# buy no accuracy and cost money plus latency on the most common path in the
# app. This is the one place where the routing set intentionally differs
# from the capability set.
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
    # --- Stage 1: faster-whisper, always ------------------------------------
    # `asyncio.to_thread` is MANDATORY here, and this is not a stylistic
    # preference. `faster_whisper_stt.transcribe` is synchronous and
    # CPU-bound: it runs ctranslate2 inference on the CPU and blocks the
    # calling thread outright for the whole call — tens of milliseconds to
    # several seconds depending on clip length. This app is a single-process
    # asyncio event loop serving EVERY connected WebSocket session on one
    # thread. Calling a blocking function directly from this coroutine's
    # body would freeze that thread for the full inference, stalling every
    # other live session's messages, timers and reminders along with it —
    # one user's three-second clip becomes three seconds of dead air for
    # everyone. `asyncio.to_thread` hands the blocking work to a worker
    # thread from the default executor and yields control back to the loop
    # for the duration, so the loop keeps servicing everyone else.
    #
    # Contrast this deliberately with the `await sarvam.transcribe(...)`
    # below, which is NOT wrapped in a thread. Sarvam's work is an HTTP
    # round-trip — I/O-bound, not CPU-bound — driven by httpx.AsyncClient,
    # so awaiting it already releases the event loop for the entire network
    # wait. Wrapping that in a thread would occupy a worker thread to sit
    # and do nothing, buying zero concurrency. Same syntax at the call site
    # (`await`), two completely different bottlenecks, two different tools:
    # threads for blocking CPU work, native await for I/O.
    whisper_result = await asyncio.to_thread(faster_whisper_stt.transcribe, audio_bytes)

    language = whisper_result.language
    confidence = whisper_result.language_probability

    # --- Stage 2: English needs nothing further -----------------------------
    if language == _ENGLISH:
        logger.info(
            "STT route: using %s for English audio (confidence=%.2f) — no second pass needed.",
            whisper_result.provider,
            confidence,
        )
        return whisper_result

    # --- Stage 3: refuse to act on an unconfident signal --------------------
    # Every branch below this point spends a paid, latency-adding API call on
    # the strength of a *guess* about what language was spoken. When that
    # guess is weak, the right move is to do nothing rather than to do
    # something confidently wrong: a mis-route buys a Sarvam transcription
    # performed under the wrong language assumption, so we would be paying
    # more money and more latency to get a WORSE transcript than the one we
    # already have in hand. Short clips, background noise, and code-mixed
    # "Hinglish" are the realistic sources of a low score here — all cases
    # where the local English-leaning transcript is the safer default.
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

    # --- Stage 5 (checked before 4): nothing downstream can help ------------
    # A confidently-detected language that Sarvam simply doesn't handle —
    # French, Japanese, anything outside the Indian-language set. There is no
    # better engine wired into this app for it, so faster-whisper's own
    # transcript is genuinely the best available answer. Logged at WARNING
    # rather than info because it names a concrete gap in coverage: if this
    # line shows up repeatedly for one language in production, that is the
    # signal to go add a provider for it.
    if language not in SARVAM_ROUTED_LANGUAGES:
        logger.warning(
            "STT route: detected language=%r (confidence=%.2f) is not supported by Sarvam — "
            "falling back to the %s transcript, whose non-English quality is weak.",
            language,
            confidence,
            whisper_result.provider,
        )
        return whisper_result

    # --- Stage 4: regional language -> re-transcribe via Sarvam -------------
    # We are about to discard `whisper_result.text` while keeping the
    # `language` it detected. That is intentional and is the core of the
    # two-stage design described in the module docstring: detection is the
    # part of faster-whisper worth trusting for this audio, transcription is
    # not. We keep `whisper_result` bound anyway — it is the fallback the
    # error handler below returns.
    logger.info(
        "STT route: detected regional language=%r (confidence=%.2f) — re-transcribing the same "
        "audio via Sarvam for accuracy, discarding the %s transcript.",
        language,
        confidence,
        whisper_result.provider,
    )

    try:
        # Awaited directly, no asyncio.to_thread — see the long note at
        # stage 1 for why the two providers are called so differently.
        sarvam_result = await sarvam.transcribe(audio_bytes)
    except Exception as error:
        # --- GRACEFUL DEGRADATION: the most important decision in this file -
        #
        # A broad `except Exception` here is deliberate, and it deliberately
        # contradicts the "never swallow exceptions" policy stated in
        # `app/rag/embeddings.py` and `app/audio/providers/sarvam.py`. Both
        # positions are correct, because they belong to different LAYERS:
        #
        #   * Those modules are INFRASTRUCTURE. Their one job is to do a
        #     single thing correctly and report failure honestly upward.
        #     Swallowing there would hide a real problem and produce
        #     silently-corrupt data — an ingestion run that writes a
        #     collection full of nulls, a retrieval that quietly answers from
        #     no context at all. Nobody is positioned to make a *product*
        #     decision down there, so the only honest move is to raise.
        #
        #   * This module is ORCHESTRATION / POLICY. Its job is not to
        #     transcribe anything; it is to decide what the USER EXPERIENCE
        #     should be when a dependency fails. And here we already hold a
        #     complete, usable transcript from stage 1 — computed before we
        #     ever reached out to Sarvam. Letting this exception propagate
        #     would throw that transcript away and kill the user's entire
        #     spoken turn (no answer, no reply, dead air) because a
        #     third-party API happened to be down, rate-limited, slow, or
        #     unconfigured. A rougher local transcript is strictly better
        #     than nothing.
        #
        # The general principle: LET INFRASTRUCTURE RAISE, LET POLICY DECIDE.
        # Failing loudly and degrading gracefully are not in conflict — they
        # simply belong at different layers. The exception still travelled
        # honestly all the way up from httpx/Sarvam to here; this is just the
        # first frame with enough context to know that a fallback exists.
        #
        # Why `Exception` and not a tuple of specific types: the set of ways
        # this call can fail is genuinely open-ended and spans several
        # libraries — SarvamAPIError for a non-2xx, RuntimeError from
        # `config.require` when SARVAM_API_KEY is unset, httpx.TimeoutException
        # / ConnectError / a TLS error for the network, a JSON decode error
        # for a malformed body. Enumerating them means the one we forgot
        # becomes the one that takes a user's turn down in production, and
        # every branch above proves we have a working fallback regardless of
        # the reason. `BaseException` is NOT caught, so asyncio.CancelledError
        # (a disconnecting client) and KeyboardInterrupt still propagate —
        # cancellation is not a failure to degrade around, it is an
        # instruction to stop.
        logger.warning(
            "STT route: Sarvam transcription failed for language=%r (%s: %s) — degrading to the "
            "%s transcript rather than failing this turn.",
            language,
            type(error).__name__,
            error,
            whisper_result.provider,
            # Full traceback at WARNING, since the *reason* is the whole
            # point of the log line: a missing API key and a network timeout
            # demand very different fixes.
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
