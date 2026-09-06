"""
Sarvam AI speech-to-text / text-to-speech provider.

Sarvam (https://docs.sarvam.ai) is this app's second STT/TTS provider — used
for regional-Indian-language audio that faster-whisper's language
*detection* flagged but whose non-English transcription quality is weak
(see `faster_whisper_stt.py`'s module docstring), and for synthesizing
spoken replies back in that same regional language. It transcribes via
`transcribe()`, returning the exact same `TranscriptionResult` shape
`faster_whisper_stt.transcribe()` does (see `app/audio/types.py`) — the
orchestrator that will pick between providers doesn't need to know or care
which one actually produced a given result.

Endpoints, headers, and response shapes below were confirmed against
Sarvam's own API reference docs (see the docstrings on each function for
the exact source pages), not written from memory.

--- Why this module is natively async, unlike faster_whisper_stt.py -------
`faster_whisper_stt.transcribe()` is synchronous and CPU-bound: it runs
model inference on the CPU, which occupies a CPU core and blocks whatever
thread calls it for the whole duration. An `await` can't help there — there
is no I/O to yield control during — so that module has to be pushed onto a
separate OS thread with `asyncio.to_thread(...)` to avoid freezing the
single-process event loop that's also servicing every other connected
WebSocket session.

This module's two functions are the opposite: the "work" is entirely an
HTTP round-trip to Sarvam's servers, i.e. I/O-bound, not CPU-bound. While
we're waiting on the network, the CPU has nothing to do but wait — so
`httpx.AsyncClient`, an async-capable HTTP client, can `await` the request
and hand control straight back to the event loop for that entire wait.
The event loop then goes on servicing other coroutines (other WebSocket
sessions' messages, timers, ...) on the *same* thread, and resumes this
coroutine only once Sarvam's response bytes actually arrive. No worker
thread, no `asyncio.to_thread`, needed or wanted here — wrapping an
I/O-bound async call in a thread would just waste a thread for no benefit.
"""

from __future__ import annotations

import base64
import io
import logging

import httpx

from app import config
from app.audio.types import TranscriptionResult

logger = logging.getLogger(__name__)

PROVIDER_NAME = "sarvam"

_REQUEST_TIMEOUT_SECONDS = 30.0


class SarvamAPIError(RuntimeError):
    """Raised when the Sarvam API returns a non-2xx response.

    Kept as its own exception type (rather than raising a bare
    RuntimeError) so a caller further up the stack — e.g. an orchestrator
    deciding whether to retry or fall back to a different provider — can
    catch this specifically without also swallowing unrelated RuntimeErrors
    from elsewhere in the app. This module is infrastructure, not an agent
    tool that reports failures back to an LLM as text, so raising here
    (rather than returning an error string) is the correct behavior — it's
    up to the caller to decide how to handle the failure.
    """


WHISPER_TO_SARVAM_LANGUAGE: dict[str, str] = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "pa": "pa-IN",
    "or": "od-IN",
}

SARVAM_TO_WHISPER_LANGUAGE: dict[str, str] = {
    sarvam_code: whisper_code for whisper_code, sarvam_code in WHISPER_TO_SARVAM_LANGUAGE.items()
}


def _to_sarvam_language_code(language_code: str) -> str:
    """Translate a plain ISO 639-1 code (e.g. "hi") to Sarvam's expected
    BCP-47-ish format (e.g. "hi-IN").

    Idempotent: a code already in Sarvam's format is returned unchanged, so
    callers don't need to know or care which format they're currently
    holding. Falls back to returning the input unchanged for anything
    unrecognized (rather than raising) — Sarvam's own API is the authority
    on which language codes are actually valid, so we let a bad code fail
    there with Sarvam's own error rather than guessing here.
    """
    if language_code in SARVAM_TO_WHISPER_LANGUAGE:
        return language_code
    return WHISPER_TO_SARVAM_LANGUAGE.get(language_code, language_code)


def _raise_for_status(response: httpx.Response, action: str) -> None:
    """Raise a SarvamAPIError with the status code and response body if
    `response` wasn't a 2xx — never swallow a Sarvam error silently.
    """
    if response.is_success:
        return
    raise SarvamAPIError(
        f"Sarvam {action} request failed with HTTP {response.status_code}: {response.text}"
    )


async def transcribe(audio_bytes: bytes) -> TranscriptionResult:
    """Transcribe raw audio bytes via Sarvam's speech-to-text REST API.

    Confirmed against https://docs.sarvam.ai/api-reference/speech-to-text/transcribe:
    - `POST {SARVAM_API_BASE_URL}/speech-to-text`
    - Auth header: `api-subscription-key: <SARVAM_API_KEY>` (a raw key value,
      not a "Bearer ..." token — that's an *alternative* Sarvam also
      accepts on the `Authorization` header, not the primary method).
    - multipart/form-data body; the audio file goes in a form field named
      `file`.
    - `language_code="unknown"` requests automatic language detection —
      exactly what this app wants, since (mirroring faster_whisper_stt's
      own transcribe()) the caller here doesn't know the language up front
      either; that's the whole point of routing to this provider.
    - Response JSON carries `transcript`, `language_code` (Sarvam's
      BCP-47-ish format, e.g. "hi-IN"), and `language_probability`.

    The returned `language` is translated back to a plain ISO 639-1 code
    (e.g. "hi") via `SARVAM_TO_WHISPER_LANGUAGE` before being returned, so
    a caller comparing this provider's output against faster-whisper's sees
    the same code format from both — the actual point of sharing
    `TranscriptionResult` at all (see app/audio/types.py).

    Args:
        audio_bytes: Raw audio file bytes (e.g. a browser MediaRecorder
            blob). Sarvam documents support for WAV, MP3, AAC, AIFF, OGG,
            OPUS, FLAC, MP4/M4A, AMR, WMA, WebM, and raw PCM — passed
            through opaquely here with a generic filename, matching the
            same "we don't need to know the container up front" approach
            faster_whisper_stt.py takes.

    Returns:
        A TranscriptionResult with the transcript text and detected
        language + confidence, in the same shape every STT provider
        returns.

    Raises:
        ValueError: if `audio_bytes` is empty.
        SarvamAPIError: if Sarvam returns a non-2xx response.
    """
    if not audio_bytes:
        raise ValueError("transcribe() received empty audio_bytes — nothing to transcribe.")

    api_key = config.require(config.SARVAM_API_KEY, "SARVAM_API_KEY")

    url = f"{config.SARVAM_API_BASE_URL}/speech-to-text"
    headers = {"api-subscription-key": api_key}
    data = {
        "model": config.SARVAM_STT_MODEL,
        "language_code": "unknown",
    }
    files = {"file": ("audio.webm", io.BytesIO(audio_bytes))}

    logger.debug("Sending %d bytes to Sarvam speech-to-text (model=%s)", len(audio_bytes), config.SARVAM_STT_MODEL)

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(url, headers=headers, data=data, files=files)

    _raise_for_status(response, action="speech-to-text")

    payload = response.json()
    sarvam_language = payload.get("language_code", "")
    whisper_style_language = SARVAM_TO_WHISPER_LANGUAGE.get(sarvam_language, sarvam_language)

    logger.debug(
        "Sarvam transcription detected language=%s (probability=%s)",
        sarvam_language,
        payload.get("language_probability"),
    )

    return TranscriptionResult(
        text=payload.get("transcript", ""),
        language=whisper_style_language,
        language_probability=payload.get("language_probability", 0.0),
        provider=PROVIDER_NAME,
    )


async def synthesize_speech(text: str, language_code: str) -> bytes:
    """Synthesize speech audio for `text` via Sarvam's text-to-speech REST API.

    Confirmed against https://docs.sarvam.ai/api-reference/text-to-speech/convert:
    - `POST {SARVAM_API_BASE_URL}/text-to-speech`
    - Auth header: `api-subscription-key: <SARVAM_API_KEY>` (same as
      transcribe() above).
    - JSON body (not multipart) with `text`, `language_code` (required,
      BCP-47-ish, e.g. "hi-IN"), `model`, and `speaker`.
    - Response JSON carries `audios`: a list of base64-encoded audio
      strings (WAV by default; Sarvam's `output_audio_codec` field could
      request other formats, but we don't set it, so we get their default).
      Sarvam returns a *list* because a request can ask for multiple
      renderings; this app only ever wants one, so we take `audios[0]`.

    Callers deal in raw bytes, not in Sarvam's base64 wire encoding — the
    decoding happens here so nothing downstream needs to know Sarvam
    base64-encodes its audio at all.

    Args:
        text: The text to speak. Sarvam documents a 2500-character limit
            for the bulbul:v3 model.
        language_code: Either a plain ISO 639-1 code (e.g. "hi", as
            faster-whisper/TranscriptionResult produce) or Sarvam's own
            BCP-47-ish format (e.g. "hi-IN") — translated to Sarvam's
            format if needed via `_to_sarvam_language_code`.

    Returns:
        Raw, decoded audio bytes.

    Raises:
        ValueError: if `text` is empty.
        SarvamAPIError: if Sarvam returns a non-2xx response, or a 2xx
            response with no audio in it.
    """
    if not text:
        raise ValueError("synthesize_speech() received empty text — nothing to synthesize.")

    api_key = config.require(config.SARVAM_API_KEY, "SARVAM_API_KEY")

    url = f"{config.SARVAM_API_BASE_URL}/text-to-speech"
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    body = {
        "text": text,
        "language_code": _to_sarvam_language_code(language_code),
        "model": config.SARVAM_TTS_MODEL,
        "speaker": config.SARVAM_TTS_SPEAKER,
    }

    logger.debug(
        "Requesting Sarvam text-to-speech (language=%s, model=%s, chars=%d)",
        body["language_code"],
        config.SARVAM_TTS_MODEL,
        len(text),
    )

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(url, headers=headers, json=body)

    _raise_for_status(response, action="text-to-speech")

    payload = response.json()
    audios = payload.get("audios") or []
    if not audios:
        raise SarvamAPIError(
            f"Sarvam text-to-speech request succeeded but returned no audio: {payload}"
        )

    return base64.b64decode(audios[0])
