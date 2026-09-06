"""
Central configuration for the Live Voice RAG Agent.

Every other module reads its settings from here instead of calling
`os.getenv(...)` directly. One place to see everything the app depends on;
one place to change a default.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file into the process environment BEFORE
# we read any of them below. This must happen at import time, at the top of
# the file — every os.getenv() call below only sees what's already in the
# environment at the moment it runs.
load_dotenv()

# --- Paths ---------------------------------------------------------------
# __file__ is this file's own path. Its parent is app/, and that directory's
# parent is the project root — resolve() turns it into an absolute path so
# it doesn't matter what directory you were in when you launched the app.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
NOTES_FILE = DATA_DIR / "notes.md"

# --- Secrets / external services ------------------------------------------
# Default to "" rather than None so callers can do a plain truthiness check
# (`if not GOOGLE_API_KEY`) without a None-check first.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MONGODB_URI = os.getenv("MONGODB_URI", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# --- Gemini model configuration --------------------------------------------
# Overridable via env so you can swap models without touching code — e.g.
# to test a cheaper/faster model, or roll back if a new one misbehaves.
GEMINI_MODEL_PRIMARY = os.getenv("GEMINI_MODEL_PRIMARY", "gemini-3.5-flash")
GEMINI_MODEL_FALLBACK = os.getenv("GEMINI_MODEL_FALLBACK", "gemini-3.5-flash-lite")
# NOTE: models/text-embedding-004 was retired on 2026-01-14 — using its
# replacement. If you change this, EMBEDDING_DIMENSION below and the Atlas
# index's numDimensions must both still match the new model's output width.
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# --- Speech-to-text configuration ------------------------------------------
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# --- Sarvam AI configuration (speech-to-text + text-to-speech) -------------
# Sarvam is the second STT/TTS provider (see app/audio/providers/sarvam.py),
# used for regional-language audio that faster-whisper's language *detection*
# flagged but can't itself transcribe well, and for synthesizing spoken
# replies back in that same regional language. Overridable via env for the
# same reason as the Gemini model settings above — swapping model versions
# shouldn't require touching code.
SARVAM_API_BASE_URL = os.getenv("SARVAM_API_BASE_URL", "https://api.sarvam.ai")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "shubh")

# --- MongoDB Atlas -----------------------------------------------------
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "voice_agent")
DOCUMENTS_COLLECTION = "documents"
MEMORY_COLLECTION = "memory"
VECTOR_INDEX_NAME = "vector_index"

# --- RAG tuning ------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# --- Logging ---------------------------------------------------------------
# Overridable so you can turn on DEBUG in one environment (e.g. staging,
# or your own shell) without touching code or redeploying — see
# app/logging_config.py for where this is actually applied.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Embedding vector shape -------------------------------------------------
# The number of floats in every embedding vector this app produces (see
# app/rag/embeddings.py). This is not a knob you can tune freely — it is a
# *contract* between three things that must all agree:
#
#   1. the embedding model named in GEMINI_EMBEDDING_MODEL above,
#   2. every vector already written into the `documents` / `memory`
#      collections in MongoDB Atlas, and
#   3. the `numDimensions` field of the Atlas Vector Search index definition
#      (VECTOR_INDEX_NAME above) that you create by hand in the Atlas UI.
#
# If (3) disagrees with (1), Atlas rejects the `$vectorSearch` query with an
# error about the query vector's dimension — confusing, because nothing about
# the *embedding* call itself failed. If you change the model, you must change
# this number, re-create the Atlas index with the new numDimensions, AND
# re-embed every existing document: old vectors of a different length (or from
# a different model) are meaningless in the new vector space, so a partially
# migrated collection returns silently-wrong neighbours rather than an error.
#
# 768 is the documented output size of `models/text-embedding-004` (fixed),
# and is also one of the recommended output sizes of `gemini-embedding-001`
# (which is flexible: 128-3072, recommended 768 / 1536 / 3072).
#
# HEADS UP: Google retired `models/text-embedding-004` on 2026-01-14; the
# supported replacement is `models/gemini-embedding-001`. Set
# GEMINI_EMBEDDING_MODEL accordingly (via .env or by changing its default
# above) — 768 remains a valid dimension for the replacement, so this constant
# does not need to change when you switch.
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# --- Language-detection confidence gate -------------------------------------
# The minimum `language_probability` faster-whisper must report before the
# STT orchestrator (app/audio/stt.py) is willing to ACT on its language
# guess by re-transcribing the same audio through Sarvam.
#
# Detection confidence is a real number, not a formality: a short clip, a
# noisy room, or a code-mixed "Hinglish" utterance can leave faster-whisper
# genuinely unsure which language it heard. Routing on an unsure guess is
# strictly worse than not routing at all — a wrong route spends a paid
# Sarvam API call, adds a network round-trip of latency, and hands back a
# transcript produced under the wrong language assumption, i.e. we pay more
# to get a worse answer. Below this threshold the orchestrator stays on the
# local faster-whisper transcript and logs a warning.
#
# 0.5 is a deliberately permissive floor: it only filters out cases where
# the model is closer to a coin flip than to a judgement. Raise it if you
# see wrong-language routes in the logs; lower it if genuinely regional
# audio is being left on the English path.
LANGUAGE_DETECTION_MIN_CONFIDENCE = float(os.getenv("LANGUAGE_DETECTION_MIN_CONFIDENCE", "0.5"))


def require(value: str, name: str) -> str:
    """Return `value`, or raise a clear, actionable error if it's empty.

    Config itself stays lenient — importing this module never crashes, even
    if no .env exists yet (useful for tests, tooling, `--help`, etc). It's
    up to whichever module actually NEEDS a given secret (e.g. the Gemini
    client, the Mongo client) to call `require(...)` at the point of use,
    so the app fails fast with a message that says exactly what's missing,
    instead of failing later with a confusing error from deep inside some
    third-party SDK.
    """
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value
