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

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
NOTES_FILE = DATA_DIR / "notes.md"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MONGODB_URI = os.getenv("MONGODB_URI", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

GEMINI_MODEL_PRIMARY = os.getenv("GEMINI_MODEL_PRIMARY", "gemini-3.5-flash")
GEMINI_MODEL_FALLBACK = os.getenv("GEMINI_MODEL_FALLBACK", "gemini-3.5-flash-lite")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

SARVAM_API_BASE_URL = os.getenv("SARVAM_API_BASE_URL", "https://api.sarvam.ai")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "shubh")

MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "voice_agent")
DOCUMENTS_COLLECTION = "documents"
MEMORY_COLLECTION = "memory"
VECTOR_INDEX_NAME = "vector_index"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "12"))

MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "5"))

MAX_REMINDER_DELAY_SECONDS = int(os.getenv("MAX_REMINDER_DELAY_SECONDS", "86400"))

WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

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
