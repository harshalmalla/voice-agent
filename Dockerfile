# Live Voice RAG Agent
#
# Build:  docker build -t voice-agent .
# Run:    docker run -p 8000:8000 --env-file .env voice-agent
#
# Sizing note: the default faster-whisper model (`base`, int8) needs roughly
# 1GB of RAM at inference time. Free tiers capped at 512MB will not run this
# image — use a 1GB+ instance or a small VPS.

FROM python:3.12-slim

# System dependencies, deliberately minimal.
#
# NO ffmpeg. faster-whisper decodes audio through PyAV, whose manylinux wheels
# bundle their own FFmpeg shared libraries, so installing the system package
# would add ~100MB for something already in the wheel. Verified: `av` is a
# direct dependency of the installed faster-whisper.
#
# libgomp1 IS needed: ctranslate2 (the inference runtime under faster-whisper)
# links against the GNU OpenMP runtime, which the slim base image does not
# ship. Without it, importing faster_whisper fails at load time with
# "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies are copied and installed BEFORE the application code, on
# purpose. Docker caches each layer and invalidates every layer after the first
# one whose inputs changed. Application code changes on every commit;
# requirements.txt changes rarely. Copying the code first would put the
# ~500MB torch-free-but-still-heavy dependency install downstream of it, and
# every one-line edit would reinstall the entire tree. This ordering means a
# code change rebuilds only the final COPY.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Run as a non-root user. A compromised process should not be able to write to
# the image or escalate inside the container.
#
# The home directory matters here: faster-whisper downloads its model weights
# on FIRST USE, not at build time, into the Hugging Face cache under $HOME. A
# non-root user with no writable home makes that download fail at the first
# voice turn rather than at startup, which is a confusing way to find out.
# HF_HOME is set explicitly so the location is stated rather than inferred.
#
# Two ways to avoid paying the download on every cold start:
#   1. Bake the weights in, by adding a build step before this that runs the
#      model once as root and copies the cache into the user's home.
#   2. Mount a cache volume:
#      docker run -v whisper-cache:/home/app/.cache ...
RUN useradd --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
ENV HF_HOME=/home/app/.cache/huggingface
USER app

EXPOSE 8000

# Bound to 0.0.0.0, not 127.0.0.1: a server listening on loopback inside a
# container is unreachable from the host's published port.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
