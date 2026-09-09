# Live Voice RAG Agent

A real-time voice assistant you talk to in the browser. It answers from your
own documents, remembers the conversation semantically rather than by
truncation, and can call tools mid-turn (web search, reminders, arithmetic,
the clock, a local notes file). Speech is transcribed locally with
faster-whisper; when the detected language is an Indian regional language the
turn is routed automatically to Sarvam AI for both transcription and spoken
reply, with no manual language picker anywhere in the UI.

Python, FastAPI and a WebSocket for the transport. Google Gemini for
generation and embeddings, called through the native `google-genai` SDK — no
LangChain. MongoDB Atlas Vector Search for both the document index and the
conversation memory. The frontend is vanilla JavaScript with no build step.

---

## Features

| Feature | What it actually does |
| --- | --- |
| Push-to-talk voice input | Hold a button, `MediaRecorder` captures the utterance, the blob is sent as one binary WebSocket frame. |
| Local speech-to-text | `faster-whisper` (`base`, `int8`) runs on CPU in a worker thread and returns the transcript plus a detected language and confidence. |
| Automatic regional-language routing | A confident non-English detection re-transcribes the same audio through Sarvam AI; the language is then carried through the prompt and the reply. |
| Retrieval over your documents | PDF / txt / md files under `data/documents/` are chunked, embedded and searched with Atlas `$vectorSearch`. |
| Semantic conversation memory | Every completed exchange is embedded and stored; older turns are recalled by meaning, pre-filtered to the current session. |
| Function-calling agent | Gemini chooses between answering directly and calling `web_search`, `get_current_time`, `calculate`, `read_notes`, `set_reminder`, `list_reminders`. |
| Live reminders | `set_reminder` schedules an asyncio task that pushes a `reminder_fired` event into the session hours later, unprompted. |
| Model fallback | Every Gemini call retries with backoff, then falls back to a lighter model with its own quota, so a rate-limited primary degrades the answer instead of losing the turn. |
| Live activity dashboard | Retrieval hits, tool calls and tool results stream to the page while the turn is still running. |
| Text input path | The same turn minus transcription, which is what makes the pipeline testable without a microphone. |

Speech output: English replies are spoken by the browser's `speechSynthesis`;
regional replies are synthesized by Sarvam TTS and sent back as a `tts_audio`
event, because browser voice support for Indian languages is inconsistent.

---

## Architecture

```
Browser (dashboard)                         FastAPI backend
──────────────────────                      ─────────────────────────────
[hold mic button] ──record──▶ audio blob ──▶ /ws/{session_id}  (WebSocket)
                                                 │
                                                 ▼
                                        app/audio/stt.py
                                        faster-whisper.transcribe()
                                        (run via asyncio.to_thread)
                                        → transcript + detected language
                                                 │
                                    en?──yes──▶ keep the local transcript
                                                 │
                                    no (hi/ta/te/bn/...) and confident
                                                 ▼
                                  app/audio/providers/sarvam.py
                                  Sarvam STT re-transcribes the same
                                  audio accurately in that language
                                                 │  transcript (+ language)
                                                 ▼
                                        app/agent/gemini_agent.py
                                        ┌──────────────────────────────┐
                                        │ 1. embed the question        │
                                        │ 2. $vectorSearch top-k from  │
                                        │      documents collection    │
                                        │      memory collection       │
                                        │      (scoped to the session) │
                                        │ 3. Gemini function calling:  │
                                        │      answer, or call a tool  │
                                        │ 4. feed tool results back,   │
                                        │      bounded by              │
                                        │      MAX_TOOL_ITERATIONS     │
                                        └──────────────────────────────┘
                                                 │  answer text (+ language)
                                                 ▼
                                  store the exchange in the memory
                                  collection, so later turns recall it
                                                 │
                              en? ──yes──▶ text only
                              no  ──────▶ Sarvam TTS audio alongside the text
                                                 │
                     ◀── WS events: transcript, retrieval, tool_call,
                         tool_result, agent_answer, tts_audio,
                         reminder_fired, error ──┘
[live dashboard updates]
[en: window.speechSynthesis.speak(answer)]
[regional: play the tts_audio clip]

Background: asyncio tasks for pending reminders push reminder_fired into the
same session's queue whenever they come due — which is why the socket has a
sender that does not wait on a receiver.
```

| Layer | Module | Responsibility |
| --- | --- | --- |
| Transport | `app/main.py` | Lifespan, CORS, static frontend, the WebSocket endpoint and one turn's choreography. The only module that knows a network exists. |
| Speech | `app/audio/stt.py`, `app/audio/providers/` | Run local STT, decide whether the language warrants a paid re-transcription, and synthesize regional speech. |
| Retrieval | `app/rag/` | Chunking and ingestion, the Atlas `$vectorSearch` data layer, embeddings, and the conversation-memory collection. |
| Reasoning | `app/agent/` | Prompt construction, the bounded function-calling loop, model fallback, and the tool implementations. |
| State | `app/services/session_manager.py`, `app/reminders/scheduler.py` | Per-session history and outbound event queue; background reminder tasks bound to a session's lifetime. |

---

## Setup

### Prerequisites

- Python 3.12
- A MongoDB Atlas cluster (the free M0 tier is enough)
- A Google AI Studio API key
- Chrome or another Chromium browser (see Known limitations)

### 1. Create a virtual environment and install

```bash
cd "Voice Agent"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No system `ffmpeg` is required: `faster-whisper` decodes audio through PyAV,
whose wheels bundle their own FFmpeg libraries.

### 2. Configure the environment

```bash
cp .env.example .env
```

Then fill in the keys. `.env.example` documents every variable and what
happens if it is missing; only the first two below are required.

| Variable | Required | Where to get it |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Yes | <https://aistudio.google.com/apikey> — free tier. Used for both chat and embeddings. |
| `MONGODB_URI` | Yes | Atlas UI, cluster, Connect, Drivers. Replace the credentials placeholders and URL-encode any special characters in the password. |
| `TAVILY_API_KEY` | No | <https://tavily.com> — free tier. Without it the `web_search` tool is unavailable; everything else works. |
| `SARVAM_API_KEY` | No | <https://www.sarvam.ai> — free tier. Without it regional speech falls back to the weak local transcript and to the browser's English voice. English is unaffected. |

Nothing in the codebase calls `os.getenv` outside `app/config.py`, so that
module and `.env.example` are the complete list of knobs.

### 3. Let the app create the Atlas vector indexes

Start the server once. On startup it calls `store.ensure_vector_index` for
both the `documents` and `memory` collections, creating each collection and
its `vector_index` search index if they are missing.

```bash
uvicorn app.main:app --reload
```

Two timing facts, verified against a live cluster and worth knowing before you
conclude that retrieval is broken:

- A newly created vector index takes roughly **30 seconds** to finish building
  and become queryable. Searches against it before then return no results
  rather than an error.
- Documents inserted **immediately after** the index is created may not be
  searchable for a few seconds while the index catches up.

So if the first ingestion run reports chunks written but the first question
retrieves nothing, wait half a minute and ask again. If index creation is
refused by your cluster tier, the app logs the exact JSON to paste into the
Atlas UI (Atlas Search, Create Search Index, JSON Editor, Vector Search) and
carries on running.

### 4. Ingest your documents

Drop PDFs, `.txt` or `.md` files into `data/documents/`, then:

```bash
python -m scripts.ingest_docs
```

It reports how many chunks were embedded per file. Re-running it deletes a
file's previous chunks before inserting the new ones, so re-ingestion replaces
rather than duplicates.

### 5. Run

```bash
uvicorn app.main:app --reload
```

Open <http://localhost:8000>, allow microphone access, hold the mic button and
ask a question grounded in one of your documents. The activity panel shows the
retrieved chunks and any tool calls as they happen.

### 6. Run the tests

```bash
pytest tests/ -v
```

`test_chunking.py` and most of `test_tools.py` are pure unit tests with no
external calls. `test_smoke_ws.py` boots the real app (Atlas indexes and all)
and drives one full turn over `/ws/{session_id}` using the typed `text_query`
path — the same one the pipeline is testable with when no microphone is
available — so it calls the live Gemini and Atlas APIs and is skipped
automatically if `GOOGLE_API_KEY` or `MONGODB_URI` is not set.

---

## Design decisions

**Two-stage speech-to-text.** faster-whisper's language *detection* is
reliable; its non-English *transcription* is not. The two are separable, so
the system uses each for what it is good at: every utterance runs through the
local model, English keeps that transcript and costs nothing, and a
confidently-detected regional language re-transcribes the same audio through
Sarvam. A detection below `LANGUAGE_DETECTION_MIN_CONFIDENCE` keeps the local
transcript rather than spending a paid call on a guess. The alternative
designs are both worse: sending everything to a paid API pays for the common
case, and a manual language picker asks the user to solve a problem the audio
already answers.

**Two-tier conversation memory.** A bounded window of recent turns is replayed
verbatim, and everything older is recalled by embedding similarity from the
`memory` collection. Plain truncation loses facts permanently — the user says
their flight is on Tuesday, twelve turns pass, and the assistant no longer
knows. Summarization loses them silently and unpredictably, which is worse,
because a summary decides what mattered before you know what will be asked.
Semantic recall keeps the prompt small and bounded while making an arbitrarily
old turn retrievable at the moment it becomes relevant. Recall is pre-filtered
by `session_id` inside the `$vectorSearch` stage rather than filtered
afterwards, so scoping never eats into the top-k.

**The tool loop is bounded.** Nothing in the function-calling protocol
requires a model to stop calling tools. A model that decides a search did not
quite answer the question will call it again with a near-identical query, and
each pass appends the previous call and its result to the request. So an
unbounded loop is not merely infinite — it is *accelerating*: every iteration
is a paid API call with a larger prompt than the last, running on a live
session while the user waits in silence. `MAX_TOOL_ITERATIONS` converts an
unbounded failure into a bounded, logged, mildly disappointing one, and the
turn returns the best text it has rather than raising.

**`calculate()` uses an AST whitelist, not `eval()`.** The expression handed
to this tool is written by an LLM whose prompt contains text retrieved from
documents the model does not control. That is a complete indirect
prompt-injection path: a sentence planted in an ingested PDF can steer the
model into emitting an expression, and `eval()` on an attacker-influenced
string is arbitrary code execution in the server process. So the expression is
parsed with `ast.parse` and walked node by node against a whitelist of numeric
literals and arithmetic operators, with exponents additionally bounded so
`9**9**9` cannot hang the event loop. Anything else is refused as a string the
model can read and recover from.

**The WebSocket handler runs two concurrent loops.** The obvious shape —
receive a message, do the work, send the reply — cannot express this
application, because most of what the server sends is not a reply to anything.
A reminder fires on its own clock twenty minutes into a silence, when a
receive-then-reply loop is parked in `receive()` with no code path that
writes; the reminder would arrive whenever the user next happened to speak,
which for a reminder is the same as not arriving. And the agent emits
`retrieval`, `tool_call` and `tool_result` *during* the single `await` such a
loop would be blocked on, so progress could only be delivered in a batch after
the turn ended, which defeats the point of emitting it separately. Instead
there is a receive loop that drives turns and a send loop that drains the
session's queue, with the queue as the seam. Either loop ending cancels the
other, because a connection that can talk but not listen — or one that spends
money transcribing and delivers nothing — is a leak, not a degraded mode.

**Failure policy: infrastructure raises, policy decides.** Modules that cannot
know what a failure means (`store.py`, `embeddings.py`, `memory.py`, the
provider clients) raise typed errors. Modules that know a person is holding a
microphone (`stt.py`, `gemini_agent.py`, `main.py`) decide to degrade:
retrieval failure answers without context, a Sarvam TTS failure still delivers
the text, an unexpected exception in a turn keeps the socket open and sends an
`error` event that clears the client's pending bubble. Tools never raise at
all; they return their failure as a string the model can act on.

---

## Known limitations

- **Reminders are in-memory.** They live as asyncio tasks in the server
  process and are cancelled when the session closes or the process restarts. A
  reminder longer than the process's uptime never fires.
- **Sessions are in-memory**, so the app is effectively single-process. Two
  workers behind a load balancer would not share session state or reminders;
  making it horizontally scalable means moving both into a shared store.
- **No authentication.** Any client that can reach the server can open a
  session at any `session_id`. CORS is wide open for local development and is
  documented in `app/main.py` as something to tighten before deploying — and
  note that browsers do not apply CORS to WebSocket handshakes at all, so an
  origin check there has to be written by hand.
- **The frontend is Chrome-centric.** `MediaRecorder` output formats and
  `speechSynthesis` voice availability vary considerably across browsers;
  development and testing were done in Chrome.
- Push-to-talk, not continuous streaming. Whisper is not a streaming model,
  and partial-ASR streaming is a materially harder problem than the rest of
  this project.

---

## Project structure

```
Voice Agent/
├── app/
│   ├── main.py                 FastAPI app, lifespan, WebSocket endpoint, static mount
│   ├── config.py               Every environment variable, read in exactly one place
│   ├── logging_config.py       Logging setup, configured once at startup
│   ├── audio/
│   │   ├── stt.py              Routing layer: local STT first, Sarvam when the language warrants it
│   │   ├── types.py            The transcription result type shared across providers
│   │   └── providers/
│   │       ├── faster_whisper_stt.py   Local Whisper wrapper and language detection
│   │       └── sarvam.py               Sarvam STT and TTS clients
│   ├── rag/
│   │   ├── ingest.py           Load, chunk, embed and write documents
│   │   ├── store.py            Atlas data layer: $vectorSearch, bulk writes, index management
│   │   ├── embeddings.py       Gemini embeddings, with the query/document asymmetry handled
│   │   └── memory.py           Conversation-memory collection: add_turn, recall
│   ├── agent/
│   │   ├── gemini_agent.py     Retrieval, prompt, bounded tool loop, model fallback
│   │   └── tools.py            Tool implementations and their Gemini declarations
│   ├── services/
│   │   └── session_manager.py  Per-session state and the outbound event queue
│   └── reminders/
│       └── scheduler.py        Background reminder tasks scoped to a session
├── frontend/
│   ├── index.html              Push-to-talk UI, transcript panel, activity feed
│   ├── app.js                  MediaRecorder capture, WebSocket client, speech playback
│   └── styles.css
├── data/
│   ├── documents/              Your source files, ingested into the vector store
│   └── notes.md                Local scratchpad read whole by the read_notes tool
├── scripts/
│   └── ingest_docs.py          CLI to build or rebuild the document index
├── docs/
│   └── architecture.md         Request lifecycle, event protocol, collection schemas
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

---

## Docker

```bash
docker build -t voice-agent .
docker run -p 8000:8000 --env-file .env voice-agent
```

The image does not bake in the Whisper weights, so the first transcription
downloads them. Mount a cache volume (`-v whisper-cache:/home/app/.cache`) or
add a download step to the build if you want a cold start that does not wait
on Hugging Face. The default `base` model needs roughly 1GB of RAM; free tiers
capped at 512MB will not run it.
