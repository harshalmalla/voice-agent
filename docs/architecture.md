# Architecture

A technical companion to the README. The README explains what the system is
and why the major decisions were taken; this document covers the mechanics —
what happens in what order, the exact wire protocol, the stored schemas, and
the error policy that ties the layers together.

---

## Request lifecycle

One user utterance, end to end.

### 1. Capture

`frontend/app.js` records while the mic button is held. On release,
`MediaRecorder` produces a single blob (typically `audio/webm;codecs=opus`)
which is sent as one **binary** WebSocket frame to `/ws/{session_id}`. Nothing
on the server decodes the container — the bytes go straight to STT.

The typed-input path skips this step entirely and sends a JSON `text_query`
frame instead. Everything downstream is identical, which is what makes the
whole pipeline exercisable without a microphone.

### 2. Receive and dispatch

`_receive_loop` in `app/main.py` reads the raw ASGI message rather than
calling `receive_text()` / `receive_bytes()`, because the connection genuinely
carries both frame kinds and cannot know which is next. `_handle_message`
routes on the presence of `bytes` versus a parsed JSON `type`. Unrecognised
frames are logged and dropped; a stray frame must not end a conversation.

Turns are processed sequentially — the handler is awaited before the next
`receive()` — so two turns on one session cannot race on the history or
interleave their activity events.

### 3. Speech to text, with routing

`app/audio/stt.py` is a routing layer, not an engine.

1. `faster_whisper_stt.transcribe()` runs on every chunk, via
   `asyncio.to_thread` because the work is CPU-bound and would otherwise stall
   every other session on the event loop. It returns a transcript, a detected
   language code, and a detection confidence.
2. If the confidence is below `LANGUAGE_DETECTION_MIN_CONFIDENCE` (default
   0.5), the local transcript is kept — a paid call is not spent on a guess.
3. If the language is `en`, or is not one Sarvam supports, the local
   transcript is kept.
4. Otherwise `sarvam.transcribe()` re-transcribes the **same audio** in that
   language, and its transcript wins.

The result carries `provider`, so the dashboard shows the route that was
actually taken. Supported regional routes: `hi`, `bn`, `ta`, `te`, `kn`, `ml`,
`mr`, `gu`, `pa`, `or`.

An empty transcript is not passed on. A push-to-talk button releases on
silence constantly, and an empty string would spend a retrieval round trip and
a model call on nothing while leaving the client's pending bubble hanging
forever. The handler emits `error` instead, which both explains the problem
and clears that bubble.

### 4. Retrieval

`gemini_agent._retrieve` embeds the question once with
`embeddings.embed_query` — the query-side task type, not the document-side
one; the asymmetry is silent and costs recall if got wrong — then runs two
`$vectorSearch` queries concurrently under `asyncio.gather(...,
return_exceptions=True)`:

| Collection | Filter | Purpose |
| --- | --- | --- |
| `documents` | none | Any chunk of any ingested file. |
| `memory` | `{"session_id": <id>}`, applied **inside** the `$vectorSearch` stage | Older turns of this conversation only. |

`return_exceptions=True` keeps the two failures independent: losing memory
recall should not also lose document retrieval. A total failure returns `[]`
and the turn proceeds without context rather than dying.

Each hit is reshaped to `{text, source, score}`; memory hits get the literal
source label `conversation memory`. The whole list is emitted as one
`retrieval` event before the model is called.

### 5. The agent loop

`_build_contents` assembles the request: a system instruction naming the
language to reply in, a CONTEXT block of the retrieved sources, the bounded
window of recent turns replayed verbatim (`MAX_HISTORY_TURNS`), and the new
question.

Then, up to `MAX_TOOL_ITERATIONS` (default 5) times:

1. Call `client.aio.models.generate_content(...)`. Automatic function calling
   is explicitly disabled — the SDK's built-in loop would hide `tool_call` and
   `tool_result` from the dashboard, cannot name session-bound closures, and
   refuses `async def` tools outright.
2. If the response carries no `function_calls`, take `response.text` and stop.
3. Otherwise, for each call: emit `tool_call`, dispatch through
   `build_tools(session_id)`, emit `tool_result`, and wrap the result as
   `types.Part.from_function_response(name=..., response={"result": ...})`.
4. Append the model's own function-call turn verbatim, then the results under
   role `"user"` (which is what the SDK's own path does), and go round again.

An unknown tool name is returned to the model as an error string rather than
raised, so a hallucinated tool is recoverable on the next pass. Exhausting the
iteration cap logs a warning and returns the best available text.

Every model call is wrapped in retry-with-backoff and then a fallback to the
lighter model, which has its own quota. Total model failure returns a fixed
apology sentence — the user hears something, and the exception does not reach
the socket.

### 6. Answer, memory, speech

`agent_answer` is emitted by the agent itself, not by the handler; the handler
must not emit a second one or the client would resolve the turn twice and
speak it twice. The completed exchange is then written to the `memory`
collection so later turns can recall it.

Back in `main.py`, if the language is not `en`, `_speak` synthesizes the
answer through Sarvam TTS and emits `tts_audio` carrying base64. English
replies carry no audio: the browser's `speechSynthesis` handles them for free,
and the client speaks immediately on `agent_answer` when
`language === "en"`.

A TTS failure here is not a failed turn. The text answer has already been
delivered and is on screen; losing the audio degrades the turn, whereas
raising would additionally show an error banner about a turn that
demonstrably worked.

### 7. Out over the wire

Nothing above writes to the socket. Producers call
`SessionManager.emit(session_id, event)`, a non-blocking `put_nowait` onto a
bounded per-session queue, and `_send_loop` is the sole coroutine that ever
calls `send_json` on that socket. One writer means no interleaved frames, and
a bounded queue means a slow client drops events rather than applying
backpressure to an agent turn or a reminder task.

### Out-of-band: reminders

`set_reminder` schedules an asyncio task in `app/reminders/scheduler.py` that
sleeps and then emits `reminder_fired` into the same session's queue — minutes
or hours later, while the user is silent. If the session is no longer live the
event is dropped harmlessly. These tasks are cancelled by
`SessionManager.close()` on disconnect, which is why that cleanup sits in a
`finally`.

---

## WebSocket event protocol

Endpoint: `ws://<host>/ws/{session_id}`. The `session_id` is generated and
persisted by the client, so a reconnect resumes the same session rather than
replacing it.

### Client to server

| Frame | Payload | Meaning |
| --- | --- | --- |
| binary | Raw `MediaRecorder` blob | One recorded utterance. |
| text (JSON) | `{"type": "text_query", "text": string}` | One typed question. Empty text is ignored. |

Anything else is logged and dropped.

### Server to client

All server frames are JSON text, dispatched by the client on `type` through
its `eventHandlers` map in `frontend/app.js`. An unknown type is logged to the
console and dropped, so a typo on the server side is silent — which is why
these shapes are produced in exactly one place each.

| `type` | Emitted by | Payload | Client behaviour |
| --- | --- | --- | --- |
| `transcript` | `app/main.py` | `{text: string, language: string, provider: string}` | Renders the user turn and queues the pending "Thinking" bubble. `provider` is the STT route (`faster-whisper`, `sarvam`, or `typed`). |
| `retrieval` | `app/agent/gemini_agent.py` | `{sources: [{text: string, source: string, score: number}]}` | Adds a retrieval entry to the activity feed. `score` is the `$vectorSearch` similarity, 0..1. |
| `tool_call` | `app/agent/gemini_agent.py` | `{name: string, args: object}` | Adds a tool-call entry, before the tool runs. |
| `tool_result` | `app/agent/gemini_agent.py` | `{name: string, result: string}` | Adds a tool-result entry, after it runs. |
| `agent_answer` | `app/agent/gemini_agent.py` | `{text: string, language: string}` | Resolves the pending bubble. Speaks immediately via `speechSynthesis` when `language === "en"`; otherwise waits for `tts_audio`. |
| `tts_audio` | `app/main.py` | `{audio_base64: string}` | Decodes and plays the clip. Non-English replies only. |
| `reminder_fired` | `app/reminders/scheduler.py` | `{message: string, reminder_id: string}` | Shows a banner and speaks the message. Arrives unprompted. |
| `error` | `app/main.py` | `{message: string}` | Shows an error banner and clears the pending bubble. |

Base64 rather than a binary frame for audio, for two reasons: the client
expects `payload.audio_base64` and `atob`s it, and the send loop writes JSON
exclusively — an interleaved binary frame would have to bypass the queue that
guarantees a single writer.

---

## MongoDB collections

Database: `MONGODB_DB_NAME` (default `voice_agent`). Collection and index
names are constants in `app/config.py`, not environment variables, because the
Atlas index definition has to be created against those exact names.

### `documents`

Written by `app/rag/ingest.py`; one document per chunk.

| Field | Type | Notes |
| --- | --- | --- |
| `text` | string | The chunk. This is what reaches the prompt. |
| `source` | string | Filename, used for citation and for delete-then-reinsert on re-ingestion. |
| `chunk_index` | int | Position within the file. |
| `embedding` | array[float] | `EMBEDDING_DIMENSION` floats (default 768). |
| `timestamp` | datetime (UTC) | When the file was ingested. |

### `memory`

Written by `app/rag/memory.py`; one document per **completed exchange**, not
per message. An assistant answer retrieved without its question is often
uninterpretable ("Yes, about three weeks"), so both halves live in one
document and there is no `role` field.

| Field | Type | Notes |
| --- | --- | --- |
| `text` | string | The formatted exchange. This is what is embedded and what reaches the prompt. |
| `user_text` | string | The question alone, kept separately so it can be displayed without re-parsing. |
| `assistant_text` | string | The answer alone. |
| `session_id` | string | The recall scope. Declared as a filter field in the index. |
| `embedding` | array[float] | Same width as above. |
| `timestamp` | datetime (UTC) | When the exchange completed. |

### Vector index

Both collections carry the same index, named `vector_index`. The app creates
it at startup via `store.ensure_vector_index`; if the cluster tier refuses
programmatic index management, the same JSON is logged for pasting into the
Atlas UI (Atlas Search, Create Search Index, JSON Editor, Vector Search).

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 768,
      "similarity": "dotProduct"
    },
    { "type": "filter", "path": "session_id" },
    { "type": "filter", "path": "source" }
  ]
}
```

`numDimensions` must match `EMBEDDING_DIMENSION` and the embedding model's
actual output width. A mismatch fails at search time, not at write time.

Both `session_id` and `source` are declared as filter fields because a field
can only be **pre-filtered** if the index says so. That matters more than it
looks. Post-filtering with a `$match` after `$vectorSearch` lets the search
commit to its top-k across the whole collection and then discards the
out-of-scope hits, so memory recall silently returns two results, or one, or
none, as soon as more than one session exists. Passing `filter` inside the
`$vectorSearch` stage applies the predicate during the ANN traversal, so a
full top-k of in-scope documents comes back — and the smaller candidate space
is faster besides.

Two timing behaviours to expect against a live cluster: a newly created index
takes roughly 30 seconds to become queryable, and documents inserted
immediately after creation may not be searchable for a few seconds. Both look
like broken retrieval and are not.

---

## Error policy: infrastructure raises, policy decides

The rule is a single question: *can this module know what a failure means?* If
it cannot, it raises a typed error and lets someone who can decide. If it can
— specifically, if it knows a person is holding a microphone waiting for a
reply — it degrades and logs.

Three modules demonstrate the split.

**`app/rag/store.py` — raises.** It is pure infrastructure. A failed
`$vectorSearch` might mean a missing index, a dimension mismatch, or an
outage, and none of those have a single correct response at this level, so it
wraps them in `VectorStoreError` with a message naming the likely cause and
raises. It makes exactly one deliberate exception: `ensure_vector_index` is
best-effort, because a cluster tier that forbids programmatic index management
is a configuration fact rather than a bug, and refusing to boot over it would
be actively unhelpful.

**`app/audio/stt.py` — decides.** It sits above two providers and knows a
turn is in flight. A Sarvam failure on a regional utterance does not lose the
turn: the local faster-whisper transcript is already in hand, so the layer
falls back to it, marks the provider accordingly, and the conversation
continues with a worse transcript instead of no transcript.

**`app/agent/gemini_agent.py` — decides, at every level.** Retrieval failure
degrades to answering without context. A hallucinated tool name comes back to
the model as a readable error string instead of a `KeyError`. Hitting the
iteration cap returns the best text available. Total model failure returns a
fixed apology. Nothing propagates into the WebSocket handler, because an
exception there means the user hears nothing at all and cannot distinguish a
Mongo outage from a broken app.

Two supporting conventions:

- **Tools never raise.** Every callable in `app/agent/tools.py` returns its
  failure as a string, because the caller is a language model and a string is
  something it can read and act on.
- **`_receive_loop` catches everything anyway.** Below that line the codebase
  already degrades rather than raises, so an exception reaching it is by
  definition unanticipated. Letting it propagate would close the socket and
  send the client into reconnect backoff over what was probably one bad turn.
  It is logged with a traceback, the socket stays open, and the user gets an
  `error` event.
