const SESSION_STORAGE_KEY = "voice-agent-session-id";
const MAX_RECONNECT_DELAY_MS = 30000;
const BASE_RECONNECT_DELAY_MS = 500;

const dom = {
  conversation: document.getElementById("conversation"),
  conversationEmpty: document.getElementById("conversation-empty"),
  activity: document.getElementById("activity"),
  banners: document.getElementById("banners"),
  status: document.getElementById("connection-status"),
  statusText: document.getElementById("connection-text"),
  sessionLabel: document.getElementById("session-label"),
  ptt: document.getElementById("ptt"),
  pttTimer: document.getElementById("ptt-timer"),
  levelFill: document.getElementById("level-fill"),
  textForm: document.getElementById("text-form"),
  textInput: document.getElementById("text-input"),
  textSend: document.getElementById("text-send"),
  muteToggle: document.getElementById("mute-toggle"),
  muteLabel: document.getElementById("mute-label"),
  clearActivity: document.getElementById("clear-activity"),
};

const state = {
  socket: null,
  connected: false,
  reconnectAttempts: 0,
  reconnectTimer: null,
  muted: false,
  recording: false,
  recorder: null,
  mediaStream: null,
  audioContext: null,
  analyser: null,
  levelFrame: null,
  timerInterval: null,
  recordingStartedAt: 0,
  awaitingRegionalAudio: false,
};

function sessionId() {
  let id = sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(SESSION_STORAGE_KEY, id);
  }
  return id;
}

function socketUrl() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const host = location.host || "localhost:8000";
  return `${protocol}//${host}/ws/${sessionId()}`;
}

function setStatus(stateName, text) {
  dom.status.dataset.state = stateName;
  dom.statusText.textContent = text;
}

function setControlsEnabled(enabled) {
  dom.ptt.disabled = !enabled;
  dom.textInput.disabled = !enabled;
  dom.textSend.disabled = !enabled;
}

function connect() {
  clearTimeout(state.reconnectTimer);
  setStatus("connecting", "Connecting");

  let socket;
  try {
    socket = new WebSocket(socketUrl());
  } catch (error) {
    scheduleReconnect();
    return;
  }

  state.socket = socket;

  socket.addEventListener("open", () => {
    state.connected = true;
    state.reconnectAttempts = 0;
    setStatus("open", "Connected");
    setControlsEnabled(true);
  });

  socket.addEventListener("message", (event) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (error) {
      console.warn("Ignoring non-JSON message", event.data);
      return;
    }
    handleEvent(payload);
  });

  socket.addEventListener("close", () => {
    state.connected = false;
    setControlsEnabled(false);
    if (state.recording) stopRecording({ discard: true });
    scheduleReconnect();
  });

  socket.addEventListener("error", () => {
    socket.close();
  });
}

function scheduleReconnect() {
  const attempt = state.reconnectAttempts++;
  const delay = Math.min(
    MAX_RECONNECT_DELAY_MS,
    BASE_RECONNECT_DELAY_MS * 2 ** attempt + Math.random() * 250
  );
  const seconds = Math.round(delay / 100) / 10;
  setStatus("closed", `Disconnected, retrying in ${seconds}s`);
  state.reconnectTimer = setTimeout(connect, delay);
}

function send(payload) {
  if (!state.connected || !state.socket) {
    showBanner("error", "Not connected — reconnecting before that can be sent.");
    return false;
  }
  state.socket.send(payload);
  return true;
}

const eventHandlers = {
  transcript(payload) {
    addUserTurn(payload.text, payload.language, payload.provider);
    addPendingAgentTurn();
  },
  retrieval(payload) {
    addActivityEntry({
      kind: "retrieval",
      title: `${(payload.sources || []).length} source${
        (payload.sources || []).length === 1 ? "" : "s"
      } retrieved`,
      body: buildSourceList(payload.sources || []),
      raw: payload,
    });
  },
  tool_call(payload) {
    addActivityEntry({
      kind: "tool-call",
      title: payload.name,
      body: buildArgList(payload.args || {}),
      raw: payload,
    });
  },
  tool_result(payload) {
    addActivityEntry({
      kind: "tool-result",
      title: payload.name,
      body: buildResultText(payload.result),
      raw: payload,
    });
  },
  agent_answer(payload) {
    resolvePendingAgentTurn(payload.text);
    const isRegional = payload.language && payload.language !== "en";
    if (isRegional) {
      state.awaitingRegionalAudio = true;
    } else {
      speak(payload.text);
    }
  },
  tts_audio(payload) {
    state.awaitingRegionalAudio = false;
    playAudio(payload.audio_base64);
  },
  reminder_fired(payload) {
    showBanner("reminder", payload.message);
    speak(payload.message);
  },
  error(payload) {
    showBanner("error", payload.message || "Something went wrong.");
    resolvePendingAgentTurn(null);
  },
};

function handleEvent(payload) {
  const handler = eventHandlers[payload.type];
  if (!handler) {
    console.warn("Unknown event type", payload.type, payload);
    return;
  }
  handler(payload);
}

function clearEmptyState() {
  if (dom.conversationEmpty) {
    dom.conversationEmpty.remove();
    dom.conversationEmpty = null;
  }
}

function scrollToLatest(container) {
  container.scrollTop = container.scrollHeight;
}

function addUserTurn(text, language, provider) {
  clearEmptyState();

  const turn = document.createElement("article");
  turn.className = "turn turn--user";

  const bubble = document.createElement("p");
  bubble.className = "bubble";
  bubble.textContent = text || "(no speech detected)";
  turn.append(bubble);

  const meta = document.createElement("div");
  meta.className = "turn-meta";
  if (language) meta.append(makeTag(`tag tag--lang`, language));
  if (provider) meta.append(makeTag(`tag tag--provider`, provider));
  if (meta.childElementCount) turn.append(meta);

  dom.conversation.append(turn);
  scrollToLatest(dom.conversation);
}

function addPendingAgentTurn() {
  const turn = document.createElement("article");
  turn.className = "turn turn--agent turn--pending";
  turn.dataset.pending = "true";

  const bubble = document.createElement("p");
  bubble.className = "bubble";
  bubble.textContent = "Thinking";
  turn.append(bubble);

  dom.conversation.append(turn);
  scrollToLatest(dom.conversation);
}

function resolvePendingAgentTurn(text) {
  const pending = dom.conversation.querySelector('[data-pending="true"]');
  if (!pending) {
    if (text) addAgentTurn(text);
    return;
  }
  if (!text) {
    pending.remove();
    return;
  }
  pending.classList.remove("turn--pending");
  delete pending.dataset.pending;
  pending.querySelector(".bubble").textContent = text;
  scrollToLatest(dom.conversation);
}

function addAgentTurn(text) {
  clearEmptyState();
  const turn = document.createElement("article");
  turn.className = "turn turn--agent";
  const bubble = document.createElement("p");
  bubble.className = "bubble";
  bubble.textContent = text;
  turn.append(bubble);
  dom.conversation.append(turn);
  scrollToLatest(dom.conversation);
}

function makeTag(className, text) {
  const tag = document.createElement("span");
  tag.className = className;
  tag.textContent = text;
  return tag;
}

function buildSourceList(sources) {
  const list = document.createElement("ul");
  list.className = "source-list";

  for (const source of sources) {
    const item = document.createElement("li");
    item.className = "source";

    const head = document.createElement("div");
    head.className = "source-head";

    const name = document.createElement("span");
    name.className = "source-name";
    name.textContent = source.source || "unknown source";
    head.append(name);

    if (typeof source.score === "number") {
      const score = document.createElement("span");
      score.className = "score";
      score.textContent = source.score.toFixed(3);
      head.append(score);
    }

    const snippet = document.createElement("p");
    snippet.className = "snippet";
    snippet.textContent = source.text || "";

    item.append(head, snippet);
    list.append(item);
  }

  return list;
}

function buildArgList(args) {
  const list = document.createElement("dl");
  list.className = "arg-list";

  for (const [key, value] of Object.entries(args)) {
    const term = document.createElement("dt");
    term.textContent = key;
    const definition = document.createElement("dd");
    definition.textContent = typeof value === "string" ? value : JSON.stringify(value);
    list.append(term, definition);
  }

  return list;
}

function buildResultText(result) {
  const paragraph = document.createElement("p");
  paragraph.className = "result-text";
  paragraph.textContent = typeof result === "string" ? result : JSON.stringify(result);
  return paragraph;
}

function addActivityEntry({ kind, title, body, raw }) {
  const emptyState = dom.activity.querySelector(".empty-state");
  if (emptyState) emptyState.remove();

  const entry = document.createElement("article");
  entry.className = `entry entry--${kind}`;

  const head = document.createElement("div");
  head.className = "entry-head";

  const kindLabel = document.createElement("span");
  kindLabel.className = "entry-kind";
  kindLabel.textContent = kind.replace("-", " ");

  const titleLabel = document.createElement("span");
  titleLabel.className = "entry-title";
  titleLabel.textContent = title;

  const time = document.createElement("time");
  time.className = "entry-time";
  time.textContent = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  head.append(kindLabel, titleLabel, time);
  entry.append(head);

  if (body) entry.append(body);
  if (raw) entry.append(buildRawDetails(raw));

  dom.activity.append(entry);
  scrollToLatest(dom.activity);
}

function buildRawDetails(payload) {
  const details = document.createElement("details");
  details.className = "raw";

  const summary = document.createElement("summary");
  summary.textContent = "Raw payload";

  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(payload, null, 2);

  details.append(summary, pre);
  return details;
}

function showBanner(kind, message) {
  const banner = document.createElement("div");
  banner.className = `banner banner--${kind}`;

  const body = document.createElement("div");
  body.className = "banner-body";

  const kindLabel = document.createElement("span");
  kindLabel.className = "banner-kind";
  kindLabel.textContent = kind;

  const text = document.createElement("p");
  text.textContent = message;

  body.append(kindLabel, text);

  const dismiss = document.createElement("button");
  dismiss.className = "banner-dismiss";
  dismiss.type = "button";
  dismiss.setAttribute("aria-label", "Dismiss");
  dismiss.textContent = "×";
  dismiss.addEventListener("click", () => banner.remove());

  banner.append(body, dismiss);
  dom.banners.append(banner);
}

function speak(text) {
  if (state.muted || !text || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

function playAudio(base64) {
  if (state.muted || !base64) return;
  window.speechSynthesis?.cancel();

  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);

  const url = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
  const audio = new Audio(url);
  audio.addEventListener("ended", () => URL.revokeObjectURL(url));
  audio.play().catch(() => URL.revokeObjectURL(url));
}

function setMuted(muted) {
  state.muted = muted;
  dom.muteToggle.setAttribute("aria-pressed", String(muted));
  dom.muteLabel.textContent = muted ? "Speech muted" : "Speech on";
  if (muted) window.speechSynthesis?.cancel();
}

async function startRecording() {
  if (state.recording || !state.connected) return;

  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    showBanner("error", "Microphone access was denied, so speech input is unavailable.");
    return;
  }

  const chunks = [];
  state.recorder = new MediaRecorder(state.mediaStream);
  state.recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  });
  state.recorder.addEventListener("stop", () => {
    const blob = new Blob(chunks, { type: state.recorder.mimeType });
    if (!state.discardRecording && blob.size > 0) send(blob);
    state.discardRecording = false;
  });

  state.recorder.start();
  state.recording = true;
  state.recordingStartedAt = performance.now();
  dom.ptt.classList.add("recording");
  dom.ptt.setAttribute("aria-pressed", "true");

  startLevelMeter();
  state.timerInterval = setInterval(updateTimer, 100);
  updateTimer();
}

function stopRecording({ discard = false } = {}) {
  if (!state.recording) return;

  state.discardRecording = discard;
  state.recording = false;
  dom.ptt.classList.remove("recording");
  dom.ptt.setAttribute("aria-pressed", "false");

  clearInterval(state.timerInterval);
  stopLevelMeter();

  if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
  state.mediaStream?.getTracks().forEach((track) => track.stop());
  state.mediaStream = null;
}

function updateTimer() {
  const elapsed = (performance.now() - state.recordingStartedAt) / 1000;
  dom.pttTimer.textContent = `${elapsed.toFixed(1)}s`;
}

function startLevelMeter() {
  state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const source = state.audioContext.createMediaStreamSource(state.mediaStream);
  state.analyser = state.audioContext.createAnalyser();
  state.analyser.fftSize = 512;
  source.connect(state.analyser);

  const samples = new Uint8Array(state.analyser.frequencyBinCount);

  const tick = () => {
    state.analyser.getByteTimeDomainData(samples);
    let sum = 0;
    for (const sample of samples) {
      const centered = (sample - 128) / 128;
      sum += centered * centered;
    }
    const level = Math.min(1, Math.sqrt(sum / samples.length) * 3);
    dom.levelFill.style.width = `${level * 100}%`;
    state.levelFrame = requestAnimationFrame(tick);
  };

  tick();
}

function stopLevelMeter() {
  cancelAnimationFrame(state.levelFrame);
  dom.levelFill.style.width = "0%";
  state.audioContext?.close();
  state.audioContext = null;
  state.analyser = null;
}

dom.ptt.addEventListener("mousedown", startRecording);
dom.ptt.addEventListener("touchstart", (event) => {
  event.preventDefault();
  startRecording();
});
dom.ptt.addEventListener("mouseup", () => stopRecording());
dom.ptt.addEventListener("mouseleave", () => stopRecording());
dom.ptt.addEventListener("touchend", (event) => {
  event.preventDefault();
  stopRecording();
});

document.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.repeat) return;
  if (document.activeElement === dom.textInput) return;
  event.preventDefault();
  startRecording();
});

document.addEventListener("keyup", (event) => {
  if (event.code !== "Space") return;
  if (document.activeElement === dom.textInput) return;
  stopRecording();
});

dom.textForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = dom.textInput.value.trim();
  if (!text) return;
  if (send(JSON.stringify({ type: "text_query", text }))) {
    addUserTurn(text, null, "typed");
    addPendingAgentTurn();
    dom.textInput.value = "";
  }
});

dom.muteToggle.addEventListener("click", () => setMuted(!state.muted));

dom.clearActivity.addEventListener("click", () => {
  dom.activity.replaceChildren();
  const empty = document.createElement("p");
  empty.className = "empty-state";
  empty.textContent = "Retrieval hits, tool calls and results appear here as the agent works.";
  dom.activity.append(empty);
});

dom.sessionLabel.textContent = `Session ${sessionId().slice(0, 8)}`;
setControlsEnabled(false);
setMuted(false);
connect();
