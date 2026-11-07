(() => {
  if (window.__YTA_ASSISTANT_MOUNTED__) return;
  window.__YTA_ASSISTANT_MOUNTED__ = true;

  const state = {
    videoId: null,
    sessionId: "",
    messages: [],
    activeTab: "chat",
    busy: false,
    ready: false,
    theme: "light",
    searchResults: [],
    summary: "",
    notes: "",
    progress: "",
    error: "",
    lastAnswer: ""
  };

  const shell = document.createElement("div");
  shell.id = "yta-assistant-root";
  shell.style.all = "initial";
  (document.body || document.documentElement).append(shell);
  const shadow = shell.attachShadow({ mode: "open" });

  const styleLink = document.createElement("link");
  styleLink.rel = "stylesheet";
  styleLink.href = chrome.runtime.getURL("content/styles.css");
  shadow.append(styleLink);

  const ui = document.createElement("div");
  ui.innerHTML = `
    <button class="launcher" type="button" aria-label="Open YouTube AI Assistant" title="YouTube AI Assistant">✦</button>
    <aside class="panel" aria-label="YouTube AI Assistant" hidden>
      <header class="topbar">
        <div class="brand"><span class="brand-mark">✦</span><div><strong>Video assistant</strong><small class="video-label">Current video</small></div></div>
        <div class="top-actions">
          <button class="icon-button theme-toggle" type="button" title="Toggle theme" aria-label="Toggle theme">◐</button>
          <button class="icon-button collapse" type="button" title="Collapse assistant" aria-label="Collapse assistant">×</button>
        </div>
      </header>
      <nav class="tabs" aria-label="Assistant tools">
        <button data-tab="chat" class="tab active" type="button">Ask</button>
        <button data-tab="summary" class="tab" type="button">Summary</button>
        <button data-tab="search" class="tab" type="button">Search</button>
        <button data-tab="notes" class="tab" type="button">Notes</button>
      </nav>
      <div class="status" role="status" aria-live="polite" hidden></div>
      <div class="error" role="alert" hidden><span class="error-text"></span><button class="retry" type="button">Retry</button></div>
      <section class="view chat-view">
        <div class="messages" aria-live="polite"></div>
        <form class="chat-form">
          <label class="sr-only" for="yta-question">Ask about this video</label>
          <textarea id="yta-question" rows="2" maxlength="4000" placeholder="Ask about this video…" required></textarea>
          <div class="composer-actions"><button class="clear-chat text-button" type="button">Clear chat</button><button class="copy-answer text-button" type="button" disabled>Copy answer</button><button class="send primary" type="submit" aria-label="Send question">Send ↑</button></div>
        </form>
      </section>
      <section class="view summary-view" hidden>
        <div class="intro"><span class="eyebrow">MAKE SENSE OF THE VIDEO</span><h2>Get the key ideas</h2><p>Summaries are grounded in the available transcript.</p></div>
        <div class="summary-actions">
          <button data-kind="quick" class="summary-button" type="button"><span>Quick summary</span><small>5–8 key points</small></button>
          <button data-kind="detailed" class="summary-button" type="button"><span>Detailed summary</span><small>Sections and explanations</small></button>
          <button data-kind="chapters" class="summary-button" type="button"><span>Chapter summaries</span><small>Chronological with timestamps</small></button>
        </div>
        <article class="summary-output" hidden></article>
      </section>
      <section class="view search-view" hidden>
        <div class="intro"><span class="eyebrow">FIND A MOMENT</span><h2>Search this video</h2><p>Describe a concept or example to find where it appears.</p></div>
        <form class="search-form"><label class="sr-only" for="yta-search">Search transcript</label><input id="yta-search" maxlength="1000" placeholder="e.g. Where is overfitting explained?" required><button class="primary" type="submit" aria-label="Search transcript">Find</button></form>
        <div class="search-results"></div>
      </section>
      <section class="view notes-view" hidden>
        <div class="intro"><span class="eyebrow">STUDY MODE</span><h2>Turn this into notes</h2><p>Build revision notes from indexed transcript passages.</p></div>
        <button class="make-notes primary wide" type="button">Create study notes</button>
        <article class="notes-output" hidden></article>
      </section>
      <footer class="footer"><span>Local transcript RAG</span><span class="privacy-dot"></span><span>Processed on this device</span></footer>
    </aside>`;
  shadow.append(ui);

  const $ = (selector) => shadow.querySelector(selector);
  const panel = $(".panel");
  const launcher = $(".launcher");

  function api(path, method = "GET", body) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ type: "YTA_API_REQUEST", path, method, body }, (response) => {
        if (chrome.runtime.lastError) {
          resolve({ ok: false, error: chrome.runtime.lastError.message });
          return;
        }
        resolve(response || { ok: false, error: "The extension could not contact its local backend." });
      });
    });
  }

  function setBusy(busy) {
    state.busy = busy;
    shadow.querySelectorAll(".send, .search-form button, .summary-button, .make-notes").forEach((button) => {
      button.disabled = busy || !state.ready;
    });
    shadow.querySelectorAll(".chat-form textarea, .search-form input").forEach((input) => { input.disabled = busy || !state.ready; });
    $(".send").textContent = busy ? "Working…" : "Send ↑";
  }

  function showError(message, retry = true) {
    state.error = message;
    $(".error-text").textContent = message;
    $(".error").hidden = false;
    $(".retry").hidden = !retry;
  }

  function clearError() {
    state.error = "";
    $(".error").hidden = true;
    $(".error-text").textContent = "";
  }

  function setProgress(message) {
    state.progress = message;
    const status = $(".status");
    status.textContent = message;
    status.hidden = !message;
  }

  function renderMessages() {
    const container = $(".messages");
    container.replaceChildren();
    if (!state.messages.length) {
      const welcome = document.createElement("div");
      welcome.className = "welcome-card";
      welcome.innerHTML = '<span class="welcome-icon">✦</span><h2>What would you like to understand?</h2><p>Ask a question and I’ll look for the answer in this video’s transcript.</p>';
      container.append(welcome);
    }
    for (const message of state.messages) {
      const row = document.createElement("article");
      row.className = `message ${message.role}`;
      const avatar = document.createElement("span");
      avatar.className = "avatar";
      avatar.textContent = message.role === "assistant" ? "✦" : "You";
      const body = document.createElement("div");
      body.className = "message-body";
      const label = document.createElement("span");
      label.className = "message-label";
      label.textContent = message.role === "assistant" ? "Assistant" : "You";
      const text = document.createElement("div");
      text.className = "message-text";
      const answer = String(message.text || "");
      const sourceMap = new Map((message.sources || []).map((source) => [source.source_id, source]));
      const citationPattern = /\[(S\d+)\]/g;
      let cursor = 0;
      let match;
      while ((match = citationPattern.exec(answer))) {
        text.append(document.createTextNode(answer.slice(cursor, match.index)));
        const source = sourceMap.get(match[1]);
        if (source) {
          const button = document.createElement("button");
          button.className = "inline-timestamp";
          button.type = "button";
          button.textContent = source.timestamp;
          button.title = "Seek to this timestamp";
          button.addEventListener("click", () => seekTo(source.start_seconds));
          text.append(button);
        } else {
          text.append(document.createTextNode(match[0]));
        }
        cursor = citationPattern.lastIndex;
      }
      text.append(document.createTextNode(answer.slice(cursor)));
      body.append(label, text);
      if (message.role === "assistant" && message.sources?.length) {
        const sources = document.createElement("div");
        sources.className = "source-list";
        for (const source of message.sources.slice(0, 5)) {
          const button = document.createElement("button");
          button.className = "source-chip";
          button.type = "button";
          button.textContent = `↗ ${source.timestamp}`;
          button.title = source.excerpt;
          button.addEventListener("click", () => seekTo(source.start_seconds));
          sources.append(button);
        }
        body.append(sources);
      }
      row.append(avatar, body);
      container.append(row);
    }
    container.scrollTop = container.scrollHeight;
    const lastAssistant = [...state.messages].reverse().find((item) => item.role === "assistant");
    $(".copy-answer").disabled = !lastAssistant;
    state.lastAnswer = lastAssistant?.text || "";
  }

  function seekTo(seconds) {
    const video = document.querySelector("video.html5-main-video") || document.querySelector("video");
    if (!video || !Number.isFinite(Number(seconds))) {
      showError("The YouTube player is not ready yet.", false);
      return;
    }
    try {
      video.currentTime = Math.max(0, Number(seconds));
      video.scrollIntoView({ behavior: "smooth", block: "center" });
      video.focus?.({ preventScroll: true });
    } catch {
      showError("Could not seek the YouTube player to that timestamp.", false);
    }
  }

  function addMessage(role, text, sources = []) {
    state.messages.push({ role, text, sources });
    renderMessages();
  }

  function showTab(tab) {
    state.activeTab = tab;
    shadow.querySelectorAll(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
    shadow.querySelectorAll(".view").forEach((view) => { view.hidden = !view.classList.contains(`${tab}-view`); });
  }

  function renderResults(results) {
    const container = $(".search-results");
    container.replaceChildren();
    if (!results.length) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "No matching transcript passages were found.";
      container.append(empty);
      return;
    }
    results.forEach((result) => {
      const card = document.createElement("article");
      card.className = "result-card";
      const heading = document.createElement("div");
      heading.className = "result-heading";
      const time = document.createElement("button");
      time.className = "timestamp-button";
      time.type = "button";
      time.textContent = `▶ ${result.timestamp}`;
      time.addEventListener("click", () => seekTo(result.start_seconds));
      const confidence = document.createElement("span");
      confidence.className = "confidence";
      confidence.textContent = `${Math.round(Math.max(0, Math.min(1, result.relevance || 0)) * 100)}% match`;
      const excerpt = document.createElement("p");
      excerpt.className = "excerpt";
      excerpt.textContent = result.excerpt;
      heading.append(time, confidence);
      card.append(heading, excerpt);
      container.append(card);
    });
  }

  async function ensureVideo(force = false) {
    if (!state.videoId) {
      showError("Open a standard YouTube video page to use the assistant.", false);
      return false;
    }
    clearError();
    state.ready = false;
    setBusy(true);
    setProgress("Connecting to the local assistant…");
    const started = await api("/api/videos/initialize", "POST", {
      video_id: state.videoId,
      languages: ["en"],
      force
    });
    if (!started.ok) {
      setProgress("");
      showError(started.error || "Could not initialize this video.");
      setBusy(false);
      return false;
    }

    const expectedVideo = state.videoId;
    let status = started.data;
    for (let attempt = 0; attempt < 240 && expectedVideo === state.videoId; attempt += 1) {
      if (status.status === "ready") break;
      if (status.status === "error") {
        setProgress("");
        showError(status.error?.message || status.message || "Transcript processing failed.");
        setBusy(false);
        return false;
      }
      setProgress(status.message || "Processing transcript…");
      await new Promise((resolve) => setTimeout(resolve, 1250));
      const polled = await api(`/api/videos/${encodeURIComponent(expectedVideo)}/status`);
      if (!polled.ok) {
        setProgress("");
        showError(polled.error || "Could not read video processing status.");
        setBusy(false);
        return false;
      }
      status = polled.data;
    }
    if (expectedVideo !== state.videoId) return false;
    if (status.status !== "ready") {
      setProgress("");
      showError("Video processing is taking longer than expected. Retry to check again.");
      setBusy(false);
      return false;
    }

    if (!state.sessionId) {
      const session = await api("/api/sessions", "POST", { video_id: state.videoId });
      if (!session.ok) {
        setProgress("");
        showError(session.error || "Could not start a chat session.");
        setBusy(false);
        return false;
      }
      state.sessionId = session.data.session_id;
    }
    state.ready = true;
    setProgress(`Ready · ${status.chunk_count || 0} passages indexed`);
    setBusy(false);
    window.setTimeout(() => { if (state.ready) setProgress(""); }, 4000);
    return true;
  }

  function resetForVideo() {
    if (!globalThis.YTA?.transitionVideo(state, location.href)) return;
    $(".video-label").textContent = state.videoId ? `Video · ${state.videoId}` : "Open a video to begin";
    $(".summary-output").replaceChildren();
    $(".summary-output").hidden = true;
    $(".notes-output").replaceChildren();
    $(".notes-output").hidden = true;
    $(".search-results").replaceChildren();
    setProgress("");
    clearError();
    renderMessages();
    setBusy(true);
    if (panel.hidden) {
      // The player page stays untouched until a user opens the assistant.
      return;
    }
    ensureVideo();
  }

  launcher.addEventListener("click", () => {
    panel.hidden = false;
    launcher.hidden = true;
    if (!state.ready) ensureVideo();
    showTab(state.activeTab);
    $("#yta-question").focus();
  });
  $(".collapse").addEventListener("click", () => {
    panel.hidden = true;
    launcher.hidden = false;
  });
  shadow.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => showTab(button.dataset.tab)));
  $(".theme-toggle").addEventListener("click", async () => {
    state.theme = state.theme === "light" ? "dark" : "light";
    shell.dataset.theme = state.theme;
    try { await chrome.storage.local.set({ ytaTheme: state.theme }); } catch { /* theme stays for this page */ }
  });
  chrome.storage.local.get("ytaTheme", (value) => {
    if (value?.ytaTheme === "dark" || value?.ytaTheme === "light") {
      state.theme = value.ytaTheme;
      shell.dataset.theme = state.theme;
    }
  });

  $(".chat-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.busy) return;
    const input = $("#yta-question");
    const question = input.value.trim();
    if (!question) return;
    if (!state.ready && !(await ensureVideo())) return;
    clearError();
    input.value = "";
    addMessage("user", question);
    setBusy(true);
    setProgress("Searching the transcript and preparing an answer…");
    const response = await api(`/api/videos/${encodeURIComponent(state.videoId)}/ask`, "POST", {
      video_id: state.videoId,
      session_id: state.sessionId,
      question
    });
    setProgress("");
    setBusy(false);
    if (!response.ok) {
      showError(response.error || "The answer could not be generated.");
      return;
    }
    addMessage("assistant", response.data.answer, response.data.sources || []);
  });

  $(".clear-chat").addEventListener("click", async () => {
    if (state.sessionId) {
      await api(`/api/sessions/${encodeURIComponent(state.sessionId)}?video_id=${encodeURIComponent(state.videoId)}`, "DELETE");
    }
    state.sessionId = "";
    state.messages = [];
    renderMessages();
    clearError();
    if (state.ready) await ensureVideo();
  });
  $(".copy-answer").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(state.lastAnswer);
      setProgress("Answer copied to clipboard.");
      window.setTimeout(() => setProgress(""), 1800);
    } catch {
      showError("Clipboard access was unavailable on this page.", false);
    }
  });
  $(".retry").addEventListener("click", () => { clearError(); ensureVideo(); });

  $(".search-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.busy) return;
    if (!state.ready && !(await ensureVideo())) return;
    const query = $("#yta-search").value.trim();
    if (!query) return;
    setBusy(true);
    clearError();
    setProgress("Searching indexed transcript passages…");
    const response = await api(`/api/videos/${encodeURIComponent(state.videoId)}/search`, "POST", {
      video_id: state.videoId,
      query,
      limit: 5
    });
    setBusy(false);
    setProgress("");
    if (!response.ok) {
      showError(response.error || "Search failed.");
      return;
    }
    state.searchResults = response.data.results || [];
    renderResults(state.searchResults);
  });

  async function generateSummary(kind, destination) {
    if (state.busy) return;
    if (!state.ready && !(await ensureVideo())) return;
    setBusy(true);
    clearError();
    setProgress("Reading transcript summaries…");
    const response = await api(`/api/videos/${encodeURIComponent(state.videoId)}/summarize`, "POST", {
      video_id: state.videoId,
      kind
    });
    setBusy(false);
    setProgress("");
    if (!response.ok) {
      showError(response.error || "Summary generation failed.");
      return;
    }
    destination.textContent = response.data.summary;
    destination.hidden = false;
    destination.dataset.processed = String(response.data.chunks_processed);
  }

  shadow.querySelectorAll(".summary-button").forEach((button) => button.addEventListener("click", () => {
    generateSummary(button.dataset.kind, $(".summary-output"));
  }));
  $(".make-notes").addEventListener("click", () => generateSummary("notes", $(".notes-output")));

  const themeMedia = window.matchMedia("(prefers-color-scheme: dark)");
  if (!shell.dataset.theme) shell.dataset.theme = themeMedia.matches ? "dark" : "light";
  resetForVideo();
  let lastHref = location.href;
  const navigationObserver = new MutationObserver(() => {
    if (location.href !== lastHref) {
      lastHref = location.href;
      resetForVideo();
    }
  });
  navigationObserver.observe(document.documentElement, { childList: true, subtree: true });
  window.addEventListener("yt-navigate-finish", () => {
    if (location.href !== lastHref) lastHref = location.href;
    resetForVideo();
  });
  window.addEventListener("popstate", resetForVideo);
})();
