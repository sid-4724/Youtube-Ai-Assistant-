((root) => {
  const VIDEO_ID_RE = /^[A-Za-z0-9_-]{11}$/;

  function extractVideoId(href) {
    try {
      const url = new URL(href);
      const match = /^\/(?:shorts|embed)\/([A-Za-z0-9_-]{11})(?:\/|$)/.exec(url.pathname);
      const candidate = url.searchParams.get("v") || match?.[1] || "";
      return VIDEO_ID_RE.test(candidate) ? candidate : "";
    } catch {
      return "";
    }
  }

  function transitionVideo(state, href) {
    const videoId = extractVideoId(href);
    if (state.videoId === videoId) return false;
    state.videoId = videoId;
    state.sessionId = "";
    state.messages = [];
    state.ready = false;
    state.searchResults = [];
    state.summary = "";
    state.notes = "";
    state.progress = "";
    state.error = "";
    state.lastAnswer = "";
    return true;
  }

  root.YTA = root.YTA || {};
  root.YTA.extractVideoId = extractVideoId;
  root.YTA.transitionVideo = transitionVideo;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { extractVideoId, transitionVideo };
  }
})(globalThis);
