const test = require("node:test");
const assert = require("node:assert/strict");
const { extractVideoId, transitionVideo } = require("../content/video-state.js");

test("extracts standard watch, shorts, and embed video IDs", () => {
  assert.equal(extractVideoId("https://www.youtube.com/watch?v=abcdefghijk&t=35"), "abcdefghijk");
  assert.equal(extractVideoId("https://www.youtube.com/shorts/lmnopqrstuv"), "lmnopqrstuv");
  assert.equal(extractVideoId("https://www.youtube.com/embed/12345678901"), "12345678901");
});

test("rejects malformed, unrelated, and invalid URLs", () => {
  assert.equal(extractVideoId("https://www.youtube.com/watch?v=too-short"), "");
  assert.equal(extractVideoId("https://www.youtube.com/@channel"), "");
  assert.equal(extractVideoId("not a URL"), "");
});

test("video navigation resets video-scoped state once per actual video change", () => {
  const state = {
    videoId: null,
    sessionId: "session-a",
    messages: [{ role: "assistant", text: "old" }],
    ready: true,
    searchResults: [{ excerpt: "old" }],
    summary: "old summary",
    notes: "old notes",
    progress: "busy",
    error: "old error",
    lastAnswer: "old answer"
  };

  assert.equal(transitionVideo(state, "https://www.youtube.com/watch?v=abcdefghijk"), true);
  assert.equal(state.videoId, "abcdefghijk");
  assert.equal(state.sessionId, "");
  assert.deepEqual(state.messages, []);
  assert.equal(state.ready, false);
  assert.deepEqual(state.searchResults, []);
  assert.equal(state.summary, "");
  assert.equal(state.notes, "");
  assert.equal(state.progress, "");
  assert.equal(state.error, "");
  assert.equal(state.lastAnswer, "");

  state.sessionId = "session-a";
  assert.equal(transitionVideo(state, "https://www.youtube.com/watch?v=abcdefghijk&t=8"), false);
  assert.equal(state.sessionId, "session-a");
  assert.equal(transitionVideo(state, "https://www.youtube.com/watch?v=lmnopqrstuv"), true);
  assert.equal(state.videoId, "lmnopqrstuv");
  assert.equal(state.sessionId, "");
});
