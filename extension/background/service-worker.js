const API_BASES = ["http://127.0.0.1:8000", "http://localhost:8000"];

async function requestApi(path, method, body) {
  if (typeof path !== "string" || !path.startsWith("/api/") || path.includes("..")) {
    throw new Error("Invalid local API route.");
  }

  let lastError = "The local backend is unavailable.";
  for (const base of API_BASES) {
    try {
      const response = await fetch(`${base}${path}`, {
        method,
        headers: body === undefined ? {} : { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
        cache: "no-store"
      });
      const data = response.status === 204 ? null : await response.json();
      if (!response.ok) {
        const error = data?.error?.message || data?.detail || `Backend returned ${response.status}.`;
        return { ok: false, status: response.status, error: String(error) };
      }
      return { ok: true, data };
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
  }
  return { ok: false, status: 0, error: `${lastError} Start the YouTube AI Assistant backend.` };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.type !== "YTA_API_REQUEST") return undefined;
  requestApi(message.path, message.method || "GET", message.body)
    .then(sendResponse)
    .catch((error) => sendResponse({ ok: false, status: 0, error: String(error) }));
  return true;
});
