# Learning guide

This order follows the data from a YouTube page through the RAG backend and back to the user.

## 1. See the whole application

Read the Mermaid diagram in `README.md`, then [ARCHITECTURE.md](ARCHITECTURE.md). Notice the boundary: the content script talks to a service worker, and only that worker calls the local HTTP API.

## 2. Follow video initialization

Start with `extension/content/content.js` and search for `ensureVideo`. It sends the current video ID to `/api/videos/initialize`, then polls `/status`. The backend routes are in `backend/app/main.py`; the orchestration is in `backend/app/services/runtime.py`.

## 3. Follow caption data

Open `backend/app/services/transcripts.py`. The external library returns timestamped caption snippets. `Transcript` and `TranscriptSegment` are the application's small data model. Read `test_transcripts.py` to see cleaning, duplicate handling, and failure cases.

## 4. Understand chunks and time metadata

Read `backend/app/rag/chunking.py`. It joins source text while tracking each segment's character offsets. LangChain chooses token-aware boundaries; those offsets map each chunk back to the segment timestamps. `test_transcripts.py` and the two-hour fixture test verify late timestamps remain attached.

## 5. Learn embeddings and FAISS

Read [RAG_EXPLAINED.md](RAG_EXPLAINED.md), then `backend/app/rag/embeddings.py` and `backend/app/rag/cache.py`. Embeddings turn text into vectors; FAISS ranks vectors. Cache manifests include video ID, transcript hash, model name, and file checksums. The tests use deterministic fake embeddings so they remain offline and reproducible.

## 6. Trace a question

In `runtime.py`, follow `answer`, `_standalone_question`, `_retrieve`, and `_source`. Retrieval produces documents and scores; source metadata is converted to timestamp links; only retrieved passages go to Ollama. `test_runtime.py` verifies citation mapping and abstention behavior.

## 7. Follow the response into the page

Read `backend/app/models/schemas.py` for response objects and `main.py` for HTTP routes. In the content script, `renderMessages` maps source IDs into clickable timestamp buttons and `seekTo` changes the existing YouTube video element's current time.

## 8. Read summaries and sessions

The runtime stores bounded per-video session histories. For long transcripts, `summarize` makes bounded intermediate summaries, reduces them in groups, then formats the final result. The model is local, and conversations reset with the backend process.

## 9. Change one setting at a time

Try `YTA_CHUNK_SIZE_TOKENS`, `YTA_CHUNK_OVERLAP_TOKENS`, `YTA_RETRIEVAL_TOP_K`, and `YTA_MIN_RELEVANCE` in `backend/.env`. Re-run tests and the synthetic evaluation. Then test with a real video and record the model, machine, video ID, and timing; don't confuse synthetic evaluation metrics with real-video performance.

## 10. Extend the project

Useful next steps include browser automation for YouTube SPA interactions, real sentence-transformer retrieval evaluation on human-labeled examples, token-count rather than character-count summary batches, streaming chat, configurable language selection in the UI, durable multi-user sessions, and a deletion control for individual cached videos.
