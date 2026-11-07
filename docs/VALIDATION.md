# Validation report

Validation was run on an Apple Silicon Mac (`arm64`) with Python 3.13.9 on 2026-11-07.

## Automated checks

| Check | Result |
|---|---|
| Backend pytest suite | **14 passed** |
| Extension state tests (`node --test`) | **3 passed** |
| JavaScript syntax checks | **Passed** for content script, service worker, and video-state module |
| Python compile check | **Passed** |
| Ruff | **Passed** |
| `pip check` | **No broken requirements found** |
| FastAPI local HTTP smoke test | **Passed**: Uvicorn started and `/api/health` returned HTTP 200 |
| Ollama reachability in smoke test | **Unavailable**; health correctly returned `ollama_reachable: false` |

The automated suite covers transcript cleanup and extraction failures, timestamp-preserving chunking, a two-hour synthetic transcript, FAISS ranking, safe cache persistence/invalidation, video-scoped caching, follow-up reformulation, grounded citation-to-time mapping, abstention, bounded hierarchical summaries, API validation, local model failure handling, and extension video-ID/session-state transitions.

## Deterministic retrieval evaluation

Command: `python3.13 evaluation/run.py`

The evaluation has six hand-written synthetic transcript passages and five corresponding questions. It uses deterministic feature-hashing embeddings, not the configured sentence transformer. Timings below are one run and describe only this tiny local fixture.

| Metric | Measured result |
|---|---:|
| Recall@3 | 1.00 (5/5 expected passages retrieved in the top 3) |
| MRR@3 | 1.00 |
| Top-1 timestamp accuracy | 1.00 (5/5) |
| Transcript cleaning and chunking | 76.816 ms; 3 chunks |
| Deterministic document embedding | 0.204 ms; 1 batch |
| Total index build | 1.171 ms |
| Mean retrieval latency | 0.304 ms |
| FAISS cache round-trip | Hit; 0 document embedding calls on load; 0.844 ms cache load |
| Time to first answer | Not measured; this evaluation does not call Ollama |
| Grounded answer quality | Not measured |

These small-fixture timings are not representative of real transcript length, sentence-transformer inference, first-time model loading, an Ollama answer, or production usage. A live timing command is provided in `evaluation/benchmark_live.py`; it was not run because Ollama and Chrome are not installed in this environment.

## Not run

- Live YouTube transcript retrieval: no test video was supplied, and transcript availability can vary by video and network.
- Real Hugging Face model download/inference: unit and evaluation tests deliberately use local deterministic embeddings.
- Real Ollama chat or summary generation: Ollama is not installed; the outage path is covered with a deterministic failure test.
- Browser end-to-end UI and player seeking: Chrome/Chromium is not installed. Video-ID and state transitions are unit-tested in Node, but player integration still needs a manual Chrome check.
