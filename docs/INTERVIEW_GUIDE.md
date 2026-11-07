# Interview guide

Answers describe the current implementation. Synthetic test embeddings and unmeasured production behavior are called out directly.

1. **What is RAG?** Retrieval-augmented generation selects relevant source passages first and gives those passages to a model for answer generation. This project uses FAISS top-k retrieval before Ollama.
2. **Why use RAG for long videos?** The whole transcript may exceed context limits and includes irrelevant content. Retrieval bounds the evidence sent with each question.
3. **What does the embedding model do?** It maps a text passage to a dense numeric vector whose geometry represents language similarity. The runtime defaults to a local MiniLM sentence transformer.
4. **What does FAISS do?** It indexes vectors and searches for nearest vectors. Here normalized vectors use an exact inner-product flat index.
5. **Why normalize vectors?** For unit-length vectors, inner product is cosine similarity, so the FAISS score ranks angular similarity without vector magnitude dominating.
6. **What is a chunk?** A bounded section of transcript text that becomes one embedding/document. Its metadata holds source video, language, and time range.
7. **Why overlap chunks?** A short overlap retains context when a definition or idea crosses a split. The default is 64 tokens over a 320-token chunk.
8. **Why 320 tokens?** The default MiniLM model has a 384 wordpiece input window. 320 `cl100k_base` tokens is a conservative configured budget with room for model tokenization and special tokens; tokenizers differ, so it is a practical limit rather than exact equivalence.
9. **How are timestamps preserved?** The chunker tracks character offsets for each transcript segment. Each chunk receives the minimum start and maximum end time for the segments its source-text range overlaps.
10. **Can a citation be wrong even if it links correctly?** Yes. A citation ID can map to the actual retrieved chunk while the generated claim misinterprets it. This is why the code and docs avoid treating citation mapping as factual validation.
11. **How does abstention work?** The chat path compares the best retrieved similarity score with configurable `YTA_MIN_RELEVANCE`. Below the threshold it returns a no-evidence response without invoking Ollama. The score is not calibrated confidence.
12. **Does semantic search need Ollama?** No. Search embeds a query and uses the FAISS index directly. Ollama is needed for chat and summaries.
13. **How are follow-ups handled?** A bounded in-memory session stores the recent dialogue. Pronoun-like follow-ups are optionally rewritten by the local model into a standalone retrieval query, with the recent conversation as input.
14. **How is history bounded?** The runtime stores at most 16 messages and puts at most six recent messages into the prompt. The configured count is an implementation bound, not a model-token guarantee.
15. **How are long summaries bounded?** Transcript chunks are grouped under a character ceiling. The model creates intermediate summaries; large lists are reduced in groups of at most eight before the final formatting prompt.
16. **Why is the transcript cache separate from the FAISS cache?** Transcript JSON allows index rebuilds for a changed model or chunk configuration without always refetching captions. FAISS reuse is validated separately against transcript content and model name.
17. **How are cache collisions avoided?** The path uses the validated video ID, and manifests validate that ID plus transcript hash, embedding model, format version, and checksums.
18. **Why not use pickle persistence?** Python pickle can execute code when loading untrusted data. The implementation uses native FAISS index files and explicit JSON records, and verifies hashes before loading.
19. **Does the checksum authenticate the cache?** No. It detects accidental corruption against the manifest. An attacker who can replace both the files and manifest can recompute hashes; therefore the cache must remain local and trusted.
20. **How does the extension isolate UI styles?** It mounts the assistant UI in a shadow root, reducing CSS collisions with YouTube. It doesn't replace or wrap the player.
21. **How does video switching work?** The content script watches URL changes and listens for YouTube navigation events, re-extracts the video ID, clears displayed conversation state, and initializes the new ID.
22. **How does seeking work?** A returned source carries start seconds. The UI sets `currentTime` on the existing YouTube video element and scrolls it into view.
23. **How does the extension reach the backend?** Its MV3 service worker sends requests to fixed localhost origins. A route prefix check prevents arbitrary path use, and the manifest grants only storage and local host access.
24. **How is SSRF avoided?** Clients send validated video IDs, not URLs. The backend's transcript adapter calls the transcript library with that ID; it does not fetch client-supplied arbitrary URLs.
25. **How is transcript prompt injection mitigated?** Retrieved passages are delimited and explicitly labeled untrusted in system instructions. This is defense in depth, not a guarantee that the model cannot be influenced.
26. **How is transcript text rendered safely?** The content script uses `textContent`, DOM text nodes, and fixed app-created controls. It does not assign caption text to `innerHTML`.
27. **What happens when captions are unavailable?** The transcript adapter wraps upstream errors as `TranscriptUnavailable`; processing status returns a structured failure. The system never creates substitute transcript text.
28. **How does the API expose processing?** Initialization schedules a background task and returns a status object. The client polls the status route for phase, progress, count, or error.
29. **What does the synthetic retrieval test prove?** It tests FAISS ranking and plumbing using deterministic feature-hash embeddings. It does not establish quality of MiniLM embeddings or user-facing semantic search.
30. **What metrics are evaluated?** The script calculates Recall@3, MRR@3, top-1 timestamp accuracy, build and retrieval timing, and a cache round-trip hit on a small fixed synthetic dataset.
31. **What is MRR?** Mean reciprocal rank averages `1 / rank` for the expected passage, assigning zero if it is outside the evaluated result list. Higher means relevant passages tend to appear earlier.
32. **Why exact `IndexFlatIP` rather than approximate search?** A per-video transcript index is small enough for exact ranking and exact results simplify testing. Approximate indexes are more useful for much larger collections.
33. **Why is inference local?** Ollama keeps answer generation on the user's machine, avoids API keys, and avoids sending transcripts to a hosted LLM provider. The model still needs local installation and memory.
34. **What happens if Ollama is unavailable?** Health reports reachability and chat/summary requests return a structured 503 error. Semantic search can continue if indexing has succeeded.
35. **What are the session durability limits?** Sessions are in-memory, video-scoped, and bounded. They disappear on backend restart; persistent multi-user sessions are not implemented.
36. **What is not measured?** Real embedding/indexing times for a neural model, first-answer latency with a real Ollama model, groundedness quality, production scale, and browser end-to-end reliability are not claimed by the synthetic evaluation.
