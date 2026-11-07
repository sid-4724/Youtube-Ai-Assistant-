"""Repeatable FAISS retrieval metrics on a tiny synthetic transcript corpus.

Uses deterministic feature hashing so it runs offline and needs no model download.
It measures the index/search/cache plumbing, not sentence-transformer quality.
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from app.rag.cache import VideoCache, build_faiss_store, canonical_hash  # noqa: E402


class HashEmbeddings(Embeddings):
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.document_embedding_ms = 0.0
        self.document_batches = 0

    def _embed(self, text: str) -> list[float]:
        result = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            digest = hashlib.sha256(token.encode()).digest()
            result[int.from_bytes(digest[:4], "little") % self.dimension] += 1.0
        length = sum(value * value for value in result) ** 0.5 or 1.0
        return [value / length for value in result]

    def embed_documents(self, texts):
        start = time.perf_counter()
        result = [self._embed(text) for text in texts]
        self.document_embedding_ms += (time.perf_counter() - start) * 1000
        self.document_batches += 1
        return result

    def embed_query(self, text):
        return self._embed(text)


def main():
    data = json.loads((Path(__file__).with_name("dataset.json")).read_text("utf-8"))
    video_id = "evalvideo01"
    from app.rag.chunking import transcript_to_documents  # noqa: E402
    from app.services.transcripts import normalize_snippets  # noqa: E402

    transcript_start = time.perf_counter()
    transcript = normalize_snippets(
        video_id,
        [
            {
                "text": item["text"],
                "start": item["start_seconds"],
                "duration": item["end_seconds"] - item["start_seconds"],
            }
            for item in data["transcript"]
        ],
    )
    chunked_documents = transcript_to_documents(
        transcript, video_id, chunk_size_tokens=48, chunk_overlap_tokens=8
    )
    transcript_processing_ms = (time.perf_counter() - transcript_start) * 1000
    # Evaluation passages are pre-labeled corpus documents; keep their labels intact for ranking.
    documents = [
        Document(
            page_content=item["text"],
            metadata={
                "video_id": video_id,
                "chunk_id": item["chunk_id"],
                "start_seconds": item["start_seconds"],
                "end_seconds": item["end_seconds"],
            },
        )
        for item in data["transcript"]
    ]
    embeddings = HashEmbeddings()
    start = time.perf_counter()
    store = build_faiss_store(documents, embeddings)
    index_seconds = time.perf_counter() - start

    ranks, latencies, timestamp_hits = [], [], 0
    for item in data["questions"]:
        start = time.perf_counter()
        results = store.similarity_search_with_score(item["question"], k=3)
        latencies.append((time.perf_counter() - start) * 1000)
        chunk_ids = [document.metadata["chunk_id"] for document, _ in results]
        rank = chunk_ids.index(item["relevant_chunk_id"]) + 1 if item["relevant_chunk_id"] in chunk_ids else None
        ranks.append(rank)
        if results and results[0][0].metadata["start_seconds"] == item["expected_start_seconds"]:
            timestamp_hits += 1

    with __import__("tempfile").TemporaryDirectory(prefix="yta-eval-") as temp:
        cache = VideoCache(Path(temp), ttl_seconds=600)
        transcript_hash = canonical_hash(data["transcript"])
        cache.save_index(video_id, transcript_hash, "deterministic-hash-v1", documents, store)
        start = time.perf_counter()
        cached = cache.load_index(video_id, transcript_hash, "deterministic-hash-v1", embeddings)
        cache_load_ms = (time.perf_counter() - start) * 1000
        cache_hit = cached is not None

    relevant_ranks = [rank for rank in ranks if rank]
    count = len(data["questions"])
    result = {
        "dataset_questions": count,
        "recall_at_3": round(len(relevant_ranks) / count, 4),
        "mrr_at_3": round(sum(1 / rank for rank in relevant_ranks) / count, 4),
        "top_1_timestamp_accuracy": round(timestamp_hits / count, 4),
        "transcript_clean_and_chunk_ms": round(transcript_processing_ms, 3),
        "chunks_produced": len(chunked_documents),
        "embedding_generation_ms": round(embeddings.document_embedding_ms, 3),
        "embedding_batches": embeddings.document_batches,
        "index_build_ms": round(index_seconds * 1000, 3),
        "mean_retrieval_latency_ms": round(sum(latencies) / len(latencies), 3),
        "cache_round_trip_hit": cache_hit,
        "cache_load_ms": round(cache_load_ms, 3),
        "cache_load_document_embed_calls": 0,
        "time_to_first_answer_ms": None,
        "time_to_first_answer_note": "Not measured: this offline fixture does not call Ollama; use benchmark_live.py with a running local backend.",
        "embedding": "deterministic feature hashing (offline evaluation only)",
        "answer_quality": "not measured; review grounded answers manually with the protocol in README.md",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
