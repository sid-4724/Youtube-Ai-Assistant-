import os
from unittest.mock import patch

from langchain_core.documents import Document

from app.rag.cache import VideoCache, build_faiss_store, canonical_hash


def test_faiss_search_returns_relevant_document(lexical_embeddings):
    documents = [
        Document(page_content="gradient descent updates model weights using the loss gradient", metadata={"video_id": "abcdefghijk", "start_seconds": 20.0, "end_seconds": 24.0}),
        Document(page_content="a convolutional neural network applies learned image filters", metadata={"video_id": "abcdefghijk", "start_seconds": 120.0, "end_seconds": 128.0}),
        Document(page_content="overfitting occurs when a model memorizes training data", metadata={"video_id": "abcdefghijk", "start_seconds": 240.0, "end_seconds": 245.0}),
    ]
    store = build_faiss_store(documents, lexical_embeddings)
    result, score = store.similarity_search_with_score("how does gradient descent update weights", k=1)[0]
    assert "gradient descent" in result.page_content
    assert score > 0.2


def test_index_cache_round_trip_is_video_specific_and_pickle_free(tmp_path, lexical_embeddings):
    documents = [Document(page_content="gradient descent minimizes loss", metadata={"video_id": "abcdefghijk", "start_seconds": 1.0, "end_seconds": 4.0})]
    store = build_faiss_store(documents, lexical_embeddings)
    cache = VideoCache(tmp_path, ttl_seconds=900)
    digest = canonical_hash({"segments": ["fixture"]})
    cache.write_transcript("abcdefghijk", {"video_id": "abcdefghijk", "segments": []})
    cache.save_index("abcdefghijk", digest, "test-embeddings", documents, store)
    if os.name == "posix":
        assert cache.video_dir("abcdefghijk").stat().st_mode & 0o077 == 0
        assert (cache.video_dir("abcdefghijk") / "index.faiss").stat().st_mode & 0o077 == 0

    loaded = cache.load_index("abcdefghijk", digest, "test-embeddings", lexical_embeddings)
    assert loaded is not None
    hit, _ = loaded.similarity_search_with_score("gradient descent", k=1)[0]
    assert hit.metadata["video_id"] == "abcdefghijk"
    assert cache.load_index("lmnopqrstuv", digest, "test-embeddings", lexical_embeddings) is None
    assert list(cache.video_dir("abcdefghijk").glob("*.pkl")) == []

    assert cache.load_index("abcdefghijk", "wrong-transcript", "test-embeddings", lexical_embeddings) is None
    assert cache.load_index(
        "abcdefghijk", digest, "test-embeddings", lexical_embeddings,
        chunk_size_tokens=256, chunk_overlap_tokens=32,
    ) is None
    (cache.video_dir("abcdefghijk") / "documents.json").write_text("{}", encoding="utf-8")
    with patch("app.rag.cache.faiss.read_index") as read_index:
        assert cache.load_index("abcdefghijk", digest, "test-embeddings", lexical_embeddings) is None
        read_index.assert_not_called()


def test_transcript_cache_corruption_and_invalidation(tmp_path):
    cache = VideoCache(tmp_path, ttl_seconds=900)
    cache.write_transcript("abcdefghijk", {"video_id": "abcdefghijk", "segments": []})
    assert cache.read_transcript("abcdefghijk")["video_id"] == "abcdefghijk"
    (cache.video_dir("abcdefghijk") / "transcript.json").write_text("not json", encoding="utf-8")
    assert cache.read_transcript("abcdefghijk") is None
    cache.clear_video("abcdefghijk")
    assert not cache.video_dir("abcdefghijk").exists()
    try:
        cache.video_dir("../../escape___")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe cache video ID was accepted")
