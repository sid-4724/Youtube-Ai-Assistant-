import pytest

from app.rag.chunking import transcript_to_documents
from app.services.transcripts import (
    TranscriptUnavailable,
    normalize_snippets,
)


def test_normalization_cleans_markup_and_nearby_exact_duplicates():
    transcript = normalize_snippets(
        "abcdefghijk",
        [
            {"text": " Hello &amp; <i>world</i>  ", "start": 0, "duration": 1.0},
            {"text": "Hello & world", "start": 0.2, "duration": 1.0},
            {"text": "again", "start": 2, "duration": 0.8},
        ],
    )
    assert [segment.text for segment in transcript.segments] == ["Hello & world", "again"]
    assert transcript.segments[1].end == pytest.approx(2.8)


def test_empty_and_invalid_transcripts_fail_without_fabricating_content():
    with pytest.raises(TranscriptUnavailable):
        normalize_snippets("abcdefghijk", [{"text": "<i></i>", "start": 0, "duration": 1}])
    with pytest.raises(ValueError, match="Invalid YouTube"):
        normalize_snippets("not-a-video-id", [{"text": "hello", "start": 0}])
    with pytest.raises(ValueError, match="finite"):
        normalize_snippets("abcdefghijk", [{"text": "hello", "start": float("inf")}])


def test_chunk_metadata_keeps_original_timestamp_spans():
    words = " ".join(f"concept{i}" for i in range(220))
    transcript = normalize_snippets(
        "abcdefghijk",
        [
            {"text": words[: len(words) // 2], "start": 12.0, "duration": 20.0},
            {"text": words[len(words) // 2 :], "start": 32.0, "duration": 18.0},
        ],
    )
    docs = transcript_to_documents(transcript, transcript.video_id, chunk_size_tokens=64, chunk_overlap_tokens=12)
    assert len(docs) > 1
    assert all(doc.metadata["video_id"] == "abcdefghijk" for doc in docs)
    assert all(doc.metadata["chunk_id"].startswith("abcdefghijk-") for doc in docs)
    assert min(doc.metadata["start_seconds"] for doc in docs) == 12.0
    assert max(doc.metadata["end_seconds"] for doc in docs) == 50.0


def test_transcript_extraction_failure_is_explicit():
    from app.services.transcripts import YouTubeTranscriptLoader

    class BrokenAPI:
        def fetch(self, *_args, **_kwargs):
            raise RuntimeError("captions disabled")

    class BrokenFactory:
        def __call__(self):
            return BrokenAPI()

    with pytest.raises(TranscriptUnavailable, match="captions disabled"):
        YouTubeTranscriptLoader(BrokenFactory()).fetch("abcdefghijk", ["en"])
