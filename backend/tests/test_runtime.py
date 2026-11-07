import asyncio

from app.config import Settings
from app.services.runtime import AssistantError, AssistantRuntime, PreparedVideo
from app.services.transcripts import normalize_snippets


class Loader:
    def __init__(self):
        self.calls = 0

    def fetch(self, video_id, languages):
        self.calls += 1
        assert languages == ["en"]
        return normalize_snippets(video_id, [
            {"text": "Gradient descent updates weights in the direction that reduces the loss function.", "start": 20, "duration": 5},
            {"text": "The instructor demonstrates the gradient step on a simple model example.", "start": 30, "duration": 6},
            {"text": "Convolutional networks apply filters to image regions.", "start": 100, "duration": 4},
        ])


class FakeChat:
    def __init__(self, **_kwargs):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        from langchain_core.messages import AIMessage

        prompt = " ".join(str(message.content) for message in messages)
        if "Return only the query" in prompt:
            return AIMessage(content="gradient descent instructor example")
        return AIMessage(content="The instructor shows that gradient descent changes weights to reduce loss [S1].")


def make_runtime(tmp_path, embeddings, loader=None, min_relevance=0.18):
    settings = Settings(
        cache_dir=tmp_path,
        cache_ttl_seconds=3600,
        embedding_model="deterministic-test",
        min_relevance=min_relevance,
        chunk_size_tokens=128,
        chunk_overlap_tokens=20,
    )
    return AssistantRuntime(settings, loader or Loader(), lambda _name: embeddings, FakeChat)


def test_initialize_builds_then_reuses_transcript_and_faiss_cache(tmp_path, lexical_embeddings):
    loader = Loader()
    runtime = make_runtime(tmp_path, lexical_embeddings, loader)
    runtime.cache.write_transcript("abcdefghijk", {"video_id": "abcdefghijk", "segments": []})

    async def start_and_wait():
        await runtime.start_processing("abcdefghijk", ["en"])
        for _ in range(500):
            current = runtime.status("abcdefghijk")
            if current["status"] in {"ready", "error"}:
                return current
            await asyncio.sleep(0.02)
        raise AssertionError("processing did not finish")

    status = asyncio.run(start_and_wait())
    assert status["status"] == "ready"
    assert status["chunk_count"] >= 1
    assert loader.calls == 1

    second_loader = Loader()
    second_runtime = make_runtime(tmp_path, lexical_embeddings, second_loader)
    second_status = asyncio.run(start_and_wait_for(second_runtime))
    assert second_status["status"] == "ready"
    assert second_loader.calls == 0


async def start_and_wait_for(runtime):
    await runtime.start_processing("abcdefghijk", ["en"])
    for _ in range(500):
        current = runtime.status("abcdefghijk")
        if current["status"] in {"ready", "error"}:
            return current
        await asyncio.sleep(0.02)
    raise AssertionError("processing did not finish")


def test_answer_citations_follow_real_retrieved_timestamp_and_sessions_are_video_scoped(tmp_path, lexical_embeddings):
    runtime = make_runtime(tmp_path, lexical_embeddings)
    asyncio.run(start_and_wait_for(runtime))
    session = runtime.create_session("abcdefghijk")
    answer, sources, abstained = runtime.answer("abcdefghijk", session, "How does gradient descent change weights?")
    assert not abstained
    assert "[S1]" in answer
    assert sources[0].source_id == "S1"
    assert sources[0].start_seconds == 20
    assert sources[0].timestamp_url.endswith("&t=20s")
    chat = runtime.chat
    follow_up, follow_up_sources, follow_up_abstained = runtime.answer(
        "abcdefghijk", session, "Can you explain that example?"
    )
    assert not follow_up_abstained
    assert follow_up_sources
    assert "[S1]" in follow_up
    assert len(chat.calls) == 3  # answer, history-aware query rewrite, answer
    assert len(runtime._sessions[session]["messages"]) == 4
    try:
        runtime.answer("lmnopqrstuv", session, "Tell me more about it")
    except AssistantError as exc:
        assert exc.code == "video_not_ready"
    else:
        raise AssertionError("session leaked across video IDs")


def test_low_relevance_question_abstains_without_calling_model(tmp_path, lexical_embeddings):
    runtime = make_runtime(tmp_path, lexical_embeddings, min_relevance=0.25)
    asyncio.run(start_and_wait_for(runtime))
    session = runtime.create_session("abcdefghijk")
    question = "What is the boiling point of mercury?"
    assert runtime.search("abcdefghijk", question, limit=1)[0].relevance < 0.25
    answer, sources, abstained = runtime.answer("abcdefghijk", session, question)
    assert abstained
    assert sources == []
    assert "couldn't find enough" in answer


def test_two_hour_transcript_chunking_retains_late_timestamps():
    segments = [
        {"text": f"At minute {minute}, the lesson explains concept{minute} using a worked example.", "start": minute * 60, "duration": 20}
        for minute in range(0, 125, 2)
    ]
    transcript = normalize_snippets("abcdefghijk", segments)
    from app.rag.chunking import transcript_to_documents

    documents = transcript_to_documents(transcript, transcript.video_id, 100, 16)
    assert documents
    assert max(document.metadata["end_seconds"] for document in documents) >= 2 * 60 * 60


def test_long_summary_uses_bounded_batches_and_returns_processed_chunk_count(tmp_path, lexical_embeddings):
    from langchain_core.documents import Document

    from app.rag.cache import build_faiss_store

    runtime = make_runtime(tmp_path, lexical_embeddings)
    transcript = normalize_snippets("abcdefghijk", [{"text": "Long lesson fixture.", "start": 0, "duration": 1}])
    documents = [
        Document(
            page_content=f"Passage {index} starts with an important definition. " + ("grounded detail " * 500),
            metadata={
                "video_id": "abcdefghijk",
                "chunk_id": f"abcdefghijk-{index:05d}",
                "start_seconds": index * 600,
                "end_seconds": index * 600 + 500,
                "start_timestamp": f"{index * 10:02d}:00",
                "end_timestamp": f"{index * 10 + 8:02d}:20",
            },
        )
        for index in range(9)
    ]
    store = build_faiss_store(documents, lexical_embeddings)
    runtime._videos["abcdefghijk"] = PreparedVideo(transcript, documents, store, "fixture")
    runtime._chat = FakeChat()

    summary, chunks_processed, language = runtime.summarize("abcdefghijk", "chapters")
    assert summary
    assert chunks_processed == 9
    assert language == "en"
    # Nine map calls, two reduce calls, and one final formatting call.
    assert len(runtime.chat.calls) == 12


def test_ollama_failure_returns_structured_service_error(tmp_path, lexical_embeddings):
    runtime = make_runtime(tmp_path, lexical_embeddings)
    asyncio.run(start_and_wait_for(runtime))
    session = runtime.create_session("abcdefghijk")

    class BrokenChat:
        def invoke(self, _messages):
            raise ConnectionError("local model is down")

    runtime._chat = BrokenChat()
    try:
        runtime.answer("abcdefghijk", session, "How does gradient descent change weights?")
    except AssistantError as exc:
        assert exc.code == "llm_unavailable"
        assert exc.status_code == 503
    else:
        raise AssertionError("unavailable Ollama model did not fail explicitly")
