from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app
from app.services.runtime import PreparedVideo
from app.services.transcripts import normalize_snippets
from test_runtime import FakeChat, make_runtime


def test_api_validation_errors_health_and_session_chat(tmp_path, lexical_embeddings):
    with TestClient(app) as client:
        runtime = make_runtime(tmp_path, lexical_embeddings)
        transcript = normalize_snippets("abcdefghijk", [{"text": "Gradient descent updates model weights to lower loss.", "start": 50, "duration": 10}])
        documents = [Document(page_content="Gradient descent updates model weights to lower loss.", metadata={"video_id": "abcdefghijk", "chunk_id": "abcdefghijk-00000", "start_seconds": 50.0, "end_seconds": 60.0, "start_timestamp": "00:50", "end_timestamp": "01:00", "language": "en", "transcript_source": "youtube_manual"})]
        store = __import__("app.rag.cache", fromlist=["build_faiss_store"]).build_faiss_store(documents, lexical_embeddings)
        runtime._videos["abcdefghijk"] = PreparedVideo(transcript, documents, store, "fixture")
        runtime._chat = FakeChat()
        app.state.runtime = runtime

        invalid = client.post("/api/videos/initialize", json={"video_id": "bad"})
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "validation_error"

        created = client.post("/api/sessions", json={"video_id": "abcdefghijk"})
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        response = client.post("/api/videos/abcdefghijk/ask", json={
            "video_id": "abcdefghijk", "session_id": session_id, "question": "What does gradient descent update?"
        })
        assert response.status_code == 200
        body = response.json()
        assert body["sources"][0]["timestamp"] == "00:50"
        assert body["sources"][0]["timestamp_url"].endswith("&t=50s")

        mismatch = client.post("/api/videos/abcdefghijk/search", json={"video_id": "lmnopqrstuv", "query": "gradient descent"})
        assert mismatch.status_code == 400
