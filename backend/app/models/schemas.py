from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VIDEO_ID_PATTERN = r"^[A-Za-z0-9_-]{11}$"


class APIModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


class InitializeRequest(APIModel):
    video_id: str = Field(pattern=VIDEO_ID_PATTERN)
    languages: list[str] = Field(default_factory=lambda: ["en"], min_length=1, max_length=8)
    force: bool = False


class ProcessingStatus(APIModel):
    video_id: str
    status: Literal["queued", "processing", "ready", "error"]
    phase: str = ""
    progress: int = Field(default=0, ge=0, le=100)
    message: str = ""
    chunk_count: int | None = None
    language: str | None = None
    error: dict[str, str] | None = None


class SessionRequest(APIModel):
    video_id: str = Field(pattern=VIDEO_ID_PATTERN)


class ChatRequest(SessionRequest):
    session_id: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=4000)


class SearchRequest(SessionRequest):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)


class SummaryRequest(SessionRequest):
    kind: Literal["quick", "detailed", "chapters", "notes"] = "quick"


class Source(BaseModel):
    source_id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    timestamp: str
    excerpt: str
    relevance: float | None = None
    timestamp_url: str


class ChatResponse(BaseModel):
    session_id: str
    video_id: str
    answer: str
    sources: list[Source]
    abstained: bool = False


class SearchResponse(BaseModel):
    video_id: str
    query: str
    results: list[Source]


class SummaryResponse(BaseModel):
    video_id: str
    kind: str
    summary: str
    chunks_processed: int
    language: str


class SessionResponse(BaseModel):
    session_id: str
    video_id: str
    status: str = "active"


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    model: str
    embeddings_loaded: bool
