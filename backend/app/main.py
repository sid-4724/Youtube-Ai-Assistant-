import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    InitializeRequest,
    ProcessingStatus,
    SearchRequest,
    SearchResponse,
    SessionRequest,
    SessionResponse,
    SummaryRequest,
    SummaryResponse,
    VIDEO_ID_PATTERN,
)
from app.services.runtime import AssistantError, AssistantRuntime


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    app.state.runtime = AssistantRuntime(settings)
    yield


app = FastAPI(
    title="YouTube AI Assistant API",
    description="Local-first transcript retrieval, semantic search, grounded chat, and summaries.",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=r"^chrome-extension://[a-p]{32}$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(AssistantError)
async def assistant_error_handler(_: Request, exc: AssistantError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": str(exc)}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"field": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "validation_error", "message": "Request validation failed.", "details": errors}},
    )


def runtime(request: Request) -> AssistantRuntime:
    return request.app.state.runtime


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health(request: Request) -> dict[str, Any]:
    return runtime(request).health()


@app.post("/api/videos/initialize", response_model=ProcessingStatus, tags=["videos"])
async def initialize(payload: InitializeRequest, request: Request) -> dict[str, Any]:
    return await runtime(request).start_processing(payload.video_id, payload.languages, payload.force)


@app.get("/api/videos/{video_id}/status", response_model=ProcessingStatus, tags=["videos"])
async def video_status(
    request: Request,
    video_id: str = Path(pattern=VIDEO_ID_PATTERN),
) -> dict[str, Any]:
    return runtime(request).status(video_id)


@app.post("/api/videos/{video_id}/ask", response_model=ChatResponse, tags=["chat"])
async def ask(
    request: Request,
    payload: ChatRequest,
    video_id: str = Path(pattern=VIDEO_ID_PATTERN),
) -> dict[str, Any]:
    if payload.video_id != video_id:
        raise AssistantError("video_mismatch", "Path video ID and request video ID must match.")
    answer, sources, abstained = await asyncio.to_thread(
        runtime(request).answer, video_id, payload.session_id, payload.question
    )
    return {
        "session_id": payload.session_id,
        "video_id": video_id,
        "answer": answer,
        "sources": [source.model_dump() for source in sources],
        "abstained": abstained,
    }


@app.post("/api/videos/{video_id}/search", response_model=SearchResponse, tags=["search"])
async def search(
    request: Request,
    payload: SearchRequest,
    video_id: str = Path(pattern=VIDEO_ID_PATTERN),
) -> dict[str, Any]:
    if payload.video_id != video_id:
        raise AssistantError("video_mismatch", "Path video ID and request video ID must match.")
    results = await asyncio.to_thread(
        runtime(request).search, video_id, payload.query, payload.limit
    )
    return {"video_id": video_id, "query": payload.query, "results": [item.model_dump() for item in results]}


@app.get("/api/videos/{video_id}/timestamps", response_model=SearchResponse, tags=["search"])
async def timestamps(
    request: Request,
    video_id: str = Path(pattern=VIDEO_ID_PATTERN),
    query: str = Query(min_length=1, max_length=1000),
    limit: int = Query(default=5, ge=1, le=20),
) -> dict[str, Any]:
    results = await asyncio.to_thread(runtime(request).search, video_id, query, limit)
    return {"video_id": video_id, "query": query, "results": [item.model_dump() for item in results]}


@app.post("/api/videos/{video_id}/summarize", response_model=SummaryResponse, tags=["summary"])
async def summarize(
    request: Request,
    payload: SummaryRequest,
    video_id: str = Path(pattern=VIDEO_ID_PATTERN),
) -> dict[str, Any]:
    if payload.video_id != video_id:
        raise AssistantError("video_mismatch", "Path video ID and request video ID must match.")
    summary, chunks_processed, language = await asyncio.to_thread(
        runtime(request).summarize, video_id, payload.kind
    )
    return {
        "video_id": video_id,
        "kind": payload.kind,
        "summary": summary,
        "chunks_processed": chunks_processed,
        "language": language,
    }


@app.post("/api/sessions", response_model=SessionResponse, tags=["sessions"])
async def create_session(payload: SessionRequest, request: Request) -> dict[str, str]:
    session_id = runtime(request).create_session(payload.video_id)
    return {"session_id": session_id, "video_id": payload.video_id, "status": "active"}


@app.delete("/api/sessions/{session_id}", status_code=204, tags=["sessions"])
async def clear_session(
    request: Request,
    session_id: str,
    video_id: str = Query(pattern=VIDEO_ID_PATTERN),
) -> None:
    runtime(request).clear_session(session_id, video_id)
