from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YTA_", env_file=".env", extra="ignore", enable_decoding=False
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "mistral:7b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: Path = Path(".cache/youtube-ai-assistant")
    cache_ttl_seconds: int = Field(default=7 * 24 * 60 * 60, ge=0)
    chunk_size_tokens: int = Field(default=320, ge=64, le=1000)
    chunk_overlap_tokens: int = Field(default=64, ge=0, le=300)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    min_relevance: float = Field(default=0.18, ge=-1.0, le=1.0)
    transcript_languages: list[str] = ["en"]
    allowed_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    @field_validator("transcript_languages", "allowed_origins", mode="before")
    @classmethod
    def split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def overlap_smaller_than_chunk(cls, value: int, info: object) -> int:
        # The splitter also checks this when instantiated; this gives a clear settings error.
        chunk_size = getattr(info, "data", {}).get("chunk_size_tokens", 320)
        if value >= chunk_size:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
