import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import faiss
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.vector_store import LocalFaissStore, build_store

CACHE_FORMAT_VERSION = 1


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class VideoCache:
    """Per-video cache using JSON metadata plus raw FAISS index files; no pickle loading."""

    def __init__(self, root: Path, ttl_seconds: int):
        self.root = root
        self.ttl_seconds = ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._restrict(self.root, 0o700)

    @staticmethod
    def _restrict(path: Path, mode: int) -> None:
        try:
            path.chmod(mode)
        except OSError:
            # Some platforms do not expose POSIX permission bits.
            pass

    def video_dir(self, video_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError("Invalid YouTube video ID for cache path")
        path = self.root / video_id
        if path.is_symlink():
            raise ValueError("Cache video directory must not be a symlink")
        return path

    def _fresh(self, path: Path) -> bool:
        return path.exists() and (
            self.ttl_seconds == 0 or time.time() - path.stat().st_mtime <= self.ttl_seconds
        )

    def read_transcript(self, video_id: str) -> dict[str, Any] | None:
        path = self.video_dir(video_id) / "transcript.json"
        if not self._fresh(path):
            return None
        try:
            value = json.loads(path.read_text("utf-8"))
            return value if value.get("video_id") == video_id else None
        except (OSError, ValueError, AttributeError):
            return None

    def write_transcript(self, video_id: str, value: dict[str, Any]) -> None:
        atomic_write(self.video_dir(video_id) / "transcript.json", json.dumps(value).encode())

    def has_manifest(self, video_id: str) -> bool:
        return self._fresh(self.video_dir(video_id) / "manifest.json")

    def clear_video(self, video_id: str) -> None:
        directory = self.video_dir(video_id)
        if directory.parent.resolve() != self.root.resolve():
            raise ValueError("Unsafe cache path")
        shutil.rmtree(directory, ignore_errors=True)

    def save_index(
        self,
        video_id: str,
        transcript_hash: str,
        embedding_model: str,
        documents: list[Document],
        store: LocalFaissStore,
        chunk_size_tokens: int = 320,
        chunk_overlap_tokens: int = 64,
    ) -> None:
        if not documents or len(documents) != store.index.ntotal:
            raise ValueError("Cannot cache mismatched or empty FAISS documents")
        if any(document.metadata.get("video_id") != video_id for document in documents):
            raise ValueError("Cannot cache documents belonging to another video")
        directory = self.video_dir(video_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._restrict(directory, 0o700)
        serialized_docs = []
        index_to_id: dict[str, str] = {}
        for index, document in enumerate(documents):
            doc_id = f"{video_id}:{index:06d}"
            serialized_docs.append(
                {"id": doc_id, "page_content": document.page_content, "metadata": document.metadata}
            )
            index_to_id[str(index)] = doc_id

        index_path = directory / "index.faiss"
        faiss.write_index(store.index, str(index_path))
        self._restrict(index_path, 0o600)
        docs_bytes = json.dumps(serialized_docs, ensure_ascii=False, separators=(",", ":")).encode()
        ids_bytes = json.dumps(index_to_id, separators=(",", ":")).encode()
        atomic_write(directory / "documents.json", docs_bytes)
        atomic_write(directory / "index_ids.json", ids_bytes)
        manifest = {
            "format_version": CACHE_FORMAT_VERSION,
            "video_id": video_id,
            "created_at": time.time(),
            "transcript_hash": transcript_hash,
            "embedding_model": embedding_model,
            "chunk_size_tokens": chunk_size_tokens,
            "chunk_overlap_tokens": chunk_overlap_tokens,
            "dimension": int(store.index.d),
            "document_count": len(documents),
            "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "documents_sha256": hashlib.sha256(docs_bytes).hexdigest(),
            "ids_sha256": hashlib.sha256(ids_bytes).hexdigest(),
        }
        atomic_write(directory / "manifest.json", json.dumps(manifest).encode())

    def load_index(
        self,
        video_id: str,
        transcript_hash: str,
        embedding_model: str,
        embeddings: Embeddings,
        chunk_size_tokens: int = 320,
        chunk_overlap_tokens: int = 64,
    ) -> LocalFaissStore | None:
        directory = self.video_dir(video_id)
        manifest_path = directory / "manifest.json"
        if not self._fresh(manifest_path):
            return None
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
            if (
                manifest.get("format_version") != CACHE_FORMAT_VERSION
                or manifest.get("video_id") != video_id
                or manifest.get("transcript_hash") != transcript_hash
                or manifest.get("embedding_model") != embedding_model
                or manifest.get("chunk_size_tokens") != chunk_size_tokens
                or manifest.get("chunk_overlap_tokens") != chunk_overlap_tokens
            ):
                return None
            index_path = directory / "index.faiss"
            documents_path = directory / "documents.json"
            ids_path = directory / "index_ids.json"
            index_bytes, docs_bytes, ids_bytes = (
                index_path.read_bytes(), documents_path.read_bytes(), ids_path.read_bytes()
            )
            checks = (
                (hashlib.sha256(index_bytes).hexdigest(), manifest["index_sha256"]),
                (hashlib.sha256(docs_bytes).hexdigest(), manifest["documents_sha256"]),
                (hashlib.sha256(ids_bytes).hexdigest(), manifest["ids_sha256"]),
            )
            if any(actual != expected for actual, expected in checks):
                return None
            records = json.loads(docs_bytes)
            ids = json.loads(ids_bytes)
            if not isinstance(records, list) or not isinstance(ids, dict):
                return None
            docs = {
                record["id"]: Document(
                    page_content=record["page_content"], metadata=record["metadata"]
                )
                for record in records
            }
            if len(docs) != len(records) or any(
                not record["id"].startswith(f"{video_id}:")
                or record["metadata"].get("video_id") != video_id
                for record in records
            ):
                return None
            index = faiss.read_index(str(index_path))
            if (
                index.d != manifest["dimension"]
                or index.ntotal != len(docs)
                or index.ntotal != len(ids)
                or manifest.get("document_count") != len(docs)
                or sorted(int(key) for key in ids) != list(range(index.ntotal))
                or any(value not in docs for value in ids.values())
            ):
                return None
            ordered_ids = {int(key): value for key, value in ids.items()}
            ordered_documents = [docs[ordered_ids[index]] for index in range(index.ntotal)]
            return LocalFaissStore(embeddings, index, ordered_documents, ordered_ids)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            AttributeError,
            IndexError,
            OverflowError,
            json.JSONDecodeError,
        ):
            return None


def build_faiss_store(documents: list[Document], embeddings: Embeddings) -> LocalFaissStore:
    """Construct a normalized cosine-similarity LangChain store backed by FAISS."""
    return build_store(documents, embeddings)
