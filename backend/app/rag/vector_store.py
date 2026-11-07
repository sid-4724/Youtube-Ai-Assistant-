from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class LocalFaissStore(VectorStore):
    """Small LangChain VectorStore adapter over a normalized, exact FAISS IP index."""

    def __init__(
        self,
        embeddings: Embeddings,
        index: Any,
        documents: list[Document],
        index_to_docstore_id: dict[int, str] | None = None,
    ):
        self.embedding_function = embeddings
        self.index = index
        self.documents = documents
        self.index_to_docstore_id = index_to_docstore_id or {
            index: f"doc-{index}" for index in range(len(documents))
        }
        if (
            not self.documents
            or self.index.d <= 0
            or len(self.documents) != self.index.ntotal
            or len(self.index_to_docstore_id) != self.index.ntotal
        ):
            raise ValueError("FAISS vectors, documents, and IDs must have matching lengths")

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "LocalFaissStore":
        if ids is not None and len(ids) != len(texts):
            raise ValueError("ids and texts must have matching lengths")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas and texts must have matching lengths")
        documents = [
            Document(page_content=text, metadata=(metadatas or [{}] * len(texts))[index])
            for index, text in enumerate(texts)
        ]
        store = build_store(documents, embedding)
        if ids is not None:
            return cls(
                embedding,
                store.index,
                documents,
                {index: doc_id for index, doc_id in enumerate(ids)},
            )
        return store

    def _search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if k < 1:
            return []
        vector = np.asarray(self.embedding_function.embed_query(query), dtype="float32").reshape(1, -1)
        if vector.shape[1] != self.index.d or not np.isfinite(vector).all():
            raise ValueError("Query embedding has an invalid dimension or non-finite values")
        faiss.normalize_L2(vector)
        scores, row_ids = self.index.search(vector, min(k, self.index.ntotal))
        results: list[tuple[Document, float]] = []
        for row_id, score in zip(row_ids[0], scores[0]):
            if row_id < 0:
                continue
            results.append((self.documents[int(row_id)], float(score)))
        return results

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> list[tuple[Document, float]]:
        return self._search(query, k)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        results = self._search(query, k if filter is None else self.index.ntotal)
        if filter is not None:
            results = [
                (document, score)
                for document, score in results
                if all(document.metadata.get(key) == value for key, value in filter.items())
            ][:k]
        return [document for document, _ in results]


def build_store(documents: list[Document], embeddings: Embeddings) -> LocalFaissStore:
    if not documents:
        raise ValueError("Cannot index an empty document collection")
    vectors = np.asarray(
        embeddings.embed_documents([document.page_content for document in documents]), dtype="float32"
    )
    if vectors.ndim != 2 or len(vectors) != len(documents) or vectors.shape[1] == 0:
        raise ValueError("Embedding model returned an invalid vector matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned non-finite values")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return LocalFaissStore(embeddings, index, documents)
