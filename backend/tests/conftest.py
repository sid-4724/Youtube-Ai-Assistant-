import hashlib
import re

import pytest
from langchain_core.embeddings import Embeddings


class LexicalEmbeddings(Embeddings):
    """Small deterministic test embedding; never used by the application at runtime."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = -1.0 if digest[4] & 1 else 1.0
            vector[index] += sign
        length = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / length for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def lexical_embeddings():
    return LexicalEmbeddings()
