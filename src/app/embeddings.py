"""Generate embeddings for document chunks and search queries.

The rest of the application depends only on the :class:`Embedder` contract: it
receives a list of texts and returns one vector per text. Changing the model (or
using a test double) only requires another implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import EMBEDDING_MODEL

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

#: E5 models expect this prefix for indexed text; searches use ``query: ``.
#: Switching to a model family without role-specific prefixes requires revisiting this.
_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "


class Embedder(ABC):
    """Convert texts into vectors."""

    #: Model identifier returned in the API response.
    model_name: str

    #: Vector size required by akpedia-server's vector(N) column.
    dimensions: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text, in the same order."""

    def embed_query(self, text: str) -> list[float]:
        """Return the vector for one search query."""
        return self.embed([text])[0]


class SentenceTransformerEmbedder(Embedder):
    """Real implementation using a sentence-transformers model on the CPU.

    The model comes from ``EMBEDDING_MODEL`` (default: multilingual e5-small).
    Loading it is expensive, so one instance is created and reused for all requests;
    see :func:`get_embedder`.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model: SentenceTransformer = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self._model.encode(
            [f"{_PASSAGE_PREFIX}{text}" for text in texts],
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vectors = self._model.encode(
            [f"{_QUERY_PREFIX}{text}"],
            normalize_embeddings=True,
        )
        return vectors[0].tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the singleton model, loaded on the first call."""
    return SentenceTransformerEmbedder()
