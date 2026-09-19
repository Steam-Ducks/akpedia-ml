"""Embedding generation.

The rest of the application depends only on the :class:`Embedder` contract: texts go
in, one vector per text comes out. Swapping the model (or standing in a double for the
tests) means writing another implementation, without touching the callers.

Role prefixes
-------------
e5 models are trained with a role prefix on every input: ``passage:`` for the text
being indexed and ``query:`` for the text being searched for. Using the wrong one is a
silent quality loss -- the vector is still valid, just farther from where it belongs --
so the prefix is applied inside the embedder and the role is picked by picking a
method. Moving ``EMBEDDING_MODEL`` to a family that does not use role prefixes means
revisiting this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import EMBEDDING_MODEL

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

#: Prefix e5 expects on indexed text.
_PASSAGE_PREFIX = "passage: "

#: Prefix e5 expects on search text.
_QUERY_PREFIX = "query: "


class Embedder(ABC):
    """Turns texts into vectors."""

    #: Model identifier, returned in the API response.
    model_name: str

    #: Size of every vector: akpedia-server needs it for the vector(N) column.
    dimensions: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per indexed text, in the same order."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return the vector of a search text.

        Comparable to what :meth:`embed_documents` produces: same model, same
        dimensions, same normalization. That is what makes the similarity search on
        akpedia-server's side meaningful.
        """


class SentenceTransformerEmbedder(Embedder):
    """Real implementation, over a sentence-transformers model running on CPU.

    Which model comes from ``EMBEDDING_MODEL`` (default: multilingual e5-small).
    Loading it is expensive (hundreds of MB), so a single instance is created once and
    reused by every request -- see :func:`get_embedder`.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model: SentenceTransformer = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(_PASSAGE_PREFIX, texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode(_QUERY_PREFIX, [text])[0]

    def _encode(self, prefix: str, texts: list[str]) -> list[list[float]]:
        """Prefix every text with its role and return the normalized vectors."""
        if not texts:
            return []

        vectors = self._model.encode(
            [f"{prefix}{text}" for text in texts],
            normalize_embeddings=True,
        )
        return vectors.tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Single instance of the model, built on the first call.

    The application warms it up during startup (see ``app.main``), so by the time the
    first request arrives the model is already in memory.
    """
    return SentenceTransformerEmbedder()
