"""Geração de embeddings dos chunks.

O resto da aplicação depende apenas do contrato :class:`Embedder`: entra uma lista de
textos, sai um vetor por texto. Trocar de modelo (ou usar um dublê nos testes) é
escrever outra implementação, sem mexer em quem chama.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import EMBEDDING_MODEL

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

#: e5 espera esse prefixo nos textos indexados; buscas usam "query: ".
#: Trocar EMBEDDING_MODEL por uma família que não usa prefixo por papel pede revisar isto.
_PASSAGE_PREFIX = "passage: "


class Embedder(ABC):
    """Converte textos em vetores."""

    #: Identificador do modelo, devolvido na resposta da API.
    model_name: str

    #: Tamanho de cada vetor: o akpedia-server precisa dele para a coluna vector(N).
    dimensions: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Devolve um vetor por texto, na mesma ordem."""


class SentenceTransformerEmbedder(Embedder):
    """Implementação real, sobre um modelo sentence-transformers rodando em CPU.

    Qual modelo vem de ``EMBEDDING_MODEL`` (default: e5-small multilíngue). Carregá-lo
    é caro (centenas de MB), então uma instância é criada uma vez e reaproveitada por
    todas as requisições — veja :func:`get_embedder`.
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


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Instância única do modelo, carregada na primeira chamada."""
    return SentenceTransformerEmbedder()
