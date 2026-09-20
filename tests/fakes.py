"""Doubles shared by the test suite."""

from app.embeddings import Embedder


class FakeEmbedder(Embedder):
    """Stands in for the real model: the test suite must not download or run it.

    Vectors are derived from the text so that each one can be told apart, which is what
    the route contracts are about: one vector per chunk in the same order, one vector
    for the query. Both methods answer the same ``dimensions``, like the real embedder
    does, so a test can compare what the two endpoints return.
    """

    model_name = "fake-embedder"
    dimensions = 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(index), 1.0] for index, text in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 1.0]
