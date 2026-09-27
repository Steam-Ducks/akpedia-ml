"""Checks against the real embedding model.

Marked ``slow`` and skipped by default: the first run downloads a few hundred MB from
Hugging Face. The rest of the suite uses a fake embedder, so nothing here is on the
normal path — this is what confirms that the contract the API publishes (dimensions,
normalized vectors) is the one the model actually delivers.

Run with ``uv run pytest -m slow``.
"""

from math import sqrt

import pytest

from app.config import DEFAULT_EMBEDDING_MODEL
from app.embeddings import SentenceTransformerEmbedder

pytestmark = pytest.mark.slow

EXPECTED_DIMENSIONS = 384


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(DEFAULT_EMBEDDING_MODEL)


def test_model_reports_the_dimensions_the_api_publishes(embedder: SentenceTransformerEmbedder):
    assert embedder.model_name == DEFAULT_EMBEDDING_MODEL
    assert embedder.dimensions == EXPECTED_DIMENSIONS


def test_vectors_are_one_per_text_and_normalized(embedder: SentenceTransformerEmbedder):
    vectors = embedder.embed(["Primeiro trecho do manual.", "Segundo trecho, sobre outro assunto."])

    assert len(vectors) == 2
    for vector in vectors:
        assert len(vector) == EXPECTED_DIMENSIONS
        assert sqrt(sum(value * value for value in vector)) == pytest.approx(1.0, abs=1e-5)


def test_embedding_no_texts_does_not_touch_the_model(embedder: SentenceTransformerEmbedder):
    assert embedder.embed([]) == []


def test_similar_texts_are_closer_than_unrelated_ones(embedder: SentenceTransformerEmbedder):
    related, rephrased, unrelated = embedder.embed(
        [
            "O prazo de garantia do equipamento é de doze meses.",
            "A garantia do aparelho vale por um ano.",
            "A receita leva três ovos e farinha de trigo.",
        ]
    )

    # Vectors are normalized, so the dot product is the cosine similarity.
    def similarity(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert similarity(related, rephrased) > similarity(related, unrelated)
