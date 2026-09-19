"""When the embedding model is loaded, and how many times.

This is what the ticket is about: loading costs seconds and hundreds of MB, so it must
happen once, during startup, and never again. The real model is replaced by a counting
double -- what is under test is the lifecycle, not the vectors.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app import embeddings as embeddings_module
from app.embeddings import Embedder, get_embedder
from app.main import app
from tests.fakes import FakeEmbedder


class CountingEmbedder(FakeEmbedder):
    """Counts how many times the application built an embedder."""

    loads = 0

    def __init__(self, model_name: str = "counting-embedder") -> None:
        type(self).loads += 1
        self.model_name = model_name


@pytest.fixture
def counting(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[CountingEmbedder]]:
    """Put the counting double where ``get_embedder`` looks for the real model."""
    CountingEmbedder.loads = 0
    monkeypatch.setattr(embeddings_module, "SentenceTransformerEmbedder", CountingEmbedder)
    get_embedder.cache_clear()
    yield CountingEmbedder
    get_embedder.cache_clear()
    app.dependency_overrides.clear()


def test_get_embedder_builds_the_model_only_once(counting: type[CountingEmbedder]):
    first = get_embedder()
    second = get_embedder()

    assert first is second
    assert counting.loads == 1


def test_startup_loads_the_model_before_any_request(counting: type[CountingEmbedder]):
    """The whole point: the first request must not be the one paying for the load."""
    with TestClient(app) as client:
        assert counting.loads == 1

        assert client.post("/api/v1/embeddings/query", json={"text": "primeira busca"}).status_code == 200
        assert client.post("/api/v1/embeddings/query", json={"text": "segunda busca"}).status_code == 200

    # Both requests reused the instance loaded at startup.
    assert counting.loads == 1


def test_startup_skips_the_load_when_the_embedder_is_overridden(counting: type[CountingEmbedder]):
    """A test double replaces the model, so warming the real one up would be waste."""

    def fake_embedder() -> Embedder:
        return FakeEmbedder()

    app.dependency_overrides[get_embedder] = fake_embedder

    with TestClient(app) as client:
        assert client.post("/api/v1/embeddings/query", json={"text": "busca"}).status_code == 200

    assert counting.loads == 0
