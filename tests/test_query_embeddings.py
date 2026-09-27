from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.embeddings import Embedder, get_embedder
from app.main import app


class FakeQueryEmbedder(Embedder):
    model_name = "fake-embedder"
    dimensions = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_embedder] = FakeQueryEmbedder
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_query_returns_an_embedding(client: TestClient):
    response = client.post("/api/v1/embeddings/query", json={"text": "manual tecnico"})

    assert response.status_code == 200
    assert response.json() == {
        "model": {"name": "fake-embedder", "dimensions": 3, "normalized": True},
        "embedding": [14.0, 1.0, 0.0],
    }


def test_blank_query_is_rejected(client: TestClient):
    response = client.post("/api/v1/embeddings/query", json={"text": "   "})

    assert response.status_code == 400
