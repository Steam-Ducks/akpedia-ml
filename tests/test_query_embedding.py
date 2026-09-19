"""The search endpoint: POST /api/v1/embeddings/query.

What matters here is the contract akpedia-server relies on -- a vector of the model's
size, described by the same ``model`` block the document endpoint answers. The real
model is not involved: see ``test_embeddings_model.py`` for that.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import embeddings as embeddings_api
from app.embeddings import get_embedder
from app.main import app
from tests.fakes import FakeEmbedder
from tests.pdf_builder import build_pdf

URL = "/api/v1/embeddings/query"
DOCUMENTS_URL = "/api/v1/documents/process"


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_embedder] = FakeEmbedder
    yield TestClient(app)
    app.dependency_overrides.clear()


def query(client: TestClient, text: str):
    return client.post(URL, json={"text": text})


def test_query_returns_a_vector_and_the_model_that_produced_it(client: TestClient):
    response = query(client, "qual é o prazo de garantia?")

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == {"name": "fake-embedder", "dimensions": 3, "normalized": True}
    assert len(body["embedding"]) == 3


def test_query_output_matches_the_document_output(client: TestClient):
    """The DoD of the ticket: both endpoints must be comparable on the server side."""
    document = client.post(
        DOCUMENTS_URL,
        files={"file": ("manual.pdf", build_pdf(["Primeira frase do manual."]), "application/pdf")},
    ).json()
    search = query(client, "primeira frase do manual").json()

    assert search["model"] == document["model"]
    assert len(search["embedding"]) == len(document["chunks"][0]["embedding"]) == search["model"]["dimensions"]


def test_whitespace_is_collapsed_before_embedding(client: TestClient):
    """A query and a chunk differing only in line breaks must not give different vectors."""
    spaced = query(client, "  prazo   de\n\tgarantia  ").json()
    plain = query(client, "prazo de garantia").json()

    assert spaced["embedding"] == plain["embedding"]


def test_empty_query_returns_422(client: TestClient):
    response = query(client, "")

    assert response.status_code == 422
    assert response.json()["code"] == "empty_query"


def test_whitespace_only_query_returns_422(client: TestClient):
    response = query(client, "   \n\t  ")

    assert response.status_code == 422
    assert response.json()["code"] == "empty_query"


def test_query_past_the_length_limit_returns_422(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(embeddings_api, "MAX_QUERY_CHARS", 10)

    response = query(client, "uma busca bem mais longa do que o limite")

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "query_too_long"
    # The length says nothing about formats, so the format field stays out of the body.
    assert "supported_extensions" not in body


def test_query_exactly_at_the_length_limit_is_accepted(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    text = "dez chars!"
    monkeypatch.setattr(embeddings_api, "MAX_QUERY_CHARS", len(text))

    assert query(client, text).status_code == 200


def test_request_without_text_is_rejected(client: TestClient):
    assert client.post(URL, json={}).status_code == 422
