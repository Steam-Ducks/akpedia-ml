from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.embeddings import Embedder, get_embedder
from app.main import app
from tests.pdf_builder import build_pdf

URL = "/api/v1/documents/process"


class FakeEmbedder(Embedder):
    """Stands in for the real model: the test suite must not download or run it.

    Vectors are derived from the text so that each chunk can be told apart, which is
    what the route contract is about: one vector per chunk, in the same order.
    """

    model_name = "fake-embedder"
    dimensions = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(index), 1.0] for index, text in enumerate(texts)]


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_embedder] = FakeEmbedder
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client: TestClient, content: bytes, filename: str, media_type: str = "application/pdf"):
    return client.post(URL, files={"file": (filename, content, media_type)})


def test_pdf_returns_one_vector_per_chunk(client: TestClient):
    response = upload(client, build_pdf(["Primeira frase do manual. Segunda frase do manual."]), "manual.pdf")

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "manual.pdf"
    assert body["model"] == {"name": "fake-embedder", "dimensions": 3, "normalized": True}
    assert body["chunk_count"] == len(body["chunks"]) >= 1
    assert [chunk["index"] for chunk in body["chunks"]] == list(range(body["chunk_count"]))
    for chunk in body["chunks"]:
        assert chunk["text"]
        assert len(chunk["embedding"]) == 3
        assert chunk["embedding"][0] == float(len(chunk["text"]))


def test_long_document_is_split_into_several_chunks(client: TestClient):
    sentence = "Esta é uma frase de tamanho razoável usada para encher o documento. "
    response = upload(client, build_pdf([sentence * 40]), "longo.pdf")

    body = response.json()
    assert response.status_code == 200
    assert body["chunk_count"] > 1
    assert all(len(chunk["text"]) <= 1000 for chunk in body["chunks"])


def test_unsupported_format_returns_415_listing_the_supported_ones(client: TestClient):
    response = upload(
        client,
        b"qualquer conteudo",
        "planilha.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert response.status_code == 415
    body = response.json()
    assert body["code"] == "unsupported_format"
    assert "planilha.xlsx" in body["message"]
    assert body["supported_extensions"] == [".pdf"]


def test_corrupt_pdf_returns_422(client: TestClient):
    response = upload(client, b"isto nao e um pdf", "quebrado.pdf")

    assert response.status_code == 422
    assert response.json()["code"] == "unreadable_document"


def test_pdf_without_text_returns_422(client: TestClient):
    response = upload(client, build_pdf([""]), "digitalizado.pdf")

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "no_extractable_text"
    assert "OCR" in body["message"]


def test_request_without_file_is_rejected(client: TestClient):
    assert client.post(URL).status_code == 422
