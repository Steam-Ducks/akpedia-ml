from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api import documents as documents_api
from app.embeddings import get_embedder
from app.main import app
from tests.fakes import FakeEmbedder
from tests.pdf_builder import build_pdf

URL = "/api/v1/documents/process"


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_embedder] = FakeEmbedder
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client: TestClient, content: bytes, filename: str, media_type: str = "application/pdf"):
    return client.post(URL, files={"file": (filename, content, media_type)})


def upload_unnamed(client: TestClient, content: bytes, media_type: str | None = "application/pdf"):
    """Post a file part whose ``filename`` is empty, as a client sending a stream would.

    httpx cannot express this through ``files=``: an empty filename makes it drop the
    parameter altogether, and the part then arrives as a plain string field instead of
    an upload. So the body is built by hand.
    """
    boundary = "----akpedia-test-boundary"
    headers = ['Content-Disposition: form-data; name="file"; filename=""']
    if media_type is not None:
        headers.append(f"Content-Type: {media_type}")
    part = f"--{boundary}\r\n" + "\r\n".join(headers) + "\r\n\r\n"
    body = part.encode() + content + f"\r\n--{boundary}--\r\n".encode()

    return client.post(
        URL,
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )


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


def test_file_without_filename_falls_back_to_the_content_type(client: TestClient):
    response = upload_unnamed(client, build_pdf(["Documento enviado sem nome de arquivo."]))

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == ""
    assert body["chunk_count"] >= 1


def test_file_without_filename_and_with_an_unsupported_content_type_returns_415(client: TestClient):
    response = upload_unnamed(client, b"qualquer conteudo", "text/csv")

    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_format"


def test_file_with_neither_filename_nor_content_type_returns_415(client: TestClient):
    response = upload_unnamed(client, b"qualquer conteudo", media_type=None)

    assert response.status_code == 415
    body = response.json()
    assert body["code"] == "unsupported_format"
    assert body["supported_extensions"] == [".pdf"]


def test_upload_past_the_size_limit_returns_413(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(documents_api, "MAX_UPLOAD_BYTES", 128)

    response = upload(client, build_pdf(["Documento grande demais."]), "grande.pdf")

    assert response.status_code == 413
    body = response.json()
    assert body["code"] == "file_too_large"
    assert "grande.pdf" in body["message"]
    # The limit says nothing about formats, so the format field stays out of the body.
    assert "supported_extensions" not in body


def test_upload_within_the_size_limit_is_processed(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    content = build_pdf(["Documento dentro do limite."])
    monkeypatch.setattr(documents_api, "MAX_UPLOAD_BYTES", len(content))

    assert upload(client, content, "justo.pdf", "application/pdf").status_code == 200


def test_errors_other_than_415_omit_supported_extensions(client: TestClient):
    response = upload(client, b"isto nao e um pdf", "quebrado.pdf")

    assert response.status_code == 422
    assert "supported_extensions" not in response.json()


def test_request_without_file_is_rejected(client: TestClient):
    assert client.post(URL).status_code == 422
