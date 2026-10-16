from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.analysis import CategoryCentroids, build_centroids, get_centroids
from app.api import documents as documents_api
from app.embeddings import Embedder, get_embedder
from app.main import app
from tests.pdf_builder import build_pdf
from tests.pdf_fixtures import Span, build_styled_pdf

URL = "/api/v1/documents/analyze"

#: One axis per category: a chunk mentioning a keyword points along its category's axis.
KEYWORDS = {"motor": "Tecnico", "anac": "Regulatorio", "contrato": "Juridico", "auditoria": "Qualidade"}
LABELS = list(KEYWORDS.values())


class KeywordEmbedder(Embedder):
    """Stands in for the real model with vectors a test can predict.

    Each dimension counts one keyword, plus a small constant so that text with no
    keyword still has a direction. The suite must not download or run the real model.
    """

    model_name = "keyword-embedder"
    dimensions = len(KEYWORDS)

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[text.lower().count(keyword) + 0.1 for keyword in KEYWORDS] for text in texts]


def keyword_centroids() -> CategoryCentroids:
    return build_centroids(LABELS, list(np.eye(len(LABELS))))


@pytest.fixture
def embedder() -> KeywordEmbedder:
    return KeywordEmbedder()


@pytest.fixture
def client(embedder: KeywordEmbedder) -> Iterator[TestClient]:
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_centroids] = keyword_centroids
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client: TestClient, content: bytes, filename: str, media_type: str = "application/pdf"):
    return client.post(URL, files={"file": (filename, content, media_type)})


@pytest.mark.parametrize(
    ("keyword", "category"),
    list(KEYWORDS.items()),
)
def test_each_category_is_suggested_with_title_description_and_confidence(
    client: TestClient, keyword: str, category: str
):
    pdf = build_styled_pdf(
        [
            [
                [Span("Documento de Referência", 22, bold=True)],
                f"Este documento trata de {keyword} e dos procedimentos associados ao {keyword}.",
                f"Mais detalhes sobre {keyword} seguem nas próximas páginas.",
            ],
            [f"Segunda página sobre {keyword}."],
        ]
    )

    response = upload(client, pdf, "referencia.pdf")

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Documento de Referência"
    assert body["description"] == f"Este documento trata de {keyword} e dos procedimentos associados ao {keyword}."
    assert body["category"] == category
    assert body["confidence"] == body["scores"][category] > 0.5
    assert set(body["scores"]) == set(LABELS)
    assert sum(body["scores"].values()) == pytest.approx(1.0)
    assert body["pages"] == 2


def test_response_carries_no_vectors(client: TestClient):
    body = upload(client, build_pdf(["Manual do motor com texto suficiente para descrever."]), "m.pdf").json()

    assert set(body) == {"title", "description", "category", "confidence", "scores", "pages"}


def test_title_comes_from_the_metadata_first(client: TestClient):
    pdf = build_styled_pdf(
        [[[Span("Capa Grande", 30)], "Texto do corpo do documento sobre o motor."]],
        title="Microsoft Word - Manual do Motor.docx",
    )

    assert upload(client, pdf, "arquivo.pdf").json()["title"] == "Manual do Motor"


def test_title_falls_back_to_the_filename(client: TestClient):
    response = upload(client, build_pdf(["Texto do corpo sem nenhum destaque tipográfico."]), "manual_motor.pdf")

    assert response.json()["title"] == "manual_motor"


def test_without_centroids_the_category_is_null(client: TestClient, embedder: KeywordEmbedder):
    app.dependency_overrides[get_centroids] = lambda: None

    response = upload(client, build_pdf(["Manual do motor com texto suficiente para descrever."]), "manual.pdf")

    assert response.status_code == 200
    body = response.json()
    assert body["category"] is None
    assert body["confidence"] is None
    assert body["scores"] == {}
    assert body["title"] == "manual"
    assert body["pages"] == 1
    # Nothing to compare with, so the document is not embedded at all.
    assert embedder.calls == []


def test_long_documents_embed_at_most_the_configured_chunks(
    client: TestClient, embedder: KeywordEmbedder, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(documents_api, "ANALYSIS_MAX_CHUNKS", 3)
    sentence = "Esta é uma frase sobre o motor usada para encher o documento. "

    assert upload(client, build_pdf([sentence * 40] * 5), "longo.pdf").status_code == 200
    assert len(embedder.calls) == 1 and len(embedder.calls[0]) == 3


def test_pdf_without_text_returns_422_with_the_process_message(client: TestClient):
    response = upload(client, build_pdf([""]), "digitalizado.pdf")

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "no_extractable_text"
    assert body["message"] == (
        "No text could be extracted from 'digitalizado.pdf'. "
        "Scanned documents need OCR, which this service does not support yet."
    )


def test_corrupt_pdf_returns_422(client: TestClient):
    response = upload(client, b"isto nao e um pdf", "quebrado.pdf")

    assert response.status_code == 422
    assert response.json()["code"] == "unreadable_document"


def test_unsupported_format_returns_415(client: TestClient):
    response = upload(client, b"qualquer conteudo", "planilha.xlsx", "application/vnd.ms-excel")

    assert response.status_code == 415
    body = response.json()
    assert body["code"] == "unsupported_format"
    assert body["supported_extensions"] == [".pdf"]


def test_upload_past_the_size_limit_returns_413(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(documents_api, "MAX_UPLOAD_BYTES", 128)

    response = upload(client, build_pdf(["Documento grande demais."]), "grande.pdf")

    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"
