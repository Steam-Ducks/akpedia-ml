from io import BytesIO

import pytest

from app.documents import DocumentReadError, PdfTextExtractor, extract_text
from tests.pdf_builder import build_pdf


def test_returns_the_text_of_a_single_page_pdf():
    pdf = build_pdf(["Akaer quality procedure"])

    assert extract_text(pdf, filename="procedure.pdf") == "Akaer quality procedure"


def test_joins_every_page_in_document_order():
    pdf = build_pdf(["First page", "Second page", "Third page"])

    assert extract_text(pdf, filename="report.pdf") == "First page\n\nSecond page\n\nThird page"


def test_preserves_accented_portuguese_text():
    pdf = build_pdf(["Revisao da secao tecnica: manutencao, inspecao e certificacao"])

    text = extract_text(pdf, filename="norma.pdf")

    assert text == "Revisao da secao tecnica: manutencao, inspecao e certificacao"


def test_drops_pages_without_text():
    pdf = build_pdf(["Page with content", "   ", "Last page"])

    assert extract_text(pdf, filename="scan.pdf") == "Page with content\n\nLast page"


def test_reads_from_an_open_binary_stream():
    pdf = build_pdf(["Streamed content"])

    with BytesIO(pdf) as stream:
        assert extract_text(stream, filename="upload.pdf") == "Streamed content"


def test_matches_the_extension_regardless_of_case():
    pdf = build_pdf(["Uppercase extension"])

    assert extract_text(pdf, filename="REPORT.PDF") == "Uppercase extension"


def test_media_type_identifies_the_format_when_the_filename_is_missing():
    pdf = build_pdf(["Identified by media type"])

    assert extract_text(pdf, media_type="application/pdf; charset=binary") == "Identified by media type"


def test_raises_document_read_error_when_the_file_is_not_a_pdf():
    with pytest.raises(DocumentReadError):
        extract_text(b"not a pdf at all", filename="broken.pdf")


def test_raises_document_read_error_when_the_pdf_is_truncated():
    pdf = build_pdf(["Truncated document"])

    with pytest.raises(DocumentReadError):
        extract_text(pdf[: len(pdf) // 2], filename="truncated.pdf")


def test_extractor_declares_the_formats_it_handles():
    extractor = PdfTextExtractor()

    assert extractor.extensions == (".pdf",)
    assert extractor.media_types == ("application/pdf",)
