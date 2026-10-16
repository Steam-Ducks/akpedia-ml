import pytest

from app.documents import DocumentReadError, inspect_pdf
from tests.pdf_builder import build_pdf
from tests.pdf_fixtures import Span, build_styled_pdf

BODY = "Este manual descreve os procedimentos de inspeção do trem de pouso."


def test_metadata_title_and_page_count_are_read():
    inspection = inspect_pdf(build_styled_pdf([[BODY], [BODY], [BODY]], title="Manual do Trem de Pouso"))

    assert inspection.metadata_title == "Manual do Trem de Pouso"
    assert inspection.page_count == 3


def test_missing_metadata_title_is_none():
    assert inspect_pdf(build_styled_pdf([[BODY]])).metadata_title is None


def test_largest_font_line_is_joined_across_spans():
    pdf = build_styled_pdf([[[Span("Landing Gear", 20, bold=True), Span(" Maintenance Manual", 20)], BODY]])

    assert inspect_pdf(pdf).largest_font_line == "Landing Gear Maintenance Manual"


def test_largest_font_line_ignores_decorative_and_page_number_lines():
    pdf = build_styled_pdf([[[Span("7", 40)], [Span("Relatório de Qualidade", 18)], BODY, BODY]])

    assert inspect_pdf(pdf).largest_font_line == "Relatório de Qualidade"


def test_page_without_a_standout_line_has_no_largest_font_line():
    assert inspect_pdf(build_pdf([BODY])).largest_font_line is None


def test_paragraphs_come_from_the_first_pages_in_order():
    pages = [["Capa"], ["Primeiro parágrafo.", "Segundo parágrafo."], ["Terceira página."], ["Quarta página."]]

    paragraphs = [" ".join(p.split()) for p in inspect_pdf(build_styled_pdf(pages)).paragraphs]

    assert paragraphs == ["Capa", "Primeiro parágrafo.", "Segundo parágrafo.", "Terceira página."]


def test_unreadable_bytes_raise_document_read_error():
    with pytest.raises(DocumentReadError):
        inspect_pdf(b"isto nao e um pdf")
