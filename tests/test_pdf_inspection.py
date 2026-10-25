import pytest

from app.documents import DocumentReadError, inspect_pdf
from app.documents.inspection import _join_lines, has_words
from tests.pdf_builder import build_pdf
from tests.pdf_fixtures import FEDERAL_REGISTER_HEADER, Columns, Span, build_federal_register_pdf, build_styled_pdf

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


# --- page furniture ----------------------------------------------------------


def test_running_header_with_a_large_page_number_is_left_out():
    # What happened with the Federal Register: the page number was the largest text,
    # and the running header was the first long paragraph.
    pdf = build_styled_pdf([[BODY], [BODY]], header=FEDERAL_REGISTER_HEADER, first_page_number=55451)

    inspection = inspect_pdf(pdf)

    assert inspection.largest_font_line is None
    assert not any("55451" in paragraph or "Federal Register" in paragraph for paragraph in inspection.paragraphs)


def test_header_of_a_single_page_document_is_neither_title_nor_first_paragraph():
    # One page gives no repetition to prove the header is furniture, so it is kept,
    # but the large page number next to it does not make it a title, and the body
    # still comes first.
    pdf = build_styled_pdf([[BODY, BODY]], header=FEDERAL_REGISTER_HEADER, first_page_number=55451)

    inspection = inspect_pdf(pdf)

    assert inspection.largest_font_line is None
    assert " ".join(inspection.paragraphs[0].split()) == BODY
    assert "Federal Register" in inspection.paragraphs[-1]


def test_footer_repeated_on_every_page_is_left_out():
    footer = "Documento controlado - copia nao controlada quando impressa"
    pdf = build_styled_pdf([[BODY], ["Segunda pagina com outro texto."]], footer=footer)

    paragraphs = [" ".join(p.split()) for p in inspect_pdf(pdf).paragraphs]

    assert paragraphs == [BODY, "Segunda pagina com outro texto."]


def test_lone_page_number_in_the_margin_is_left_out():
    pdf = build_styled_pdf([[BODY]], footer="- {page} -", first_page_number=7)

    assert [" ".join(p.split()) for p in inspect_pdf(pdf).paragraphs] == [BODY]


def test_title_printed_in_the_top_margin_is_still_found():
    # A converted spreadsheet starts close to the top edge: its title row sits in the
    # margin band, and nothing in the body stands out.
    pdf = build_styled_pdf([["Item A - 10 unidades", "Item B - 20 unidades"]], header=[Span("Planilha de Custos", 18)])

    assert inspect_pdf(pdf).largest_font_line == "Planilha de Custos"


def test_columns_are_read_one_after_the_other():
    pdf = build_styled_pdf([Columns([["Coluna um, linha um.", "Coluna um, linha dois."], ["Coluna dois, linha um."]])])

    paragraphs = [" ".join(p.split()) for p in inspect_pdf(pdf).paragraphs]

    assert paragraphs == ["Coluna um, linha um.", "Coluna um, linha dois.", "Coluna dois, linha um."]


def test_text_repeated_on_every_page_is_left_out_even_outside_the_margins():
    running = "Manual de Manutencao - Revisao 4 - pagina {page}"
    pdf = build_styled_pdf([[running, BODY], [running, "Segunda pagina com outro texto."]])

    paragraphs = [" ".join(p.split()) for p in inspect_pdf(pdf).paragraphs]

    assert paragraphs == [BODY, "Segunda pagina com outro texto."]


def test_single_page_keeps_text_that_could_only_repeat_with_more_pages():
    assert len(inspect_pdf(build_styled_pdf([["Linha unica.", BODY]])).paragraphs) == 2


@pytest.mark.parametrize("line", ["55451", "2026-17-07", "§ 39.13", "12/08", "AB"])
def test_lines_without_words_are_not_a_title(line: str):
    pdf = build_styled_pdf([[[Span(line, 24)], BODY, BODY]])

    assert inspect_pdf(pdf).largest_font_line is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Airworthiness Directives; Lycoming Engines", True),
        ("Relatório de Qualidade 2026", True),
        ("55451", False),
        ("2026-17-07", False),
        ("§ 39.13", False),
    ],
)
def test_has_words(text: str, expected: bool):
    assert has_words(text) is expected


# --- Federal Register structure -----------------------------------------------


def test_federal_register_subject_and_summary_are_found_by_position():
    inspection = inspect_pdf(build_federal_register_pdf())

    assert inspection.regulatory_title == "Airworthiness Directives; Lycoming Engines"
    assert inspection.regulatory_summary == "The FAA adopts an AD for Lycoming engines."
    # The largest text on the page belongs to the previous document: that is why the
    # subject line is looked for next to AGENCY: and not by font size.
    assert inspection.largest_font_line == "PART 39-AIRWORTHINESS"


def test_documents_without_agency_have_no_regulatory_parts():
    inspection = inspect_pdf(
        build_styled_pdf([[[Span("Manual de Manutencao", 20)], BODY, "SUMMARY: nao vale sozinho."]])
    )

    assert inspection.regulatory_title is None
    assert inspection.regulatory_summary is None


# --- title on covers -----------------------------------------------------------


def test_near_tie_in_size_goes_to_the_line_higher_on_the_page():
    # An OCR'd cover measured its footer slogan a point larger than its title.
    pdf = build_styled_pdf(
        [[[Span("Powerlines and Thunderstorms", 27)], BODY, BODY, [Span("accident prevention", 28)]]]
    )

    assert inspect_pdf(pdf).largest_font_line == "Powerlines and Thunderstorms"


def test_clearly_larger_line_still_wins_over_a_higher_one():
    pdf = build_styled_pdf([[[Span("Secao 1", 14)], BODY, BODY, [Span("Titulo Principal", 28)]]])

    assert inspect_pdf(pdf).largest_font_line == "Titulo Principal"


def test_title_page_is_the_first_page_with_text():
    # A cover that is only an image has no text: the title page is the next one.
    pdf = build_styled_pdf([[], [[Span("Balloon Flying Handbook", 22)], BODY, BODY]])

    assert inspect_pdf(pdf).largest_font_line == "Balloon Flying Handbook"


def test_body_size_comes_from_the_first_pages_not_the_cover_alone():
    # The cover's title is the size with the most characters on the cover itself.
    cover = [[Span("Manual de Operacao do Balao", 26)], [Span("Edicao 2026", 9)]]
    pdf = build_styled_pdf([cover, [BODY, BODY, BODY]])

    assert inspect_pdf(pdf).largest_font_line == "Manual de Operacao do Balao"


def test_line_ending_in_a_dash_is_joined_without_a_space():
    assert _join_lines(["superseding AD 2024–21–", "02, which applies"]) == "superseding AD 2024–21–02, which applies"
    assert _join_lines(["primeira linha", "segunda linha"]) == "primeira linha\nsegunda linha"
