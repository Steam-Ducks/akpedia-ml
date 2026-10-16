import pytest

from app.analysis import (
    MAX_DESCRIPTION_LENGTH,
    MAX_TITLE_LENGTH,
    suggest_description,
    suggest_title,
    truncate_at_word,
)


def title(metadata: str | None = None, largest: str | None = None, filename: str | None = "arquivo.pdf") -> str:
    return suggest_title(metadata_title=metadata, largest_font_line=largest, filename=filename)


# --- title ------------------------------------------------------------------


def test_metadata_title_wins_over_the_other_rules():
    assert title("Manual de Manutenção", largest="Capa", filename="x.pdf") == "Manual de Manutenção"


@pytest.mark.parametrize(
    "metadata",
    [
        "Microsoft Word - Manual de Manutenção.docx",
        "Microsoft PowerPoint - Manual de Manutenção.pptx",
        "microsoft word – Manual de Manutenção",
        "Manual de Manutenção.pdf",
    ],
)
def test_producer_prefix_and_source_extension_are_removed(metadata: str):
    assert title(metadata) == "Manual de Manutenção"


@pytest.mark.parametrize("metadata", ["Untitled", "Sem título", "SEM TITULO", "Documento1", "untitled-2", "   "])
def test_generic_metadata_title_falls_through_to_the_largest_font(metadata: str):
    assert title(metadata, largest="Relatório de Qualidade") == "Relatório de Qualidade"


def test_prefix_leaving_only_a_generic_name_is_discarded():
    assert title("Microsoft Word - Documento1.docx", largest="Relatório Anual") == "Relatório Anual"


def test_largest_font_line_is_used_without_metadata():
    assert title(None, largest="  Relatório   de Qualidade ") == "Relatório de Qualidade"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("manual_trem_de_pouso.pdf", "manual_trem_de_pouso"),
        ("C:\\Users\\ana\\relatorio.final.pdf", "relatorio.final"),
        ("pasta/sem-extensao", "sem-extensao"),
    ],
)
def test_filename_without_extension_is_the_last_resort(filename: str, expected: str):
    assert title(None, largest=None, filename=filename) == expected


def test_title_is_empty_when_there_is_nothing_to_go_on():
    assert title(None, None, filename=None) == ""


def test_title_fits_the_server_column():
    long_title = " ".join(["Manutenção"] * 60)

    result = title(long_title)

    assert len(result) <= MAX_TITLE_LENGTH
    assert long_title.startswith(result)
    assert not result.endswith(" ")
    assert result.split()[-1] == "Manutenção"


# --- description ------------------------------------------------------------


def test_first_long_enough_paragraph_is_the_description():
    paragraphs = ["Sumário", "1. Introdução", "Este manual descreve os procedimentos de inspeção.", "Outro parágrafo."]

    assert suggest_description(paragraphs) == "Este manual descreve os procedimentos de inspeção."


def test_paragraph_equal_to_the_title_is_skipped():
    heading = "Procedimentos de inspeção do trem de pouso principal"
    paragraphs = [heading, "Este manual descreve os procedimentos de inspeção periódica."]

    assert suggest_description(paragraphs, title=heading.upper()) == paragraphs[1]


def test_whitespace_is_normalized():
    paragraph = "Este manual   descreve\nos procedimentos\n\tde inspeção periódica.\n"

    assert suggest_description([paragraph]) == "Este manual descreve os procedimentos de inspeção periódica."


def test_description_is_cut_at_a_word_boundary():
    paragraph = "palavra " * 100

    result = suggest_description([paragraph])

    assert result is not None
    assert len(result) <= MAX_DESCRIPTION_LENGTH
    assert result.split() == ["palavra"] * len(result.split())
    assert not result.endswith(" ")


def test_description_is_none_when_no_paragraph_qualifies():
    assert suggest_description(["Capa", "Sumário", "   "]) is None


# --- truncation -------------------------------------------------------------


def test_text_within_the_limit_is_untouched():
    assert truncate_at_word("curto", 10) == "curto"


def test_cut_falls_on_the_last_whole_word():
    assert truncate_at_word("um dois tres quatro", 12) == "um dois tres"
    assert truncate_at_word("um dois tres quatro", 11) == "um dois"


def test_single_word_past_the_limit_is_cut_hard():
    assert truncate_at_word("anticonstitucionalissimamente", 10) == "anticonsti"
