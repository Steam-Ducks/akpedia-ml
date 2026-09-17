from typing import BinaryIO, ClassVar

import pytest

from app.documents import (
    DocumentTextExtractor,
    ExtractorRegistry,
    PdfTextExtractor,
    UnsupportedDocumentFormatError,
    default_registry,
    extract_text,
    normalize_text_parts,
)
from tests.pdf_builder import build_pdf


class PlainTextExtractor(DocumentTextExtractor):
    """Stand-in for a future format (DOCX, XLSX...), used to exercise the extension point."""

    extensions: ClassVar[tuple[str, ...]] = (".txt",)
    media_types: ClassVar[tuple[str, ...]] = ("text/plain",)

    def extract_text(self, source: BinaryIO) -> str:
        return normalize_text_parts([source.read().decode("utf-8")])


def test_pdf_is_registered_by_default():
    assert default_registry.supported_extensions == (".pdf",)
    assert default_registry.supports("manual.pdf")
    assert not default_registry.supports("sheet.xlsx")


def test_a_new_format_plugs_in_without_changing_the_pdf_path():
    registry = ExtractorRegistry()
    registry.register(PdfTextExtractor())
    registry.register(PlainTextExtractor())

    pdf = build_pdf(["Still a PDF"])

    assert extract_text(pdf, filename="doc.pdf", registry=registry) == "Still a PDF"
    assert extract_text(b"plain content", filename="notes.txt", registry=registry) == "plain content"
    assert registry.supported_extensions == (".pdf", ".txt")


def test_resolution_falls_back_to_the_media_type():
    registry = ExtractorRegistry()
    registry.register(PlainTextExtractor())

    assert isinstance(registry.resolve(media_type="text/plain"), PlainTextExtractor)
    assert isinstance(registry.resolve(filename="unknown.bin", media_type="text/plain"), PlainTextExtractor)


def test_unsupported_format_error_lists_the_supported_extensions():
    with pytest.raises(UnsupportedDocumentFormatError) as error:
        extract_text(b"anything", filename="spreadsheet.xlsx")

    assert "spreadsheet.xlsx" in str(error.value)
    assert ".pdf" in str(error.value)


def test_file_without_extension_is_unsupported():
    with pytest.raises(UnsupportedDocumentFormatError):
        extract_text(b"anything", filename="README")


def test_registering_an_extractor_without_extensions_is_rejected():
    class NamelessExtractor(DocumentTextExtractor):
        def extract_text(self, source: BinaryIO) -> str:
            return ""

    with pytest.raises(ValueError, match="at least one extension"):
        ExtractorRegistry().register(NamelessExtractor())


def test_format_must_be_identifiable():
    with pytest.raises(ValueError, match="filename or media_type"):
        extract_text(b"anything")


def test_normalization_keeps_the_output_uniform_across_formats():
    parts = ["Title   \r\nline one  ", "", "   ", "Section\n\n\n\nnext paragraph\n\n"]

    assert normalize_text_parts(parts) == "Title\nline one\n\nSection\n\nnext paragraph"
