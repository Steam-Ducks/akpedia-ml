"""Document reading and processing.

PDF is supported today; other formats are added by writing a ``DocumentTextExtractor``
and registering it in ``default_registry``, without touching the callers of
``extract_text``. ``split_text`` then turns the extracted text into overlapping,
sentence-aligned chunks ready for embedding.
"""

from app.documents.base import DocumentTextExtractor, normalize_text_parts
from app.documents.chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, split_sentences, split_text
from app.documents.errors import (
    DocumentProcessingError,
    DocumentReadError,
    FileTooLargeError,
    NoExtractableTextError,
    UnsupportedDocumentFormatError,
)
from app.documents.pdf import PdfTextExtractor
from app.documents.registry import ExtractorRegistry, file_extension
from app.documents.service import default_registry, extract_text

__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DocumentProcessingError",
    "DocumentReadError",
    "DocumentTextExtractor",
    "ExtractorRegistry",
    "FileTooLargeError",
    "NoExtractableTextError",
    "PdfTextExtractor",
    "UnsupportedDocumentFormatError",
    "default_registry",
    "extract_text",
    "file_extension",
    "normalize_text_parts",
    "split_sentences",
    "split_text",
]
