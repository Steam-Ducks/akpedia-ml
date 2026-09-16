"""Document reading and processing.

PDF is supported today; other formats are added by writing a ``DocumentTextExtractor``
and registering it in ``default_registry``, without touching the callers of
``extract_text``.
"""

from app.documents.base import DocumentTextExtractor, normalize_text_parts
from app.documents.errors import (
    DocumentProcessingError,
    DocumentReadError,
    UnsupportedDocumentFormatError,
)
from app.documents.pdf import PdfTextExtractor
from app.documents.registry import ExtractorRegistry, file_extension
from app.documents.service import default_registry, extract_text

__all__ = [
    "DocumentProcessingError",
    "DocumentReadError",
    "DocumentTextExtractor",
    "ExtractorRegistry",
    "PdfTextExtractor",
    "UnsupportedDocumentFormatError",
    "default_registry",
    "extract_text",
    "file_extension",
    "normalize_text_parts",
]
