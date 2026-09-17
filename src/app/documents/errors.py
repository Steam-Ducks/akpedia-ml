"""Errors raised by document processing.

They all inherit from :class:`DocumentProcessingError`, so callers can catch a single
exception or tell an unsupported format apart from an unreadable file when translating
the failure into an HTTP response.
"""


class DocumentProcessingError(Exception):
    """Base class for any failure while reading or processing a document."""


class UnsupportedDocumentFormatError(DocumentProcessingError):
    """No registered extractor handles the file format."""


class DocumentReadError(DocumentProcessingError):
    """The format is supported, but the file could not be read (corrupt, protected, truncated)."""


class NoExtractableTextError(DocumentProcessingError):
    """The file was read successfully, but holds no text (a scan without OCR)."""
