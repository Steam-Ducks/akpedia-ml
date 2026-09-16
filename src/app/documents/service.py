"""Entry point of document processing.

The only module the rest of the application (endpoints, indexing pipeline) needs to
import: file content goes in, text comes out.
"""

from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

from app.documents.pdf import PdfTextExtractor
from app.documents.registry import ExtractorRegistry

#: Application-wide registry. New formats are registered here.
default_registry = ExtractorRegistry()
default_registry.register(PdfTextExtractor())


def extract_text(
    content: bytes | bytearray | BinaryIO,
    *,
    filename: str | None = None,
    media_type: str | None = None,
    registry: ExtractorRegistry | None = None,
) -> str:
    """Return the text extracted from the document.

    The format is picked from the extension of ``filename`` and, when it is missing,
    from ``media_type``. ``content`` accepts the raw file bytes or an already open
    binary stream.

    Raises :class:`~app.documents.errors.UnsupportedDocumentFormatError` when no
    extractor handles the format, and :class:`~app.documents.errors.DocumentReadError`
    when the file cannot be read.
    """
    if filename is None and media_type is None:
        raise ValueError("Provide filename or media_type to identify the document format.")

    extractor = (registry or default_registry).resolve(filename=filename, media_type=media_type)
    if isinstance(content, bytes | bytearray):
        with BytesIO(bytes(content)) as stream:
            return extractor.extract_text(stream)
    return extractor.extract_text(content)
