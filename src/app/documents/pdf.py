"""Text extraction for PDF files."""

from __future__ import annotations

from typing import BinaryIO, ClassVar

from pypdf import PasswordType, PdfReader
from pypdf.errors import PdfReadError

from app.documents.base import DocumentTextExtractor, normalize_text_parts
from app.documents.errors import DocumentReadError


class PdfTextExtractor(DocumentTextExtractor):
    """Read the text embedded in every page of the PDF, in document order.

    Only the text layer is extracted: an image-only PDF (a scan without OCR) is read
    successfully and yields empty text.
    """

    extensions: ClassVar[tuple[str, ...]] = (".pdf",)
    media_types: ClassVar[tuple[str, ...]] = ("application/pdf",)

    def extract_text(self, source: BinaryIO) -> str:
        try:
            reader = PdfReader(source)
            if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
                raise DocumentReadError("Password-protected PDF.")
            pages = [page.extract_text() or "" for page in reader.pages]
        except DocumentReadError:
            raise
        except (PdfReadError, ValueError, OSError, KeyError, TypeError) as exc:
            raise DocumentReadError(f"Could not read the PDF: {exc}") from exc
        return normalize_text_parts(pages)
