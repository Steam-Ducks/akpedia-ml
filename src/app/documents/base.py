"""Contract shared by every text extractor.

Adding a new format (DOCX, XLSX, TXT...) means writing a :class:`DocumentTextExtractor`
subclass and registering it in the ``ExtractorRegistry``. Nothing outside the
``app.documents`` package knows about concrete formats: the rest of the application
only calls ``extract_text`` and gets a ``str`` back.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import BinaryIO, ClassVar

_TRAILING_SPACES = re.compile(r"[ \t]+$", re.MULTILINE)
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


class DocumentTextExtractor(ABC):
    """Extracts text from one specific file format."""

    #: Handled file extensions, lowercase and dot-prefixed (e.g. ``(".pdf",)``).
    extensions: ClassVar[tuple[str, ...]] = ()

    #: Handled media types (MIME), lowercase (e.g. ``("application/pdf",)``).
    media_types: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def extract_text(self, source: BinaryIO) -> str:
        """Return the text held in ``source``.

        Implementations must raise :class:`~app.documents.errors.DocumentReadError`
        when the file cannot be read, and normalize their output through
        :func:`normalize_text_parts` so that every format hands the indexing pipeline
        text in the same shape.
        """


def normalize_text_parts(parts: Iterable[str]) -> str:
    """Join extracted fragments (pages, sheets, sections) into a single text.

    Keeps the output uniform across formats: newline endings, no trailing spaces, no
    runs of blank lines and no empty fragments. This is the text that later gets split
    into chunks and handed to the embedding model.
    """
    cleaned: list[str] = []
    for part in parts:
        if not part:
            continue
        text = part.replace("\r\n", "\n").replace("\r", "\n")
        text = _TRAILING_SPACES.sub("", text)
        text = _EXTRA_BLANK_LINES.sub("\n\n", text).strip()
        if text:
            cleaned.append(text)
    return "\n\n".join(cleaned)
