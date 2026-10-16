"""Title and description suggested for an uploaded document.

Pure functions over what was already read from the file, so every rule can be tested
without building a PDF.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from pathlib import PurePosixPath

#: Size of ``documents.name`` in akpedia-server.
MAX_TITLE_LENGTH = 255

#: Longest description suggested, in characters.
MAX_DESCRIPTION_LENGTH = 500

#: Shorter paragraphs are headings, captions or labels, not a description.
MIN_DESCRIPTION_LENGTH = 40

#: Prefixes that office suites and print-to-PDF drivers put in front of the file name.
_PRODUCER_PREFIX = re.compile(
    r"^\s*(?:microsoft\s+(?:office\s+)?(?:word|powerpoint|excel)|libreoffice\s+\w+|openoffice\s+\w+)\s*[-–—:]\s*",
    re.IGNORECASE,
)

#: What is left of a producer prefix is the source file name, extension included.
_SOURCE_EXTENSION = re.compile(r"\.(?:docx?|dotx?|pptx?|xlsx?|odt|odp|ods|rtf|txt|pdf)$", re.IGNORECASE)

#: Placeholder titles, compared after :func:`_title_key` (accents, case, digits and
#: punctuation removed, so "Sem Título", "Documento1" and "Untitled-2" all match).
_GENERIC_TITLES = frozenset(
    {
        "untitled",
        "untitleddocument",
        "semtitulo",
        "documentosemtitulo",
        "document",
        "documento",
        "title",
        "titulo",
        "presentation",
        "apresentacao",
        "slide",
        "microsoftword",
        "pdf",
    }
)


def suggest_title(
    *,
    metadata_title: str | None,
    largest_font_line: str | None,
    filename: str | None,
) -> str:
    """Pick the document title, by priority.

    1. The PDF's ``/Info /Title``, without producer prefixes such as
       ``"Microsoft Word - "`` and ignored when it is a placeholder (``Untitled``).
    2. The line in the largest font on the first page.
    3. The file name without its extension.

    The result fits in :data:`MAX_TITLE_LENGTH`. It is empty only when the upload has
    no file name and the PDF gave nothing to go on.
    """
    candidates = (_clean_metadata_title(metadata_title), _collapse(largest_font_line), _filename_stem(filename))
    for candidate in candidates:
        if candidate:
            return truncate_at_word(candidate, MAX_TITLE_LENGTH)
    return ""


def suggest_description(paragraphs: Iterable[str], *, title: str | None = None) -> str | None:
    """Return the first paragraph long enough to describe the document.

    Paragraphs shorter than :data:`MIN_DESCRIPTION_LENGTH` and the paragraph holding
    just the title are skipped. Whitespace is collapsed and the result is cut at a word
    boundary to :data:`MAX_DESCRIPTION_LENGTH`. ``None`` when nothing qualifies.
    """
    title_key = _collapse(title).casefold()
    for paragraph in paragraphs:
        text = _collapse(paragraph)
        if len(text) < MIN_DESCRIPTION_LENGTH or text.casefold() == title_key:
            continue
        return truncate_at_word(text, MAX_DESCRIPTION_LENGTH)
    return None


def truncate_at_word(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` characters without splitting a word.

    The cut is hard only when the first word alone is already past the limit.
    """
    if len(text) <= limit:
        return text
    cut = text[: limit + 1].rsplit(" ", 1)[0].rstrip()
    if len(cut) > limit or not cut:
        return text[:limit]
    return cut


def _clean_metadata_title(title: str | None) -> str:
    # Print drivers often store the source file name as the title, with or without a
    # producer prefix: either way its extension is not part of the title.
    stripped = _SOURCE_EXTENSION.sub("", _PRODUCER_PREFIX.sub("", _collapse(title))).strip()
    if not stripped or _title_key(stripped) in _GENERIC_TITLES:
        return ""
    return stripped


def _filename_stem(filename: str | None) -> str:
    if not filename:
        return ""
    path = PurePosixPath(filename.replace("\\", "/"))
    return _collapse(path.stem if path.suffix else path.name)


def _title_key(title: str) -> str:
    decomposed = unicodedata.normalize("NFKD", title.casefold())
    return "".join(char for char in decomposed if "a" <= char <= "z")


def _collapse(text: str | None) -> str:
    return " ".join(text.split()) if text else ""
