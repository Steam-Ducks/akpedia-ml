"""Layout facts about a PDF that plain text extraction does not keep.

The text itself still comes from :func:`~app.documents.service.extract_text` (pypdf).
This module opens the same bytes with PyMuPDF only for what pypdf's text throws
away: the ``/Info`` title, the page count, font sizes and paragraph boundaries. pypdf
joins the lines of a page with single newlines, so a paragraph cannot be told apart
from the next one in its output; PyMuPDF's text blocks follow the visual layout.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pymupdf

from app.documents.errors import DocumentReadError

#: Pages scanned for paragraphs: the description is looked for near the start only.
PARAGRAPH_PAGES = 3

#: Lines shorter than this are decoration (a drop cap, a page number), not a title.
_MIN_TITLE_LINE = 3

#: Lines longer than this are body text that happens to be set large, not a title.
_MAX_TITLE_LINE = 300

#: How much larger than the body text a line must be to count as standing out, in points.
_SIZE_TOLERANCE = 0.5

#: ``get_text("dict")`` block type for text (``1`` is an image).
_TEXT_BLOCK = 0


@dataclass(frozen=True)
class PdfInspection:
    """What :func:`inspect_pdf` found in the document."""

    page_count: int
    #: ``/Info /Title`` as stored, or ``None`` when absent or blank.
    metadata_title: str | None
    #: Text of the line set in the largest font on the first page, when one stands out.
    largest_font_line: str | None
    #: Text blocks of the first :data:`PARAGRAPH_PAGES` pages, in reading order.
    paragraphs: tuple[str, ...]


def inspect_pdf(content: bytes) -> PdfInspection:
    """Open the PDF with PyMuPDF and collect its title candidates and paragraphs.

    Raises :class:`~app.documents.errors.DocumentReadError` when the file cannot be
    opened or is password protected.
    """
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass and not document.authenticate(""):
                raise DocumentReadError("Password-protected PDF.")
            title = (document.metadata or {}).get("title") or ""
            first_page = document[0].get_text("dict") if document.page_count else {"blocks": []}
            paragraphs = [
                block[4]
                for page in document.pages(0, min(PARAGRAPH_PAGES, document.page_count))
                for block in page.get_text("blocks", sort=True)
                if block[6] == _TEXT_BLOCK
            ]
            page_count = document.page_count
    except DocumentReadError:
        raise
    except (RuntimeError, ValueError) as exc:
        raise DocumentReadError(f"Could not read the PDF: {exc}") from exc

    return PdfInspection(
        page_count=page_count,
        metadata_title=title.strip() or None,
        largest_font_line=_largest_font_line(first_page),
        paragraphs=tuple(paragraphs),
    )


def _largest_font_line(page: dict) -> str | None:
    """Return the line set in the largest font, or ``None`` if nothing stands out.

    Sizes are compared per line, not per span: a title is often split into several
    spans (one word in bold, the rest regular), and the largest span alone would be a
    fragment of it. A title wrapped over several lines of the same block is joined
    back. A line "stands out" only when it is larger than the body text, the size
    carrying most characters on the page; otherwise the first line of a plain page
    would be taken for a title.
    """
    lines: list[tuple[int, float, str]] = []
    characters_by_size: Counter[float] = Counter()
    for block_index, block in enumerate(page["blocks"]):
        if block.get("type") != _TEXT_BLOCK:
            continue
        for line in block["lines"]:
            spans = [span for span in line["spans"] if span["text"].strip()]
            if not spans:
                continue
            text = " ".join("".join(span["text"] for span in spans).split())
            size = max(round(span["size"], 1) for span in spans)
            for span in spans:
                characters_by_size[round(span["size"], 1)] += len(span["text"].strip())
            if _MIN_TITLE_LINE <= len(text) <= _MAX_TITLE_LINE:
                lines.append((block_index, size, text))

    if not lines:
        return None
    body_size = characters_by_size.most_common(1)[0][0]
    largest = max(size for _, size, _ in lines)
    if largest <= body_size + _SIZE_TOLERANCE:
        return None

    start = next(index for index, (_, size, _) in enumerate(lines) if size == largest)
    title_block = lines[start][0]
    parts: list[str] = []
    for block_index, size, text in lines[start:]:
        if block_index != title_block or size != largest:
            break
        parts.append(text)
    return " ".join(parts)
