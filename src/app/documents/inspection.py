"""Layout facts about a PDF that plain text extraction does not keep.

The text itself still comes from :func:`~app.documents.service.extract_text` (pypdf).
This module opens the same bytes with PyMuPDF only for what pypdf's text throws
away: the ``/Info`` title, the page count, font sizes, positions and paragraph
boundaries. pypdf joins the lines of a page with single newlines, so a paragraph
cannot be told apart from the next one in its output; PyMuPDF's text blocks follow
the visual layout.

Page furniture is left out before anything is suggested: text repeated on several
pages (running headers such as "55451 Federal Register / Vol. 91, No. 166 / ...") and
lone page numbers in the top and bottom margins. Otherwise a page number set in a large
font becomes the title and the running header becomes the description. Blocks are read
column by column, so a multi-column page does not come out interleaved.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

import pymupdf

from app.documents.errors import DocumentReadError

#: Pages scanned for paragraphs: the description is looked for near the start only.
PARAGRAPH_PAGES = 3

#: Pages compared to find running headers and footers.
_REPEAT_PAGES = 5

#: Share of the page height, at the top and at the bottom, where headers and footers go.
_MARGIN_RATIO = 0.08

#: Blocks at least this share of the page width span the page instead of one column.
_FULL_WIDTH_RATIO = 0.6

#: Left edges further apart than this share of the page width start different columns.
_COLUMN_GAP_RATIO = 0.15

#: Lines shorter than this are decoration (a drop cap, a page number), not a title.
_MIN_TITLE_LINE = 3

#: Lines longer than this are body text that happens to be set large, not a title.
_MAX_TITLE_LINE = 300

#: How much larger than the body text a line must be to count as standing out, in points.
_SIZE_TOLERANCE = 0.5

#: Lines within this share of the largest size are tied for the title.
_SIZE_TIE_RATIO = 0.1

#: Largest gap between two lines of one title, as a share of the line height above.
_TITLE_LINE_GAP = 0.6

#: ``get_text("dict")`` block type for text (``1`` is an image).
_TEXT_BLOCK = 0

#: Federal Register documents (rules, notices, airworthiness directives) always open with
#: the subject line, then ``AGENCY:``, ``ACTION:`` and ``SUMMARY:``.
_AGENCY = re.compile(r"^AGENCY\s*:", re.IGNORECASE)
_SUMMARY = re.compile(r"^SUMMARY\s*:\s*", re.IGNORECASE)

_DIGITS = re.compile(r"\d+")


@dataclass(frozen=True)
class PdfInspection:
    """What :func:`inspect_pdf` found in the document."""

    page_count: int
    #: ``/Info /Title`` as stored, or ``None`` when absent or blank.
    metadata_title: str | None
    #: Text of the line set in the largest font on the first page, when one stands out.
    largest_font_line: str | None
    #: Text blocks of the first :data:`PARAGRAPH_PAGES` pages, in reading order,
    #: without headers, footers and running text.
    paragraphs: tuple[str, ...]
    #: Subject line right above ``AGENCY:``, in documents with that structure.
    regulatory_title: str | None = None
    #: Text of the ``SUMMARY:`` section, without the label, in documents that have one.
    regulatory_summary: str | None = None


class _Line(NamedTuple):
    """One line of text, its font size and where it sits on the page."""

    size: float
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class _Block:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    #: Whether the block sits in the top or bottom band where headers and footers go.
    in_margin: bool = False
    #: Non-blank lines, in order.
    lines: list[_Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return _join_lines(line.text for line in self.lines)


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
            page_count = document.page_count
            scanned = min(max(PARAGRAPH_PAGES, _REPEAT_PAGES), page_count)
            blocks = [block for index, page in enumerate(document.pages(0, scanned)) for block in _blocks(index, page)]
    except DocumentReadError:
        raise
    except (RuntimeError, ValueError) as exc:
        raise DocumentReadError(f"Could not read the PDF: {exc}") from exc

    kept = _without_page_furniture(blocks)
    body = [block for block in kept if not block.in_margin]
    margin = [block for block in kept if block.in_margin]
    # Not the de-duplicated blocks: a Federal Register PDF often carries several
    # documents of the same agency, so "AGENCY: Federal Aviation Administration (FAA),
    # DOT." repeats on more than one page and would be taken for a running header.
    regulatory_title, regulatory_summary = _regulatory_parts([block for block in blocks if not block.in_margin])
    # "The first page" is the first one with text: a cover that is only an image has
    # no title to read, and the title page is the one after it.
    title_page = min((block.page for block in kept), default=0)
    # The body size is measured over the first pages, not the title page alone: a cover
    # holds so little text that its title can be the size with the most characters.
    body_size = _body_size([block for block in body if block.page < title_page + PARAGRAPH_PAGES] or kept)
    return PdfInspection(
        page_count=page_count,
        metadata_title=title.strip() or None,
        largest_font_line=(
            _largest_font_line([block for block in body if block.page == title_page], body_size)
            # A title printed high on the page (a spreadsheet's first row, a tight top
            # margin) sits in the margin band: it is only looked at when nothing in the
            # body stands out.
            or _largest_font_line([block for block in kept if block.page == title_page], body_size)
        ),
        # Text in the margin band that survived is kept, but after the body: it may be
        # real content near the edge, or a header that only appears once.
        paragraphs=tuple(block.text for block in [*body, *margin] if block.page < PARAGRAPH_PAGES),
        regulatory_title=regulatory_title,
        regulatory_summary=regulatory_summary,
    )


def _blocks(page_index: int, page: pymupdf.Page) -> list[_Block]:
    """Text blocks of one page, in reading order, flagged when they sit in the margin band."""
    width, height = page.rect.width, page.rect.height
    top, bottom = height * _MARGIN_RATIO, height * (1 - _MARGIN_RATIO)
    blocks: list[_Block] = []
    for raw in page.get_text("dict")["blocks"]:
        if raw.get("type") != _TEXT_BLOCK:
            continue
        x0, y0, x1, y1 = raw["bbox"]
        block = _Block(page_index, x0, y0, x1, y1, in_margin=y1 <= top or y0 >= bottom)
        for line in raw["lines"]:
            spans = [span for span in line["spans"] if span["text"].strip()]
            if not spans:
                continue
            text = " ".join("".join(span["text"] for span in spans).split())
            # A page number printed large next to a running header must not make the
            # whole line look like a title: sizes come from the spans that have letters.
            sized = [span for span in spans if any(char.isalpha() for char in span["text"])] or spans
            block.lines.append(_Line(max(round(span["size"], 1) for span in sized), text, *line["bbox"]))
        if block.lines:
            blocks.append(block)
    return _reading_order(blocks, width)


def _reading_order(blocks: list[_Block], page_width: float) -> list[_Block]:
    """Order blocks the way the page is read: column by column, top to bottom.

    Sorting by position alone interleaves the columns of a two- or three-column page
    line by line. Blocks spanning most of the page width (a title, a full-width
    paragraph) split the page into bands; inside each band the narrower blocks are
    grouped into columns by their left edge and read one column after the other.
    """
    ordered: list[_Block] = []
    band: list[_Block] = []

    def flush() -> None:
        starts: list[float] = []
        for x0 in sorted(block.x0 for block in band):
            if not starts or x0 - starts[-1] > page_width * _COLUMN_GAP_RATIO:
                starts.append(x0)

        def column(block: _Block) -> int:
            return max(index for index, start in enumerate(starts) if block.x0 >= start)

        ordered.extend(sorted(band, key=lambda block: (column(block), block.y0)))
        band.clear()

    for block in sorted(blocks, key=lambda block: (block.y0, block.x0)):
        if block.x1 - block.x0 >= page_width * _FULL_WIDTH_RATIO:
            flush()
            ordered.append(block)
        else:
            band.append(block)
    flush()
    return ordered


def _without_page_furniture(blocks: list[_Block]) -> list[_Block]:
    """Drop running headers and footers, and margin text without words.

    A block is furniture when its text, numbers aside, shows up on more than one page
    (a running header repeats with only the page number changing), or when it sits in
    the margin band and has no words (a lone page number). Text in the margin band
    with words is kept: without proof that it repeats, it may be content printed close
    to the edge, like the first row of a converted spreadsheet.
    """

    def key(block: _Block) -> str:
        return " ".join(_DIGITS.sub("#", block.text).casefold().split())

    pages_by_key: dict[str, set[int]] = {}
    for block in blocks:
        pages_by_key.setdefault(key(block), set()).add(block.page)
    return [
        block
        for block in blocks
        if len(pages_by_key[key(block)]) < 2 and not (block.in_margin and not has_words(block.text))
    ]


def _regulatory_parts(blocks: list[_Block]) -> tuple[str | None, str | None]:
    """Subject line and summary of a document laid out as ``subject / AGENCY: / SUMMARY:``.

    The subject is found by position, not by order in the text: these documents are
    printed in columns and the first page usually starts with the end of the previous
    document, so "the line before AGENCY:" in reading order may belong to another
    column. It is the closest text above ``AGENCY:`` in the same column, or the lines
    before ``AGENCY:`` when the two share one block.
    """
    agency = next(
        (
            (index, line)
            for index, block in enumerate(blocks)
            for line, (_, text, *_) in enumerate(block.lines)
            if _AGENCY.match(text)
        ),
        None,
    )
    if agency is None:
        return None, None
    block_index, line_index = agency
    agency_block = blocks[block_index]

    if line_index > 0:
        subject = _join_lines(entry.text for entry in agency_block.lines[:line_index])
    else:
        above = [
            block
            for block in blocks
            if block.page == agency_block.page
            and block.y1 <= agency_block.y0 + 1
            and block.x0 < agency_block.x1
            and agency_block.x0 < block.x1
        ]
        subject = max(above, key=lambda block: block.y1).text if above else ""

    summary = None
    for block in blocks[block_index:]:
        for line, (_, text, *_) in enumerate(block.lines):
            if _SUMMARY.match(text):
                rest = [_SUMMARY.sub("", text), *(later.text for later in block.lines[line + 1 :])]
                summary = " ".join(_join_lines(rest).split()) or None
                break
        if summary:
            break

    subject = " ".join(subject.split())
    return (subject if _MIN_TITLE_LINE <= len(subject) <= _MAX_TITLE_LINE else None), summary


def _body_size(blocks: list[_Block]) -> float | None:
    """The font size carrying the most characters: the size of the running text."""
    characters_by_size: Counter[float] = Counter()
    for block in blocks:
        for line in block.lines:
            characters_by_size[line.size] += len(line.text)
    return characters_by_size.most_common(1)[0][0] if characters_by_size else None


def _largest_font_line(blocks: list[_Block], body_size: float | None) -> str | None:
    """Return the title-sized text of the page, or ``None`` if nothing stands out.

    Sizes are compared per line, not per span: a title is often split into several
    spans (one word in bold, the rest regular), and the largest span alone would be a
    fragment of it. A line "stands out" only when it is larger than the body text, the
    size carrying most characters over the first pages; otherwise the first line of a plain
    page would be taken for a title. Lines without words (page numbers, dates, codes)
    are never a title.

    Sizes within :data:`_SIZE_TIE_RATIO` of the largest count as a tie, and the tie
    goes to the line highest on the page: an OCR'd cover can measure a footer slogan a
    point larger than the title above it. The title then takes in the lines right
    below it set in the same size, even across blocks, since covers often put each
    line of a title in a block of its own.
    """
    lines = [line for block in blocks for line in block.lines]
    candidates = [
        line for line in lines if _MIN_TITLE_LINE <= len(line.text) <= _MAX_TITLE_LINE and has_words(line.text)
    ]
    if not candidates:
        return None

    if body_size is None:
        return None
    largest = max(line.size for line in candidates)
    floor = max(largest * (1 - _SIZE_TIE_RATIO), body_size + _SIZE_TOLERANCE)
    title_sized = sorted((line for line in candidates if line.size > floor - 1e-6), key=lambda line: (line.y0, line.x0))
    if largest <= body_size + _SIZE_TOLERANCE or not title_sized:
        return None

    start = title_sized[0]
    parts = [start]
    for line in title_sized[1:]:
        if line.x1 <= start.x0 or line.x0 >= start.x1:
            continue  # beside the title, in another column
        previous = parts[-1]
        if line.y0 - previous.y1 > (previous.y1 - previous.y0) * _TITLE_LINE_GAP:
            break
        parts.append(line)
    return " ".join(_join_lines(line.text for line in parts).split())


def has_words(text: str) -> bool:
    """Whether ``text`` reads as words rather than a number, date or code.

    At least three letters, and letters make up at least half of the non-space
    characters: "55451", "2026-17-07" and "§ 39.13" fail, "Airworthiness Directives;
    Lycoming Engines" passes.
    """
    letters = sum(char.isalpha() for char in text)
    visible = sum(not char.isspace() for char in text)
    return letters >= 3 and letters * 2 >= visible


def _join_lines(lines: Iterable[str]) -> str:
    """Join a block's lines, keeping words broken at a hyphen or dash in one piece.

    "2024–21–" followed by "02" on the next line is "2024–21–02", not "2024–21– 02".
    """
    joined = ""
    for line in lines:
        if not joined:
            joined = line
        elif joined.endswith(("-", "‐", "‑", "–")):
            joined += line
        else:
            joined += "\n" + line
    return joined
