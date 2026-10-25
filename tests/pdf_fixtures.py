"""Build PDFs with document metadata and per-line font sizes, which ``pdf_builder`` cannot express.

The analysis endpoint reads the ``/Info /Title`` entry and compares font sizes on the
first page, so its tests need control over both. Lines are laid out top-down with a
generous gap between them, so every line reads as a paragraph of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

BODY_SIZE = 11.0


@dataclass(frozen=True)
class Span:
    """A run of text in one font: a line may mix bold and regular spans."""

    text: str
    size: float = BODY_SIZE
    bold: bool = False


#: A line is plain text in the body size, or a list of spans printed side by side.
Line = str | Sequence[Span]


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _spans(line: Line) -> Sequence[Span]:
    return [Span(line)] if isinstance(line, str) else line


@dataclass(frozen=True)
class Columns:
    """A page printed in side-by-side columns, each laid out from the top on its own."""

    columns: Sequence[Sequence[Line]]


#: Where the body starts, below the band the analysis treats as the page header.
_BODY_TOP = 700.0
_COLUMN_WIDTH = 180.0


def _draw(commands: list[str], line: Line, x: float, y: float, page_number: int) -> None:
    # Each span is positioned explicitly: a rough advance of half the font size per
    # character keeps spans from overlapping without measuring glyphs.
    for span in _spans(line):
        font = "F2" if span.bold else "F1"
        text = _escape(span.text.replace("{page}", str(page_number)))
        commands.append(f"BT /{font} {span.size:g} Tf {x:.1f} {y:.1f} Td ({text}) Tj ET")
        x += len(span.text) * span.size * 0.55


def _content_stream(
    page: Sequence[Line] | Columns, page_number: int, header: Line | None, footer: Line | None
) -> bytes:
    commands: list[str] = []
    columns = page.columns if isinstance(page, Columns) else [page]
    for index, lines in enumerate(columns):
        y = _BODY_TOP
        for line in lines:
            y -= max(span.size for span in _spans(line)) * 2.5
            _draw(commands, line, 72.0 + index * _COLUMN_WIDTH, y, page_number)
    if header is not None:
        _draw(commands, header, 72.0, 770.0, page_number)
    if footer is not None:
        _draw(commands, footer, 72.0, 24.0, page_number)
    return "\n".join(commands).encode("cp1252")


def build_styled_pdf(
    pages: Sequence[Sequence[Line] | Columns],
    *,
    title: str | None = None,
    header: Line | None = None,
    footer: Line | None = None,
    first_page_number: int = 1,
) -> bytes:
    """Build a PDF with the given lines per page and, optionally, an ``/Info /Title``.

    ``header`` and ``footer`` are printed on every page, in the top and bottom margins,
    with ``{page}`` replaced by the page number.
    """
    first_page_id = 6
    page_ids = [first_page_id + 2 * index for index in range(len(pages))]
    content_ids = [page_id + 1 for page_id in page_ids]

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    info = f"<< /Title ({_escape(title)}) >>" if title is not None else "<< >>"
    bodies: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode(),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        5: info.encode("cp1252"),
    }
    for index, (page, page_id, content_id) in enumerate(zip(pages, page_ids, content_ids, strict=True)):
        bodies[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        stream = _content_stream(page, first_page_number + index, header, footer)
        bodies[content_id] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for object_id in sorted(bodies):
        offsets[object_id] = len(out)
        out += b"%d 0 obj\n%s\nendobj\n" % (object_id, bodies[object_id])

    startxref = len(out)
    size = max(bodies) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for object_id in range(1, size):
        out += b"%010d 00000 n \n" % offsets[object_id]
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info 5 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, startxref)
    return bytes(out)


#: Running header of a Federal Register page: small text and a large page number.
FEDERAL_REGISTER_HEADER = [
    Span("Federal Register / Vol. 91, No. 166 / Friday, August 28, 2026 / Rules and Regulations", 8),
    Span("   {page}", 16, bold=True),
]


def build_federal_register_pdf() -> bytes:
    """A two-page airworthiness directive laid out like the Federal Register.

    Mirrors what made the first version suggest "55451" as the title: the page number
    is the largest text on the page, the first page opens with the end of the previous
    document (whose heading is set larger than this one's subject), and the subject,
    ``AGENCY:`` and ``SUMMARY:`` sit in the right-hand column.
    """
    previous_document = [
        [Span("PART 39-AIRWORTHINESS", 12, bold=True)],
        [Span("1. The authority citation", 8)],
        [Span("for part 39 continues", 8)],
        [Span("(j) Alternative Methods", 8)],
        [Span("of Compliance (AMOCs)", 8)],
    ]
    this_document = [
        [Span("DEPARTMENT OF TRANSPORTATION", 9, bold=True)],
        [Span("Federal Aviation Administration", 9, bold=True)],
        [Span("14 CFR Part 39", 9, bold=True)],
        [Span("[Docket No. FAA-2026-8789]", 8)],
        [Span("RIN 2120-AA64", 8)],
        [Span("Airworthiness Directives; Lycoming Engines", 10, bold=True)],
        [Span("AGENCY: Federal Aviation Administration, DOT.", 8)],
        [Span("ACTION: Final rule.", 8)],
        [Span("SUMMARY: The FAA adopts an AD for Lycoming engines.", 8)],
    ]
    return build_styled_pdf(
        [
            Columns([previous_document, this_document]),
            [[Span("DATES: This AD is effective October 2, 2026, for the listed engines.", 8)]],
        ],
        header=FEDERAL_REGISTER_HEADER,
        first_page_number=55451,
    )
