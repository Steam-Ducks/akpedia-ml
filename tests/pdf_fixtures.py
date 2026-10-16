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


def _content_stream(lines: Sequence[Line]) -> bytes:
    commands: list[str] = []
    y = 760.0
    for line in lines:
        spans = _spans(line)
        size = max(span.size for span in spans)
        y -= size * 2.5
        # Each span is positioned explicitly: a rough advance of half the font size per
        # character keeps spans from overlapping without measuring glyphs.
        x = 72.0
        for span in spans:
            font = "F2" if span.bold else "F1"
            commands.append(f"BT /{font} {span.size:g} Tf {x:.1f} {y:.1f} Td ({_escape(span.text)}) Tj ET")
            x += len(span.text) * span.size * 0.55
    return "\n".join(commands).encode("cp1252")


def build_styled_pdf(pages: Sequence[Sequence[Line]], *, title: str | None = None) -> bytes:
    """Build a PDF with the given lines per page and, optionally, an ``/Info /Title``."""
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
    for lines, page_id, content_id in zip(pages, page_ids, content_ids, strict=True):
        bodies[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        stream = _content_stream(lines)
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
