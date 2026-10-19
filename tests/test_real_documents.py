"""Title and description suggestions against real documents, as a regression net.

The PDFs in ``tests/fixtures/real`` are public-domain US government publications of the
kind the service receives:

- ``federal-register-2026-17584.pdf``: AD 2026-17-10 (91 FR 55451). The document the
  first version titled "55451" and described with the running header.
  https://www.govinfo.gov/content/pkg/FR-2026-08-28/pdf/2026-17584.pdf
- ``federal-register-2026-04281.pdf``: AD 2026-04-11 (91 FR 10506). Carries a second
  FAA document, so "AGENCY: ..." shows up on two pages.
  https://www.govinfo.gov/content/pkg/FR-2026-03-04/pdf/2026-04281.pdf
- ``faa-p-8740-34-powerlines-and-thunderstorms.pdf``: FAA "Balloon Safety Tips"
  pamphlet, a scan with an OCR text layer whose cover holds little text.
  https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/powerlines_and_thunderstorms.pdf

Every case runs with and without the PDF's own ``/Info /Title``: an upload converted
on the way loses it, and the layout rules must stand on their own.
"""

from pathlib import Path

import pytest

from app.analysis import suggest_description, suggest_title
from app.documents import inspect_pdf

FIXTURES = Path(__file__).parent / "fixtures" / "real"

CASES = [
    (
        "federal-register-2026-17584.pdf",
        "Airworthiness Directives; Lycoming Engines",
        "The FAA is adopting a new airworthiness directive (AD) for certain Lycoming Engines (Lycoming) Model",
    ),
    (
        "federal-register-2026-04281.pdf",
        "Airworthiness Directives; Lycoming Engines",
        "The FAA is superseding Airworthiness Directive (AD) 2024–21–02, which applies to Lycoming",
    ),
    (
        "faa-p-8740-34-powerlines-and-thunderstorms.pdf",
        "Powerlines and Thunderstorms",
        "The purpose of this series of publications is to provide the general aviation flying public",
    ),
]


@pytest.mark.parametrize("use_metadata", [True, False], ids=["with-metadata", "without-metadata"])
@pytest.mark.parametrize(("filename", "title", "description_start"), CASES, ids=[case[0] for case in CASES])
def test_real_document_gets_its_title_and_description(
    filename: str, title: str, description_start: str, use_metadata: bool
):
    inspection = inspect_pdf((FIXTURES / filename).read_bytes())

    suggested_title = suggest_title(
        regulatory_title=inspection.regulatory_title,
        metadata_title=inspection.metadata_title if use_metadata else None,
        largest_font_line=inspection.largest_font_line,
        filename=filename,
    )
    description = suggest_description(
        inspection.paragraphs, title=suggested_title, summary=inspection.regulatory_summary
    )

    # The pamphlet's cover adds a subtitle under the title when the layout is read.
    assert suggested_title.startswith(title)
    assert not any(char.isdigit() for char in suggested_title)
    assert description is not None and description.startswith(description_start)
