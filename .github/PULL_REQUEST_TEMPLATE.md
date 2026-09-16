
## Description

## AKP-23 — Read the text inside the files (PDF)

**Status:** implemented, tests and lint passing on Python 3.12 (CI version). Not yet merged.

### What was delivered
A new `app.documents` package that reads a file and returns its text as a `str`, meeting the
DoD ("the method returns the text extracted from the files"). PDF is the only format implemented;
the package is structured so DOCX, XLSX and others plug in later without changing existing code.

Public entry point:
    extract_text(content, filename=..., media_type=...) -> str

### Design decisions and why
1. **Single entry point, no format leaks.** Callers (the upload endpoint from Task 2, the indexing
   pipeline) only call `extract_text` and never import a format-specific class. Adding a format can
   therefore never break a consumer.
2. **`DocumentTextExtractor` abstract contract.** Each format is one class declaring the extensions
   and MIME types it handles and implementing `extract_text(source) -> str`. This is the extension
   point required by the ticket.
3. **`ExtractorRegistry`.** Maps extension -> extractor, with MIME type as a fallback when the
   filename has no extension. A new format is a single `register()` call. Chosen over an if/elif
   chain because it keeps the PDF code untouched when new formats arrive (covered by a test).
4. **Output normalization shared by all formats.** Every extractor passes its fragments (pages,
   sheets, sections) through `normalize_text_parts`: Unix line endings, no trailing spaces, no runs
   of blank lines, empty pages dropped. Rationale: the ADR for the embedding model
   (multilingual-e5-small) requires the same text shape for every ingested format before chunking
   and the `passage:` prefix; normalizing here means the indexing step does not need per-format rules.
5. **Error hierarchy.** `DocumentProcessingError` (base) -> `UnsupportedDocumentFormatError`
   (no extractor for the format) and `DocumentReadError` (corrupt, truncated, password-protected).
   Lets the endpoint map them to distinct HTTP responses (e.g. 415 vs 422) without inspecting messages.
6. **Accepts bytes or an open binary stream.** Matches what FastAPI's `UploadFile` provides, so
   Task 2 can wire it in with one call.
7. **Text layer only, no OCR.** An image-only PDF (a scan without OCR) is read successfully and
   returns empty text instead of failing. OCR was left out of scope on purpose (see follow-ups).

### Dependencies added and why
- **pypdf 6.19 (runtime).** Pure-Python PDF reader, BSD-3 license, no native binaries or system
  packages, so it runs inside the existing slim Docker image and on CPU-only hardware, in line with
  the project's non-functional requirements. Actively maintained, supports encrypted PDFs and
  page-by-page extraction.
  Alternatives considered and rejected:
  - PyMuPDF (fitz): faster, but AGPL-licensed (commercial license required otherwise), which conflicts
    with the licensing driver already adopted in the embedding-model ADR; also ships a native binary.
  - pdfplumber / pdfminer.six: layout- and table-oriented, heavier and slower; those capabilities are
    not needed to return plain text for indexing.
- **No new dev dependencies.** The tests build minimal valid PDFs by hand (`tests/pdf_builder.py`)
  instead of adding a PDF-writing library such as reportlab just for fixtures.
- `uv.lock` regenerated because CI installs with `uv sync --frozen`.

### Tests (19 passing, all new ones added by this ticket)
- PDF: single page, multi-page in document order, accented text, pages without text dropped,
  reading from an open stream, case-insensitive extension (`.PDF`), resolution by MIME type when the
  filename is missing, non-PDF bytes and truncated PDF raising `DocumentReadError`.
- Registry / extension point: PDF registered by default, a fake `.txt` extractor plugged into a new
  registry while the PDF path stays unchanged, MIME-type fallback, unsupported format error listing
  the supported extensions, file without extension rejected, extractor without extensions rejected,
  call without filename or media type rejected, normalization behaviour.

### Verification
`uv run ruff check .`, `uv run ruff format --check .` and `uv run pytest` executed locally on
CPython 3.12.14 — the same version and commands the CI pipeline runs.

### Documentation
README updated: package layout, supported-formats table, usage example, error types, and a
step-by-step guide for adding a new format.

---

## Type of change

<!-- Mark with an "x" whatever applies. -->

- [X] `feat`: new feature
- [ ] `fix`: bug fix
- [ ] `refactor` / `perf` / `style`: refactor (no behavior change)
- [ ] `docs`: documentation
- [ ] `test`: tests
- [ ] `chore` / `build` / `ci`: configuration, build or structure

---
## Checklist

- [X] The branch follows the `AKP-<number>` pattern
- [X] The commits follow the `type(AKP-<number>): description` pattern
- [X] I ran the tests locally (`uv run pytest`) and they passed
- [X] I ran the lint locally (`uv run ruff check .`) and there are no violations
- [X] I updated the documentation, if needed
