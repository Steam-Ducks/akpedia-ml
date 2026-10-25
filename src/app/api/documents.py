"""HTTP endpoints for uploaded documents.

``/process`` turns a document into embedded chunks for indexing; ``/analyze``
suggests a title, description and category for the upload form.

The service is stateless: nothing here is stored. ``akpedia-server`` receives the
results and is the only one that talks to the database.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app.analysis import (
    CategoryCentroids,
    document_vector,
    get_centroids,
    sample_chunks,
    suggest_description,
    suggest_title,
)
from app.config import ANALYSIS_MAX_CHUNKS, CATEGORY_TEMPERATURE, MAX_UPLOAD_BYTES
from app.documents import UnsupportedDocumentFormatError, extract_text, inspect_pdf, split_text
from app.documents.errors import FileTooLargeError, NoExtractableTextError
from app.embeddings import Embedder, get_embedder

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


class ProcessedChunk(BaseModel):
    """One chunk of the document and its numeric representation."""

    index: int = Field(description="Position of the chunk in the document, starting at 0.", examples=[0])
    text: str = Field(description="Chunk text, as extracted.", examples=["Primeiro trecho do documento."])
    embedding: list[float] = Field(description="Normalized vector of the chunk.", examples=[[0.012, -0.045]])


class EmbeddingModelInfo(BaseModel):
    """Which model produced the vectors, and in what shape."""

    name: str = Field(description="Model identifier.", examples=["intfloat/multilingual-e5-small"])
    dimensions: int = Field(description="Size of every vector.", examples=[384])
    normalized: bool = Field(default=True, description="Vectors have length 1, so cosine equals dot product.")


class ProcessDocumentResponse(BaseModel):
    """Everything ``akpedia-server`` needs to index the document."""

    # ``model`` collides with Pydantic's reserved ``model_`` namespace.
    model_config = ConfigDict(protected_namespaces=())

    filename: str = Field(description="Name of the uploaded file.", examples=["manual.pdf"])
    model: EmbeddingModelInfo
    chunk_count: int = Field(description="Number of chunks returned.", examples=[2])
    chunks: list[ProcessedChunk]


class AnalyzedDocument(BaseModel):
    """Suggestions to pre-fill the upload form. The user can change any of them."""

    title: str = Field(description="Suggested title, at most 255 characters.", examples=["Landing Gear Manual"])
    description: str | None = Field(
        description="First substantial paragraph, at most 500 characters; null when there is none.",
        examples=["This manual describes the inspection procedures..."],
    )
    category: str | None = Field(
        description="Most likely category; null when no category centroids are deployed.", examples=["Tecnico"]
    )
    confidence: float | None = Field(description="Probability of `category`, from 0 to 1.", examples=[0.82])
    scores: dict[str, float] = Field(
        description="Probability of every category, summing to 1; empty when `category` is null.",
        examples=[{"Tecnico": 0.82, "Regulatorio": 0.09, "Juridico": 0.03, "Qualidade": 0.06}],
    )
    pages: int = Field(description="Number of pages in the PDF.", examples=[34])


class ErrorResponse(BaseModel):
    """Body of every error answered by this service."""

    code: str = Field(description="Stable identifier of the failure.", examples=["unsupported_format"])
    message: str = Field(description="Human readable description.")
    supported_extensions: list[str] | None = Field(
        default=None, description="Formats this service handles, when the failure is about the format."
    )


@router.post(
    "/process",
    response_model=ProcessDocumentResponse,
    summary="Extract, chunk and embed a document",
    responses={
        413: {"model": ErrorResponse, "description": "Upload is larger than the accepted limit."},
        415: {"model": ErrorResponse, "description": "No extractor handles this format."},
        422: {"model": ErrorResponse, "description": "File could not be read, or holds no text."},
    },
)
def process_document(
    file: UploadFile,
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> ProcessDocumentResponse:
    """Read the document, split it into chunks and return one vector per chunk.

    The format is picked from the file extension and, when it is missing, from the
    upload's media type.
    """
    if not file.filename and not file.content_type:
        raise UnsupportedDocumentFormatError("The upload has neither a filename nor a media type.")

    content = _read_within_limit(file)
    text = extract_text(content, filename=file.filename, media_type=file.content_type)
    chunks = split_text(text)
    if not chunks:
        raise no_text_error(file.filename)

    return build_process_response(file.filename, embedder, chunks, embedder.embed(chunks))


def build_process_response(
    filename: str | None,
    embedder: Embedder,
    chunks: list[str],
    vectors: list[list[float]],
) -> ProcessDocumentResponse:
    """Assemble the ``/process`` answer; the indexing worker sends akpedia-server the same body."""
    return ProcessDocumentResponse(
        filename=filename or "",
        model=EmbeddingModelInfo(name=embedder.model_name, dimensions=embedder.dimensions),
        chunk_count=len(chunks),
        chunks=[
            ProcessedChunk(index=index, text=chunk, embedding=vector)
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ],
    )


@router.post(
    "/analyze",
    response_model=AnalyzedDocument,
    summary="Suggest a title, description and category for a document",
    responses={
        413: {"model": ErrorResponse, "description": "Upload is larger than the accepted limit."},
        415: {"model": ErrorResponse, "description": "No extractor handles this format."},
        422: {"model": ErrorResponse, "description": "File could not be read, or holds no text."},
    },
)
def analyze_document(
    file: UploadFile,
    embedder: Annotated[Embedder, Depends(get_embedder)],
    centroids: Annotated[CategoryCentroids | None, Depends(get_centroids)],
) -> AnalyzedDocument:
    """Read a PDF and suggest what the upload form asks for.

    Only the text layer is read: a scanned document without one answers ``422``.
    Without a centroids file the category is ``null``; the title and description are
    still suggested. Nothing is stored and no vector is returned.
    """
    if not file.filename and not file.content_type:
        raise UnsupportedDocumentFormatError("The upload has neither a filename nor a media type.")

    content = _read_within_limit(file)
    # The registry goes first: an unsupported format is a 415 from it, not a PDF
    # that PyMuPDF failed to open.
    text = extract_text(content, filename=file.filename, media_type=file.content_type)
    chunks = split_text(text)
    if not chunks:
        raise no_text_error(file.filename)

    inspection = inspect_pdf(content)
    title = suggest_title(
        metadata_title=inspection.metadata_title,
        largest_font_line=inspection.largest_font_line,
        regulatory_title=inspection.regulatory_title,
        filename=file.filename,
    )
    description = suggest_description(inspection.paragraphs, title=title, summary=inspection.regulatory_summary)

    category = None
    if centroids is not None:
        vectors = embedder.embed(sample_chunks(chunks, ANALYSIS_MAX_CHUNKS))
        category = centroids.score(document_vector(vectors), temperature=CATEGORY_TEMPERATURE)

    return AnalyzedDocument(
        title=title,
        description=description,
        category=category.category if category else None,
        confidence=category.confidence if category else None,
        scores=category.scores if category else {},
        pages=inspection.page_count,
    )


def no_text_error(filename: str | None) -> NoExtractableTextError:
    """The failure for a document whose text layer is empty, worded the same everywhere."""
    return NoExtractableTextError(
        f"No text could be extracted from '{filename or 'the uploaded file'}'. "
        "Scanned documents need OCR, which this service does not support yet."
    )


def _read_within_limit(file: UploadFile) -> bytes:
    """Return the upload's bytes, refusing anything past :data:`MAX_UPLOAD_BYTES`.

    The read is capped instead of unbounded so an oversized upload is not pulled into
    memory only to be rejected afterwards: one byte over the limit already answers the
    question. ``file.size`` is checked first because it settles the common case without
    reading anything at all.
    """
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise FileTooLargeError(_too_large_message(file))

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise FileTooLargeError(_too_large_message(file))
    return content


def _too_large_message(file: UploadFile) -> str:
    name = f"'{file.filename}'" if file.filename else "The uploaded file"
    return f"{name} is larger than the {_human_size(MAX_UPLOAD_BYTES)} limit."


def _human_size(size: int) -> str:
    """Render a byte count in the unit the limit was most likely written in."""
    mebibyte = 1024 * 1024
    return f"{size / mebibyte:g} MiB" if size >= mebibyte else f"{size} bytes"
