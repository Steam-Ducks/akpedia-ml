"""HTTP endpoint that turns an uploaded document into embedded chunks.

The service is stateless: nothing here is stored. ``akpedia-server`` receives the
chunks with their vectors and is the only one that talks to the database.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app.documents import UnsupportedDocumentFormatError, extract_text, split_text
from app.documents.errors import NoExtractableTextError
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

    content = file.file.read()
    text = extract_text(content, filename=file.filename, media_type=file.content_type)
    chunks = split_text(text)
    if not chunks:
        raise NoExtractableTextError(
            f"No text could be extracted from '{file.filename or 'the uploaded file'}'. "
            "Scanned documents need OCR, which this service does not support yet."
        )

    vectors = embedder.embed(chunks)
    return ProcessDocumentResponse(
        filename=file.filename or "",
        model=EmbeddingModelInfo(name=embedder.model_name, dimensions=embedder.dimensions),
        chunk_count=len(chunks),
        chunks=[
            ProcessedChunk(index=index, text=chunk, embedding=vector)
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ],
    )
