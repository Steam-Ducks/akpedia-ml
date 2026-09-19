from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.documents import router as documents_router
from app.api.embeddings import EmptyQueryError, QueryTooLongError
from app.api.embeddings import router as embeddings_router
from app.api.schemas import ErrorResponse
from app.documents import (
    DocumentReadError,
    FileTooLargeError,
    NoExtractableTextError,
    UnsupportedDocumentFormatError,
    default_registry,
)
from app.embeddings import get_embedder


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the embedding model before the first request is served.

    Loading costs seconds and hundreds of MB. Doing it here, instead of lazily on the
    first call, means no user pays for it and the process only reports itself as ready
    once the model is actually in memory -- which is what an orchestrator polling
    ``/health`` should be told. Every later request reuses the same instance, since
    ``get_embedder`` is cached.

    Tests that swap the embedder through ``dependency_overrides`` never load the real
    model: there is nothing to warm up for an instance that will not be used.
    """
    if get_embedder not in app.dependency_overrides:
        get_embedder()
    yield


app = FastAPI(title="akpedia-ml", version="0.1.0", lifespan=lifespan)

app.include_router(documents_router)
app.include_router(embeddings_router)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check simples usado por orquestração e pelo CI."""
    return {"status": "UP"}


def _error(status_code: int, error: ErrorResponse) -> JSONResponse:
    # ``exclude_none`` keeps ``supported_extensions`` out of the failures it says
    # nothing about, instead of answering them with a null field.
    return JSONResponse(status_code=status_code, content=error.model_dump(exclude_none=True))


@app.exception_handler(UnsupportedDocumentFormatError)
def handle_unsupported_format(request: Request, exc: UnsupportedDocumentFormatError) -> JSONResponse:
    """415: no registered extractor handles the uploaded format."""
    return _error(
        415,
        ErrorResponse(
            code="unsupported_format",
            message=str(exc),
            supported_extensions=list(default_registry.supported_extensions),
        ),
    )


@app.exception_handler(DocumentReadError)
def handle_unreadable_document(request: Request, exc: DocumentReadError) -> JSONResponse:
    """422: the format is supported, but the file is corrupt, truncated or protected."""
    return _error(422, ErrorResponse(code="unreadable_document", message=str(exc)))


@app.exception_handler(NoExtractableTextError)
def handle_no_extractable_text(request: Request, exc: NoExtractableTextError) -> JSONResponse:
    """422: the file was read, but there is no text to index (a scan without OCR)."""
    return _error(422, ErrorResponse(code="no_extractable_text", message=str(exc)))


@app.exception_handler(FileTooLargeError)
def handle_file_too_large(request: Request, exc: FileTooLargeError) -> JSONResponse:
    """413: the upload is past the accepted size, so it was never read."""
    return _error(413, ErrorResponse(code="file_too_large", message=str(exc)))


@app.exception_handler(EmptyQueryError)
def handle_empty_query(request: Request, exc: EmptyQueryError) -> JSONResponse:
    """422: there is no search text to embed."""
    return _error(422, ErrorResponse(code="empty_query", message=str(exc)))


@app.exception_handler(QueryTooLongError)
def handle_query_too_long(request: Request, exc: QueryTooLongError) -> JSONResponse:
    """422: the search text is past the accepted length."""
    return _error(422, ErrorResponse(code="query_too_long", message=str(exc)))
