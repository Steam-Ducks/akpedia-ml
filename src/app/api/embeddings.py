"""HTTP endpoints for query embeddings used by document search."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.documents import EmbeddingModelInfo
from app.embeddings import Embedder, get_embedder

router = APIRouter(prefix="/api/v1/embeddings", tags=["embeddings"])


class QueryEmbeddingRequest(BaseModel):
    """Text typed by the user in the search box."""

    text: str = Field(description="Text to embed for semantic search.", examples=["manual tecnico"])


class QueryEmbeddingResponse(BaseModel):
    """Embedding of one search query."""

    model: EmbeddingModelInfo
    embedding: list[float]


@router.post(
    "/query",
    response_model=QueryEmbeddingResponse,
    summary="Embed a search query",
    responses={400: {"description": "Search text is blank."}},
)
def embed_query(
    request: QueryEmbeddingRequest,
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> QueryEmbeddingResponse:
    """Embed one query using the model's query-specific representation."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Search text cannot be blank.")

    vector = embedder.embed_query(request.text)
    return QueryEmbeddingResponse(
        model=EmbeddingModelInfo(name=embedder.model_name, dimensions=embedder.dimensions),
        embedding=vector,
    )
