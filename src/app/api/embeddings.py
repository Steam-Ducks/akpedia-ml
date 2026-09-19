"""HTTP endpoint that turns a search text into its numeric representation.

The counterpart of ``POST /api/v1/documents/process``: that one vectorizes what goes
into the index, this one vectorizes what the user typed. Both answer the same
``model`` block and vectors of the same size, produced by the same instance of the
model -- that is what lets akpedia-server compare the two by cosine similarity.

The service is stateless: the search text is embedded and forgotten. Matching the
vector against the index is akpedia-server's job, since only it talks to the database.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.api.schemas import EmbeddingModelInfo, ErrorResponse
from app.config import MAX_QUERY_CHARS
from app.embeddings import Embedder, get_embedder

router = APIRouter(prefix="/api/v1/embeddings", tags=["embeddings"])


class EmptyQueryError(Exception):
    """The search text is empty, or holds nothing but whitespace."""


class QueryTooLongError(Exception):
    """The search text is past :data:`~app.config.MAX_QUERY_CHARS`."""


class QueryRequest(BaseModel):
    """The text the user typed in the search box."""

    text: str = Field(description="Search text.", examples=["qual é o prazo de garantia do equipamento?"])


class QueryEmbeddingResponse(BaseModel):
    """The search vector, plus the model that produced it.

    The ``model`` block is the same one ``POST /api/v1/documents/process`` answers:
    akpedia-server can check that the index it is about to search was built by this
    very model before trusting the distances.
    """

    # ``model`` collides with Pydantic's reserved ``model_`` namespace.
    model_config = ConfigDict(protected_namespaces=())

    model: EmbeddingModelInfo
    embedding: list[float] = Field(description="Normalized vector of the search text.", examples=[[0.012, -0.045]])


@router.post(
    "/query",
    response_model=QueryEmbeddingResponse,
    summary="Embed a search text",
    responses={422: {"model": ErrorResponse, "description": "The search text is empty or too long."}},
)
def embed_query(
    request: QueryRequest,
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> QueryEmbeddingResponse:
    """Return the vector of the search text, comparable to the document chunks'."""
    text = _normalize(request.text)
    if not text:
        raise EmptyQueryError("The search text is empty.")
    if len(text) > MAX_QUERY_CHARS:
        raise QueryTooLongError(
            f"The search text has {len(text)} characters, past the {MAX_QUERY_CHARS} character limit."
        )

    return QueryEmbeddingResponse(
        model=EmbeddingModelInfo(name=embedder.model_name, dimensions=embedder.dimensions),
        embedding=embedder.embed_query(text),
    )


def _normalize(text: str) -> str:
    """Collapse every run of whitespace into a single space, and trim the ends.

    The same normalization the chunks went through in ``split_sentences``: a query and
    a chunk that differ only in line breaks should not land on different vectors.
    """
    return " ".join(text.split())
