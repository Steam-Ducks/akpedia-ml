"""Response pieces shared by every endpoint of the service.

Both routes describe the same model and fail in the same shape, so the schemas live
here instead of being written twice: akpedia-server parses one ``model`` block and one
error body, whichever endpoint it called.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbeddingModelInfo(BaseModel):
    """Which model produced the vectors, and in what shape."""

    name: str = Field(description="Model identifier.", examples=["intfloat/multilingual-e5-small"])
    dimensions: int = Field(description="Size of every vector.", examples=[384])
    normalized: bool = Field(default=True, description="Vectors have length 1, so cosine equals dot product.")


class ErrorResponse(BaseModel):
    """Body of every error answered by this service."""

    code: str = Field(description="Stable identifier of the failure.", examples=["unsupported_format"])
    message: str = Field(description="Human readable description.")
    supported_extensions: list[str] | None = Field(
        default=None, description="Formats this service handles, when the failure is about the format."
    )
