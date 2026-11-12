"""HTTP endpoint that answers a chat question from retrieved passages.

akpedia-server retrieves the passages, applies permissions and sends them here; this
service only writes the answer.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from app.api.documents import ErrorResponse
from app.chat.answer import answer_question
from app.chat.llm import LLMClient, get_llm_client
from app.chat.prompt import Passage

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class _CamelModel(BaseModel):
    # akpedia-server speaks camelCase (``documentId``).
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class PassageIn(_CamelModel):
    """One passage retrieved for the question, already filtered by permission."""

    document_id: str = Field(description="Id of the document the passage belongs to.", examples=["42"])
    name: str = Field(description="Document name.", examples=["Manual de Reembolso"])
    category: str | None = Field(default=None, description="Document category.", examples=["Financeiro"])
    content: str = Field(
        description="Passage text. Blank passages are ignored.",
        examples=["O reembolso é feito em até 7 dias corridos."],
    )


class ChatAnswerRequest(_CamelModel):
    question: str = Field(description="Question typed by the user.", examples=["Qual o prazo de reembolso?"])
    passages: list[PassageIn] = Field(
        description="Passages to answer from, most relevant first: the last ones are dropped when they do not fit."
    )


class ChatAnswerResponse(_CamelModel):
    answer: str = Field(
        description="Short answer in Portuguese, each statement tagged with the passage number it came from.",
        examples=["O reembolso é feito em até 7 dias corridos [1]."],
    )
    found: bool = Field(description="False when the passages do not answer the question.")
    sources: list[str] = Field(description="`documentId` of each document the answer cites, without repeats.")


@router.post(
    "/answer",
    response_model=ChatAnswerResponse,
    summary="Answer a question from retrieved passages",
    responses={
        400: {"model": ErrorResponse, "description": "Question is blank."},
        503: {"model": ErrorResponse, "description": "The language model failed or did not answer in time."},
    },
)
def answer(
    request: ChatAnswerRequest,
    llm: Annotated[LLMClient, Depends(get_llm_client)],
) -> ChatAnswerResponse | JSONResponse:
    """Answer only from the passages and say which documents were cited.

    With no passages, or passages that do not answer, ``found`` is ``false`` and
    ``sources`` is empty.
    """
    if not request.question.strip():
        error = ErrorResponse(code="blank_question", message="Question cannot be blank.")
        return JSONResponse(status_code=400, content=error.model_dump(exclude_none=True))

    passages = [
        Passage(document_id=p.document_id, name=p.name, category=p.category, content=p.content)
        for p in request.passages
    ]
    result = answer_question(llm, request.question, passages)
    return ChatAnswerResponse(answer=result.text, found=result.found, sources=result.sources)
