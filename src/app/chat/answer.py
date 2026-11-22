"""Answer a question from passages retrieved by akpedia-server.

This service neither reads the database nor checks permissions: the passages arrive
ready and already filtered, and the answer only says which of them it used.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app import config
from app.chat.llm import LLMClient
from app.chat.prompt import (
    Passage,
    build_messages,
    cited_passages,
    drop_unfinished_sentence,
    fit_passages,
    is_not_found,
    strip_invalid_citations,
)

logger = logging.getLogger(__name__)

#: What the chatbot says when the passages do not answer the question.
NOT_FOUND_ANSWER = "Não encontrei essa informação nos documentos disponíveis."

#: Low temperature: the answer should stick to the passages, not be creative.
ANSWER_TEMPERATURE = 0.1


@dataclass(frozen=True)
class Answer:
    text: str
    found: bool
    #: ``document_id`` of each cited document, in order of first citation, without repeats.
    sources: list[str]


def answer_question(
    llm: LLMClient,
    question: str,
    passages: Sequence[Passage],
    *,
    max_prompt_tokens: int = config.LLM_CONTEXT_TOKENS - config.LLM_MAX_OUTPUT_TOKENS,
) -> Answer:
    """Ask the model and keep only the documents its answer actually cites.

    Blank passages are ignored, and the least relevant ones are dropped when they do not
    fit in ``max_prompt_tokens``. An answer without any valid citation counts as not
    found: nothing ties it to the passages, so it may have come from the model's own
    knowledge.

    Raises :class:`app.chat.llm.LLMError` when the model fails or does not answer in time.
    """
    usable = [passage for passage in passages if passage.content.strip()]
    fitted = fit_passages(question, usable, max_prompt_tokens)
    if len(fitted) < len(usable):
        # Só contagens: o conteúdo dos trechos é sensível.
        logger.warning("chat_passages_trimmed received=%d kept=%d", len(usable), len(fitted))
    if not fitted:
        return _not_found()

    completion = llm.complete(build_messages(question, fitted), temperature=ANSWER_TEMPERATURE)
    text = completion.text.strip()
    if completion.finish_reason == "length":
        text = drop_unfinished_sentence(text)
    if not text or is_not_found(text):
        return _not_found()

    cited = cited_passages(text, len(fitted))
    if not cited:
        logger.info("chat_answer_without_citation")
        return _not_found()

    sources: list[str] = []
    for number in cited:
        document_id = fitted[number - 1].document_id
        if document_id not in sources:
            sources.append(document_id)

    return Answer(text=strip_invalid_citations(text, len(fitted)), found=True, sources=sources)


def _not_found() -> Answer:
    return Answer(text=NOT_FOUND_ANSWER, found=False, sources=[])
