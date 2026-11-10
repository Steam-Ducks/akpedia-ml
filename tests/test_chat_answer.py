"""Tests of POST /api/v1/chat/answer and its prompt, with a fake model: no Ollama."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest
from fastapi.testclient import TestClient

from app.chat.answer import NOT_FOUND_ANSWER, answer_question
from app.chat.llm import (
    ChatMessage,
    Completion,
    FakeLLMClient,
    LLMClient,
    LLMUnavailableError,
    get_llm_client,
)
from app.chat.prompt import (
    NOT_FOUND_TOKEN,
    SYSTEM_PROMPT,
    TEMPLATE_OVERHEAD_TOKENS,
    Passage,
    cited_passages,
    drop_unfinished_sentence,
    estimate_tokens,
    fit_passages,
    strip_invalid_citations,
)
from app.main import app

PASSAGES = [
    {"documentId": "doc-a", "name": "Manual de Reembolso", "category": "Financeiro", "content": "Prazo: 7 dias."},
    {"documentId": "doc-b", "name": "Política de Viagens", "category": None, "content": "Hotel até R$ 300."},
    {"documentId": "doc-a", "name": "Manual de Reembolso", "category": "Financeiro", "content": "Envie a nota."},
]
QUESTION = "Qual o prazo de reembolso?"


class ScriptedLLM(LLMClient):
    """Answers one text with a chosen ``finish_reason``; FakeLLMClient always says ``stop``."""

    model_name = "scripted"

    def __init__(self, text: str, finish_reason: str = "stop") -> None:
        self._text = text
        self._finish_reason = finish_reason

    def complete(self, messages: Sequence[ChatMessage], **kwargs) -> Completion:
        return Completion(self._text, self.model_name, self._finish_reason, None, None, 0.0)


class UnavailableLLM(LLMClient):
    model_name = "down"

    def complete(self, messages: Sequence[ChatMessage], **kwargs) -> Completion:
        raise LLMUnavailableError("The model server at http://ollama.internal:11434 did not answer in time.")


@pytest.fixture
def use_llm() -> Iterator:
    def install(llm: LLMClient) -> TestClient:
        app.dependency_overrides[get_llm_client] = lambda: llm
        return TestClient(app)

    yield install
    app.dependency_overrides.clear()


def ask(client: TestClient, question: str = QUESTION, passages=PASSAGES):
    return client.post("/api/v1/chat/answer", json={"question": question, "passages": passages})


def passage(content: str, document_id: str = "doc") -> Passage:
    return Passage(document_id=document_id, name="Documento", category=None, content=content)


# --- Endpoint -------------------------------------------------------------------


def test_returns_answer_and_only_cited_documents(use_llm) -> None:
    client = use_llm(FakeLLMClient(["O prazo é de 7 dias [1] e é preciso enviar a nota [3]."]))

    response = ask(client)

    assert response.status_code == 200
    assert response.json() == {
        "answer": "O prazo é de 7 dias [1] e é preciso enviar a nota [3].",
        "found": True,
        "sources": ["doc-a"],
    }


def test_sources_follow_citation_order(use_llm) -> None:
    client = use_llm(FakeLLMClient(["Hotel até R$ 300 [2, 1]."]))

    assert ask(client).json()["sources"] == ["doc-b", "doc-a"]


def test_prompt_numbers_passages_and_carries_question(use_llm) -> None:
    fake = FakeLLMClient(["Resposta [1]."])
    ask(use_llm(fake))

    system, user = fake.calls[0]
    assert system.role == "system" and "português" in system.content and NOT_FOUND_TOKEN in system.content
    assert "[1] Documento: Manual de Reembolso (categoria: Financeiro)\nPrazo: 7 dias." in user.content
    assert "[2] Documento: Política de Viagens\nHotel até R$ 300." in user.content
    assert user.content.endswith(f"Pergunta: {QUESTION}")


def test_not_found_token(use_llm) -> None:
    response = ask(use_llm(FakeLLMClient([NOT_FOUND_TOKEN])))

    assert response.json() == {"answer": NOT_FOUND_ANSWER, "found": False, "sources": []}


def test_answer_without_citation_is_not_found(use_llm) -> None:
    response = ask(use_llm(FakeLLMClient(["O prazo é de 30 dias."])))

    assert response.json()["found"] is False


def test_no_passages_skips_the_model(use_llm) -> None:
    fake = FakeLLMClient()
    response = ask(use_llm(fake), passages=[])

    assert response.json() == {"answer": NOT_FOUND_ANSWER, "found": False, "sources": []}
    assert fake.calls == []


def test_blank_passages_are_ignored(use_llm) -> None:
    fake = FakeLLMClient(["Hotel até R$ 300 [1]."])
    blank = {**PASSAGES[0], "content": "   "}

    response = ask(use_llm(fake), passages=[blank, PASSAGES[1]])

    # The blank one is not numbered, so [1] is the hotel passage.
    assert response.json()["sources"] == ["doc-b"]
    assert "[2]" not in fake.calls[0][1].content


def test_only_blank_passages_skip_the_model(use_llm) -> None:
    fake = FakeLLMClient()
    response = ask(use_llm(fake), passages=[{**PASSAGES[0], "content": ""}])

    assert response.json()["found"] is False
    assert fake.calls == []


def test_made_up_citation_is_dropped(use_llm) -> None:
    response = ask(use_llm(FakeLLMClient(["O prazo é de 7 dias [1][9]."])))

    assert response.json() == {"answer": "O prazo é de 7 dias [1].", "found": True, "sources": ["doc-a"]}


def test_truncated_answer_keeps_complete_sentences(use_llm) -> None:
    llm = ScriptedLLM("O prazo é de 7 dias [1]. Hotel até R$ 300 [2]. Além disso, a no", finish_reason="length")

    response = ask(use_llm(llm))

    assert response.json() == {
        "answer": "O prazo é de 7 dias [1]. Hotel até R$ 300 [2].",
        "found": True,
        "sources": ["doc-a", "doc-b"],
    }


def test_truncated_citation_of_dropped_sentence_is_not_a_source(use_llm) -> None:
    llm = ScriptedLLM("O prazo é de 7 dias [1]. Hotel até R$ 300 [2] e", finish_reason="length")

    assert ask(use_llm(llm)).json()["sources"] == ["doc-a"]


def test_truncated_before_first_sentence_is_not_found(use_llm) -> None:
    llm = ScriptedLLM("O prazo de reembolso [1] depende de", finish_reason="length")

    assert ask(use_llm(llm)).json()["found"] is False


def test_finished_answer_is_not_cut(use_llm) -> None:
    # Without "length", a last sentence without a period is kept.
    response = ask(use_llm(ScriptedLLM("O prazo é de 7 dias [1]")))

    assert response.json()["answer"] == "O prazo é de 7 dias [1]"


def test_blank_question_is_rejected(use_llm) -> None:
    response = ask(use_llm(FakeLLMClient()), question="   ")

    assert response.status_code == 400
    assert response.json() == {"code": "blank_question", "message": "Question cannot be blank."}


def test_provider_timeout_is_503_without_internal_details(use_llm, caplog) -> None:
    response = ask(use_llm(UnavailableLLM()))

    assert response.status_code == 503
    assert response.json()["code"] == "llm_unavailable"
    assert "ollama.internal" not in response.text
    assert "ollama.internal" in caplog.text


def test_provider_error_is_503(use_llm) -> None:
    # FakeLLMClient without answers raises a plain LLMError.
    assert ask(use_llm(FakeLLMClient([]))).status_code == 503


def test_camel_case_contract_in_openapi(use_llm) -> None:
    schemas = use_llm(FakeLLMClient()).get("/openapi.json").json()["components"]["schemas"]

    assert set(schemas["PassageIn"]["properties"]) == {"documentId", "name", "category", "content"}
    assert set(schemas["ChatAnswerResponse"]["properties"]) == {"answer", "found", "sources"}


# --- Context window ---------------------------------------------------------------


def test_passages_that_do_not_fit_are_dropped_from_the_end() -> None:
    fixed = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens("pergunta") + TEMPLATE_OVERHEAD_TOKENS
    passages = [passage("a" * 300, "first"), passage("b" * 300, "second"), passage("c" * 300, "third")]
    each = estimate_tokens("[1] Documento: Documento\n" + "a" * 300) + 1

    # Room for two passages and half of the third.
    kept = fit_passages("pergunta", passages, max_prompt_tokens=fixed + 2 * each + each // 2)

    assert [p.document_id for p in kept] == ["first", "second"]


def test_oversized_first_passage_is_shortened() -> None:
    budget = estimate_tokens(SYSTEM_PROMPT) + 200
    kept = fit_passages("pergunta", [passage("x" * 100_000), passage("y")], max_prompt_tokens=budget)

    assert len(kept) == 1
    assert 0 < len(kept[0].content) < 1000


def test_prompt_never_exceeds_budget_and_trimming_is_logged(caplog) -> None:
    budget = 3000
    fake = FakeLLMClient(["Resposta [1]."])
    passages = [passage("palavra " * 500, f"doc-{n}") for n in range(20)]

    answer_question(fake, "pergunta", passages, max_prompt_tokens=budget)

    prompt_tokens = sum(estimate_tokens(m.content) for m in fake.calls[0])
    assert prompt_tokens <= budget
    assert "chat_passages_trimmed received=20" in caplog.text
    assert "palavra" not in caplog.text


# --- Parsing helpers --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("a [1] b [2, 3] c [1]", [1, 2, 3]), ("[0] [4] [2]", [2]), ("sem citação", [])],
)
def test_cited_passages(text: str, expected: list[int]) -> None:
    assert cited_passages(text, 3) == expected


def test_strip_invalid_citations() -> None:
    assert strip_invalid_citations("A [5]. B [1, 7].", 3) == "A. B [1]."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Primeira [1]. Segunda inco", "Primeira [1]."),
        ("Primeira. [1] Segunda inco", "Primeira. [1]"),
        ("Nenhuma frase completa", ""),
    ],
)
def test_drop_unfinished_sentence(text: str, expected: str) -> None:
    assert drop_unfinished_sentence(text) == expected
