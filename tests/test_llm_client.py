"""Tests of the language model client, with httpx's mock transport: no network, no Ollama."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

import httpx
import pytest

from app.chat.llm import (
    CHAT_PATH,
    ChatMessage,
    FakeLLMClient,
    LLMError,
    LLMUnavailableError,
    OllamaClient,
)

SECRET = "test-secret-key"
QUESTION = ChatMessage("user", "Qual o prazo de reembolso? [conteudo sensivel]")
MESSAGES = [ChatMessage("system", "Responda com base nos trechos."), QUESTION]

OLLAMA_RESPONSE = {
    "model": "qwen2.5:3b-instruct",
    "message": {"role": "assistant", "content": "O prazo é de 7 dias corridos [1]."},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 1430,
    "eval_count": 18,
}


def make_client(handler, **kwargs) -> OllamaClient:
    kwargs.setdefault("context_tokens", 8192)
    kwargs.setdefault("max_output_tokens", 512)
    return OllamaClient("http://ollama.test", "qwen2.5:3b-instruct", transport=httpx.MockTransport(handler), **kwargs)


def answer_with(payload, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def test_sends_native_chat_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OLLAMA_RESPONSE)

    make_client(handler).complete(MESSAGES, temperature=0.1)

    request = seen[0]
    assert request.method == "POST"
    assert request.url.path == CHAT_PATH
    body = json.loads(request.content)
    assert body["model"] == "qwen2.5:3b-instruct"
    assert body["messages"] == [
        {"role": "system", "content": "Responda com base nos trechos."},
        {"role": "user", "content": QUESTION.content},
    ]
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.1, "num_ctx": 8192, "num_predict": 512}


def test_max_tokens_overrides_default() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=OLLAMA_RESPONSE)

    make_client(handler).complete(MESSAGES, max_tokens=64)

    assert seen[0]["options"]["num_predict"] == 64


def test_parses_completion() -> None:
    completion = make_client(answer_with(OLLAMA_RESPONSE)).complete(MESSAGES)

    assert completion.text == "O prazo é de 7 dias corridos [1]."
    assert completion.model == "qwen2.5:3b-instruct"
    assert completion.finish_reason == "stop"
    assert completion.prompt_tokens == 1430
    assert completion.completion_tokens == 18
    assert completion.duration_ms >= 0


def test_reports_truncated_answer() -> None:
    completion = make_client(answer_with({**OLLAMA_RESPONSE, "done_reason": "length"})).complete(MESSAGES)

    assert completion.finish_reason == "length"


def test_authorization_header_only_with_key() -> None:
    headers: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers)
        return httpx.Response(200, json=OLLAMA_RESPONSE)

    make_client(handler).complete(MESSAGES)
    make_client(handler, api_key=SECRET).complete(MESSAGES)

    assert "authorization" not in headers[0]
    assert headers[1]["authorization"] == f"Bearer {SECRET}"


def test_connection_refused_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(LLMUnavailableError):
        make_client(handler).complete(MESSAGES)


def test_timeout_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(LLMUnavailableError):
        make_client(handler).complete(MESSAGES)


def test_missing_model_tells_how_to_pull() -> None:
    with pytest.raises(LLMError, match="ollama pull qwen2.5:3b-instruct"):
        make_client(answer_with({"error": "model not found"}, status=404)).complete(MESSAGES)


def test_server_error() -> None:
    with pytest.raises(LLMError, match="HTTP 500") as excinfo:
        make_client(answer_with({"error": "boom"}, status=500)).complete(MESSAGES)

    assert not isinstance(excinfo.value, LLMUnavailableError)


@pytest.mark.parametrize("payload", [{}, {"message": {}}, {"message": {"content": None}}, ["not", "a", "dict"]])
def test_malformed_response(payload) -> None:
    with pytest.raises(LLMError, match="without message content"):
        make_client(answer_with(payload)).complete(MESSAGES)


def test_non_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>proxy error</html>")

    with pytest.raises(LLMError, match="without message content"):
        make_client(handler).complete(MESSAGES)


def test_empty_messages_rejected() -> None:
    with pytest.raises(ValueError):
        make_client(answer_with(OLLAMA_RESPONSE)).complete([])


def test_secrets_and_content_stay_out_of_repr_errors_and_logs(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    client = make_client(answer_with({"error": "boom"}, status=500), api_key=SECRET)

    with pytest.raises(LLMError) as excinfo:
        client.complete(MESSAGES)
    make_client(answer_with(OLLAMA_RESPONSE), api_key=SECRET).complete(MESSAGES)

    for text in (repr(client), str(excinfo.value), caplog.text):
        assert SECRET not in text
        assert "conteudo sensivel" not in text
        assert "7 dias" not in text


def test_warns_when_prompt_fills_context(caplog) -> None:
    caplog.set_level(logging.WARNING)
    make_client(answer_with({**OLLAMA_RESPONSE, "prompt_eval_count": 8000})).complete(MESSAGES)

    assert "llm_prompt_near_context_limit" in caplog.text


def test_fake_answers_in_order_and_records_calls() -> None:
    fake = FakeLLMClient(["primeira", "segunda"])

    assert fake.complete(MESSAGES).text == "primeira"
    assert fake.complete([QUESTION]).text == "segunda"
    assert fake.calls == [MESSAGES, [QUESTION]]
    with pytest.raises(LLMError):
        fake.complete(MESSAGES)


def test_invalid_timeout_fails_at_startup() -> None:
    # Processo separado: config é lido no import e não deve vazar para os outros testes.
    env = {**os.environ, "LLM_TIMEOUT_SECONDS": "abc"}
    result = subprocess.run(
        [sys.executable, "-c", "import app.config"], env=env, capture_output=True, text=True, check=False
    )

    assert result.returncode != 0
    assert "LLM_TIMEOUT_SECONDS must be a number" in result.stderr
