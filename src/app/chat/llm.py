"""Generate natural-language text with a local language model.

The rest of the application depends only on the :class:`LLMClient` contract: it
receives a list of chat messages and returns a :class:`Completion`. The model never
touches the vector database; whoever calls :meth:`LLMClient.complete` puts the
retrieved chunks inside the messages.

The real implementation talks to Ollama, running inside our own infrastructure, so no
document text leaves the machine. Tests use :class:`FakeLLMClient`.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import httpx

from app import config

logger = logging.getLogger(__name__)

#: Ollama's native chat route. The OpenAI-compatible ``/v1`` route ignores ``num_ctx``.
CHAT_PATH = "/api/chat"

#: How long Ollama keeps the model in memory after a call. Loading it again takes seconds.
KEEP_ALIVE = "30m"

#: Share of the context window above which a prompt may have been truncated by Ollama.
CONTEXT_WARNING_RATIO = 0.95

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """One message of the conversation sent to the model."""

    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    """The model's answer and the numbers needed to tune the prompt size."""

    text: str
    model: str
    #: ``"stop"`` when the model finished; ``"length"`` when ``max_tokens`` cut it.
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_ms: float


class LLMError(Exception):
    """The model could not produce an answer.

    Messages never carry the prompt, the answer or the API key: documents are sensitive.
    """


class LLMUnavailableError(LLMError):
    """The model server could not be reached or did not answer in time."""


class LLMClient(ABC):
    """Turn a conversation into the model's next message."""

    #: Model identifier reported with each completion.
    model_name: str

    @abstractmethod
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Completion:
        """Return the model's answer to ``messages``."""


class OllamaClient(LLMClient):
    """Real implementation calling Ollama's native chat API through httpx."""

    def __init__(
        self,
        base_url: str = config.LLM_BASE_URL,
        model: str = config.LLM_MODEL,
        *,
        api_key: str = config.LLM_API_KEY,
        timeout: float = config.LLM_TIMEOUT_SECONDS,
        context_tokens: int = config.LLM_CONTEXT_TOKENS,
        max_output_tokens: int = config.LLM_MAX_OUTPUT_TOKENS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model_name = model
        self._base_url = base_url
        self._context_tokens = context_tokens
        self._max_output_tokens = max_output_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout, transport=transport)

    def __repr__(self) -> str:
        return f"OllamaClient(base_url={self._base_url!r}, model={self.model_name!r})"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Completion:
        if not messages:
            raise ValueError("messages must not be empty.")

        body = {
            "model": self.model_name,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "temperature": temperature,
                "num_ctx": self._context_tokens,
                "num_predict": max_tokens or self._max_output_tokens,
            },
        }

        started = time.perf_counter()
        try:
            response = self._client.post(CHAT_PATH, json=body)
        except httpx.TimeoutException:
            raise LLMUnavailableError(f"The model server at {self._base_url} did not answer in time.") from None
        except httpx.TransportError:
            raise LLMUnavailableError(f"Could not reach the model server at {self._base_url}.") from None
        duration_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 404:
            raise LLMError(
                f"Model '{self.model_name}' not found on the model server. Pull it first: ollama pull {self.model_name}"
            )
        if response.is_error:
            raise LLMError(f"The model server answered HTTP {response.status_code}.")

        completion = self._parse(response, duration_ms)
        self._log(completion)
        return completion

    def close(self) -> None:
        self._client.close()

    def _parse(self, response: httpx.Response, duration_ms: float) -> Completion:
        try:
            payload: Any = response.json()
            text = payload["message"]["content"]
        except (ValueError, KeyError, TypeError):
            raise LLMError("The model server returned a response without message content.") from None
        if not isinstance(text, str):
            raise LLMError("The model server returned a response without message content.")

        return Completion(
            text=text,
            model=payload.get("model") or self.model_name,
            finish_reason=payload.get("done_reason"),
            prompt_tokens=payload.get("prompt_eval_count"),
            completion_tokens=payload.get("eval_count"),
            duration_ms=duration_ms,
        )

    def _log(self, completion: Completion) -> None:
        # Só números e o nome do modelo: prompt e resposta carregam dados sensíveis.
        logger.info(
            "llm_completion model=%s prompt_tokens=%s completion_tokens=%s finish_reason=%s duration_ms=%.0f",
            completion.model,
            completion.prompt_tokens,
            completion.completion_tokens,
            completion.finish_reason,
            completion.duration_ms,
        )
        if completion.prompt_tokens and completion.prompt_tokens >= self._context_tokens * CONTEXT_WARNING_RATIO:
            logger.warning(
                "llm_prompt_near_context_limit prompt_tokens=%s context_tokens=%s",
                completion.prompt_tokens,
                self._context_tokens,
            )


class FakeLLMClient(LLMClient):
    """Test double: answers from a queue and records every conversation it receives."""

    def __init__(self, answers: Iterable[str] = ("Fake answer.",), model_name: str = "fake-llm") -> None:
        self.model_name = model_name
        self._answers = deque(answers)
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Completion:
        self.calls.append(list(messages))
        if not self._answers:
            raise LLMError("FakeLLMClient has no answers left.")
        return Completion(
            text=self._answers.popleft(),
            model=self.model_name,
            finish_reason="stop",
            prompt_tokens=None,
            completion_tokens=None,
            duration_ms=0.0,
        )


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Return the singleton client, created on the first call."""
    return OllamaClient()
