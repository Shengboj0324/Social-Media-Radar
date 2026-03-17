"""Ollama local LLM provider for offline air-gapped deployments.

Implements docs/competitive_analysis.md §5.2 — Offline LLM: OllamaProvider.

Provides zero-API-key inference by calling the Ollama REST API
(https://github.com/ollama/ollama) running locally.  Supports the same
``generate_simple`` / ``generate`` / ``embed_text`` interface as all other
providers and integrates with the existing circuit-breaker and retry stack
defined in ``app/llm/base_client.py``.

No live Ollama instance is required for tests — all HTTP calls are mocked
using ``unittest.mock.AsyncMock``.
"""

import logging
import time
from typing import AsyncIterator, List, Optional

import aiohttp

from app.core.config import settings
from app.llm.base_client import EnhancedBaseLLMClient
from app.llm.config import MODEL_REGISTRY, LLMServiceConfig
from app.llm.exceptions import (
    LLMProviderError,
    LLMServerError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from app.llm.models import (
    EmbeddingResponse,
    FinishReason,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    PerformanceMetrics,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# Default Ollama base URL — overridable via settings.local_llm_url
_DEFAULT_OLLAMA_URL = "http://localhost:11434"


def map_ollama_error(error: Exception, provider: str, model: str) -> Exception:
    """Map raw HTTP / aiohttp errors to typed LLM exceptions.

    Args:
        error: The original exception from the HTTP call.
        provider: Provider name string (``"ollama"``).
        model: Ollama model tag being used (e.g. ``"llama3.1:8b"``).

    Returns:
        A typed exception from ``app.llm.exceptions`` — never raises itself.
    """
    error_str = str(error).lower()
    if isinstance(error, aiohttp.ClientTimeout) or "timeout" in error_str:
        return LLMTimeoutError(
            f"Ollama request timed out: {error}",
            provider=provider,
            model=model,
            original_error=error,
        )
    if isinstance(error, aiohttp.ClientConnectionError) or "connection" in error_str:
        return LLMServiceUnavailableError(
            f"Ollama service unavailable (is it running?): {error}",
            provider=provider,
            model=model,
            original_error=error,
        )
    if "500" in error_str or "internal server" in error_str:
        return LLMServerError(
            f"Ollama server error: {error}",
            provider=provider,
            model=model,
            original_error=error,
        )
    return LLMProviderError(
        f"Ollama error: {error}",
        provider=provider,
        model=model,
        retryable=True,
        original_error=error,
    )


class OllamaProvider(EnhancedBaseLLMClient):
    """Async Ollama provider implementing the EnhancedBaseLLMClient interface.

    Calls:
    - ``POST <base_url>/api/generate``   — text generation (non-streaming)
    - ``POST <base_url>/api/embeddings`` — text embedding

    The provider inherits circuit-breaker, retry, rate-limiter, and Prometheus
    metric recording from ``EnhancedBaseLLMClient`` — no additional reliability
    code is required here.

    Args:
        model_name: Ollama model tag (must be present in ``MODEL_REGISTRY``).
        base_url: Base URL of the Ollama REST API.
        service_config: Optional ``LLMServiceConfig``; uses defaults if None.

    Raises:
        ValueError: If ``model_name`` is not found in ``MODEL_REGISTRY``.
    """

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: Optional[str] = None,
        service_config: Optional[LLMServiceConfig] = None,
    ) -> None:
        """Initialise the Ollama provider.

        Args:
            model_name: Ollama model tag registered in ``MODEL_REGISTRY``.
            base_url: Ollama REST API base URL.  Falls back to
                ``settings.local_llm_url`` then ``_DEFAULT_OLLAMA_URL``.
            service_config: Optional service configuration.

        Raises:
            ValueError: If ``model_name`` is not in ``MODEL_REGISTRY``.
        """
        model_config = MODEL_REGISTRY.get(model_name)
        if model_config is None:
            raise ValueError(
                f"Unknown Ollama model: '{model_name}'. "
                f"Add an entry to MODEL_REGISTRY in app/llm/config.py."
            )

        super().__init__(
            provider=LLMProvider.OLLAMA,
            model_config=model_config,
            service_config=service_config,
            api_key=None,  # No API key required for local models
        )

        self.base_url = (
            base_url
            or settings.local_llm_url
            or _DEFAULT_OLLAMA_URL
        ).rstrip("/")

        logger.info(
            "OllamaProvider initialised: base_url=%s model=%s",
            self.base_url,
            model_name,
        )

    # ------------------------------------------------------------------
    # Abstract method implementation (required by EnhancedBaseLLMClient)
    # ------------------------------------------------------------------

    async def _generate_impl(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> LLMResponse:
        """Call ``POST /api/generate`` and return a structured LLMResponse.

        Ollama's ``/api/generate`` endpoint accepts a ``prompt`` string (not a
        messages array).  We serialise the messages list into a single prompt
        that preserves the system / user / assistant turn structure.

        Args:
            messages: Conversation turns as LLMMessage objects.
            temperature: Sampling temperature in [0.0, 1.0].
            max_tokens: Optional maximum number of tokens to generate.
            **kwargs: Ignored; reserved for future provider-specific params.

        Returns:
            :class:`~app.llm.models.LLMResponse` with content, usage, and
            performance metrics populated.

        Raises:
            LLMTimeoutError: On request timeout.
            LLMServiceUnavailableError: When Ollama is not reachable.
            LLMServerError: On HTTP 5xx responses.
            LLMProviderError: On any other Ollama-side error.
        """
        start_time = time.time()
        prompt = self._messages_to_prompt(messages)

        payload: dict = {
            "model": self.model_config.name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        url = f"{self.base_url}/api/generate"

        try:
            timeout = aiohttp.ClientTimeout(total=self.model_config.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status >= 500:
                        raise LLMServerError(
                            f"Ollama HTTP {resp.status}",
                            provider=LLMProvider.OLLAMA.value,
                            model=self.model_config.name,
                        )
                    resp.raise_for_status()
                    data = await resp.json()

        except (LLMServerError, LLMTimeoutError, LLMServiceUnavailableError):
            raise
        except Exception as exc:
            raise map_ollama_error(exc, LLMProvider.OLLAMA.value, self.model_config.name)

        content: str = data.get("response", "")
        prompt_tokens: int = data.get("prompt_eval_count", 0)
        completion_tokens: int = data.get("eval_count", 0)
        latency_ms = int((time.time() - start_time) * 1000)

        usage = self.calculate_cost(prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=self.model_config.name,
            provider=LLMProvider.OLLAMA,
            finish_reason=FinishReason.STOP,
            usage=usage,
            metrics=PerformanceMetrics(latency_ms=latency_ms),
        )

    # ------------------------------------------------------------------
    # Streaming implementation (required by EnhancedBaseLLMClient)
    # ------------------------------------------------------------------

    async def _generate_stream_impl(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream tokens from ``POST /api/generate`` with ``stream: true``.

        Ollama streams newline-delimited JSON objects.  Each object contains
        a ``"response"`` field with a partial token.  When ``"done"`` is
        ``true``, the stream is complete.

        Args:
            messages: Conversation turns as LLMMessage objects.
            temperature: Sampling temperature.
            max_tokens: Optional maximum tokens to generate.
            **kwargs: Ignored; reserved for future provider-specific params.

        Yields:
            Individual token / partial text chunks as ``str``.

        Raises:
            LLMServiceUnavailableError: When Ollama is not reachable.
            LLMProviderError: On any other error during streaming.
        """
        import json as _json

        prompt = self._messages_to_prompt(messages)
        payload: dict = {
            "model": self.model_config.name,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        url = f"{self.base_url}/api/generate"

        try:
            timeout = aiohttp.ClientTimeout(total=self.model_config.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue
                        try:
                            chunk = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
        except (LLMServiceUnavailableError, LLMProviderError):
            raise
        except Exception as exc:
            raise map_ollama_error(exc, LLMProvider.OLLAMA.value, self.model_config.name)

    # ------------------------------------------------------------------
    # Embedding support
    # ------------------------------------------------------------------

    async def _embed_impl(self, texts: List[str]) -> List["EmbeddingResponse"]:
        """Call ``POST /api/embeddings`` for each text in the batch.

        Ollama's embeddings endpoint processes one text at a time, so this
        method issues sequential requests.  Batching may be added in future if
        Ollama adds native batch support.

        Args:
            texts: List of texts to embed.

        Returns:
            List of :class:`~app.llm.models.EmbeddingResponse` objects in the
            same order as ``texts``.

        Raises:
            LLMServiceUnavailableError: When Ollama is not reachable.
            LLMProviderError: On any other embedding error.
        """
        url = f"{self.base_url}/api/embeddings"
        results: List[EmbeddingResponse] = []

        try:
            timeout = aiohttp.ClientTimeout(total=30.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for text in texts:
                    payload = {"model": self.model_config.name, "prompt": text}
                    async with session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                    embedding = data.get("embedding", [])
                    results.append(
                        EmbeddingResponse(
                            embedding=embedding,
                            model=self.model_config.name,
                            provider=LLMProvider.OLLAMA,
                            usage=TokenUsage(
                                prompt_tokens=len(text.split()),
                                completion_tokens=0,
                                total_tokens=len(text.split()),
                            ),
                        )
                    )
        except (LLMServiceUnavailableError, LLMProviderError):
            raise
        except Exception as exc:
            raise map_ollama_error(exc, LLMProvider.OLLAMA.value, self.model_config.name)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _messages_to_prompt(messages: List[LLMMessage]) -> str:
        """Serialise a list of LLMMessages into a single Ollama prompt string.

        Ollama's ``/api/generate`` endpoint takes a flat ``prompt`` string.
        We prefix each turn with its role so the model retains conversational
        context.

        Args:
            messages: List of LLMMessage objects (system, user, assistant).

        Returns:
            Single string with labelled turns.
        """
        parts = []
        for msg in messages:
            role = msg.role.upper()
            parts.append(f"[{role}]\n{msg.content}")
        return "\n\n".join(parts)
