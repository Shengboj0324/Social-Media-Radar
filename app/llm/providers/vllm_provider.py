"""vLLM provider for local/self-hosted models.

This module provides integration with vLLM for running open-source models:
- Llama 3.1 405B, 70B, 8B
- Mixtral 8x22B, 8x7B
- Qwen 2.5 72B
- Any HuggingFace model
- Cost-effective alternative to commercial APIs
- Full control over infrastructure
"""

import logging
import time
from datetime import datetime
from typing import AsyncIterator, List, Optional

import aiohttp

from app.core.config import settings
from app.llm.base_client import EnhancedBaseLLMClient
from app.llm.config import MODEL_REGISTRY, LLMServiceConfig
from app.llm.exceptions import (
    LLMInvalidRequestError,
    LLMProviderError,
    LLMServerError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from app.llm.models import (
    FinishReason,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    PerformanceMetrics,
)

logger = logging.getLogger(__name__)


def map_vllm_error(error: Exception, provider: str, model: str) -> Exception:
    """Map vLLM errors to LLM exceptions.

    Args:
        error: Original error
        provider: Provider name
        model: Model name

    Returns:
        Mapped LLM exception
    """
    error_str = str(error).lower()

    # Timeout errors
    if isinstance(error, aiohttp.ClientTimeout) or "timeout" in error_str:
        return LLMTimeoutError(
            f"vLLM request timeout: {str(error)}",
            provider=provider,
            model=model,
            original_error=error,
        )

    # Connection errors
    if isinstance(error, aiohttp.ClientConnectionError) or "connection" in error_str:
        return LLMServiceUnavailableError(
            f"vLLM service unavailable: {str(error)}",
            provider=provider,
            model=model,
            original_error=error,
        )

    # Server errors
    if "500" in error_str or "internal server" in error_str:
        return LLMServerError(
            f"vLLM server error: {str(error)}",
            provider=provider,
            model=model,
            original_error=error,
        )

    # Invalid request
    if "400" in error_str or "bad request" in error_str:
        return LLMInvalidRequestError(
            f"vLLM invalid request: {str(error)}",
            provider=provider,
            model=model,
            original_error=error,
        )

    # Generic error
    return LLMProviderError(
        f"vLLM error: {str(error)}",
        provider=provider,
        model=model,
        retryable=True,
        original_error=error,
    )


class VLLMClient(EnhancedBaseLLMClient):
    """vLLM client for local/self-hosted models.

    vLLM provides:
    - High-throughput inference
    - PagedAttention for efficient memory
    - Continuous batching
    - OpenAI-compatible API
    - Support for all major open-source models
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-405B-Instruct",
        endpoint: Optional[str] = None,
        service_config: Optional[LLMServiceConfig] = None,
    ):
        """Initialize vLLM client.

        Args:
            model_name: Model name (HuggingFace format)
            endpoint: vLLM endpoint URL
            service_config: Service configuration
        """
        # Get model config from registry
        model_config = MODEL_REGISTRY.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown vLLM model: {model_name}")

        super().__init__(
            provider=LLMProvider.VLLM,
            model_config=model_config,
            service_config=service_config,
            api_key=None,  # No API key for local models
        )

        # vLLM endpoint
        self.endpoint = endpoint or settings.vllm_endpoint or "http://localhost:8000"
        if not self.endpoint.endswith("/v1"):
            self.endpoint = f"{self.endpoint}/v1"

        logger.info(f"Initialized vLLM client: endpoint={self.endpoint}, model={model_name}")

    async def _generate_impl(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> LLMResponse:
        """vLLM-specific generation implementation."""
        start_time = time.time()
        request_time = datetime.utcnow()

        try:
            # Convert messages to OpenAI format (vLLM is OpenAI-compatible)
            formatted_messages = [
                {
                    "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                    "content": msg.content,
                }
                for msg in messages
            ]

            # Build request payload
            payload = {
                "model": self.model_config.name,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 2048,
                **kwargs,
            }

            # Make request to vLLM
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.endpoint}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.model_config.timeout_seconds),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMProviderError(
                            f"vLLM returned status {response.status}: {error_text}",
                            provider=self.provider.value,
                            model=self.model_config.name,
                        )

                    result = await response.json()

            # Calculate metrics
            response_time = datetime.utcnow()
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response data
            choice = result["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")

            # Extract usage
            usage_data = result.get("usage", {})
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)

            # Calculate cost (infrastructure cost, not API cost)
            usage = self.calculate_cost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            # Build performance metrics
            metrics = PerformanceMetrics(
                latency_ms=latency_ms,
                request_time=request_time,
                response_time=response_time,
            )

            return LLMResponse(
                content=content,
                model=self.model_config.name,
                provider=self.provider,
                usage=usage,
                metrics=metrics,
                finish_reason=FinishReason(finish_reason),
            )

        except Exception as e:
            raise map_vllm_error(e, self.provider.value, self.model_config.name)

    async def _generate_stream_impl(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> AsyncIterator[str]:
        """vLLM-specific streaming implementation."""
        try:
            # Convert messages
            formatted_messages = [
                {
                    "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                    "content": msg.content,
                }
                for msg in messages
            ]

            # Build request payload
            payload = {
                "model": self.model_config.name,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 2048,
                "stream": True,
                **kwargs,
            }

            # Stream from vLLM
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.endpoint}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.model_config.timeout_seconds),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMProviderError(
                            f"vLLM returned status {response.status}: {error_text}",
                            provider=self.provider.value,
                            model=self.model_config.name,
                        )

                    # Parse SSE stream
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if not line or line == "data: [DONE]":
                            continue

                        if line.startswith("data: "):
                            try:
                                import json

                                data = json.loads(line[6:])
                                if "choices" in data and data["choices"]:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            raise map_vllm_error(e, self.provider.value, self.model_config.name)

