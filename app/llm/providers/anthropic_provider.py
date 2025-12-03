"""Enhanced Anthropic provider with industrial-grade reliability.

This module provides comprehensive Anthropic Claude integration:
- Latest models (Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus)
- Comprehensive error handling
- Cost tracking
- Performance metrics
- Streaming support
- 200K context window
"""

import logging
import time
from datetime import datetime
from typing import AsyncIterator, List, Optional

import anthropic
from anthropic import AsyncAnthropic

from app.core.config import settings
from app.llm.base_client import EnhancedBaseLLMClient
from app.llm.config import MODEL_REGISTRY, LLMServiceConfig
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMContextLengthError,
    LLMInvalidRequestError,
    LLMProviderError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMServerError,
    LLMServiceUnavailableError,
)
from app.llm.models import (
    FinishReason,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    PerformanceMetrics,
)

logger = logging.getLogger(__name__)


def map_anthropic_error(error: Exception, provider: str, model: str) -> Exception:
    """Map Anthropic errors to LLM exceptions.

    Args:
        error: Original Anthropic error
        provider: Provider name
        model: Model name

    Returns:
        Mapped LLM exception
    """
    error_str = str(error).lower()

    # Authentication errors
    if isinstance(error, anthropic.AuthenticationError):
        return LLMAuthenticationError(
            f"Anthropic authentication failed: {str(error)}",
            provider=provider,
            model=model,
            original_error=error,
        )

    # Rate limit errors
    if isinstance(error, anthropic.RateLimitError):
        retry_after = None
        if hasattr(error, "response") and error.response:
            retry_after = error.response.headers.get("retry-after")
            if retry_after:
                retry_after = int(retry_after)

        if "quota" in error_str or "billing" in error_str:
            return LLMQuotaExceededError(
                f"Anthropic quota exceeded: {str(error)}",
                provider=provider,
                model=model,
                original_error=error,
            )

        return LLMRateLimitError(
            f"Anthropic rate limit exceeded: {str(error)}",
            retry_after=retry_after,
            provider=provider,
            model=model,
            original_error=error,
        )

    # Invalid request errors
    if isinstance(error, anthropic.BadRequestError):
        if "context" in error_str or "too long" in error_str:
            return LLMContextLengthError(
                f"Anthropic context length exceeded: {str(error)}",
                provider=provider,
                model=model,
                original_error=error,
            )

        if "content" in error_str or "policy" in error_str:
            return LLMContentFilterError(
                f"Anthropic content filtered: {str(error)}",
                provider=provider,
                model=model,
                original_error=error,
            )

        return LLMInvalidRequestError(
            f"Anthropic invalid request: {str(error)}",
            provider=provider,
            model=model,
            original_error=error,
        )

    # Server errors
    if isinstance(error, anthropic.InternalServerError):
        return LLMServerError(
            f"Anthropic server error: {str(error)}",
            provider=provider,
            model=model,
            original_error=error,
        )

    # Service unavailable
    if isinstance(error, anthropic.APIConnectionError):
        return LLMServiceUnavailableError(
            f"Anthropic service unavailable: {str(error)}",
            provider=provider,
            model=model,
            original_error=error,
        )

    # Generic provider error
    return LLMProviderError(
        f"Anthropic error: {str(error)}",
        provider=provider,
        model=model,
        retryable=True,
        original_error=error,
    )


class AnthropicLLMClient(EnhancedBaseLLMClient):
    """Enhanced Anthropic Claude LLM client with full reliability stack.

    Claude is known for:
    - Superior factual accuracy
    - Better instruction following
    - Longer context windows (200K tokens)
    - More nuanced understanding
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        service_config: Optional[LLMServiceConfig] = None,
    ):
        """Initialize Anthropic LLM client.

        Args:
            model_name: Model name (e.g., "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key
            service_config: Service configuration
        """
        # Get model config from registry
        model_config = MODEL_REGISTRY.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown Anthropic model: {model_name}")

        super().__init__(
            provider=LLMProvider.ANTHROPIC,
            model_config=model_config,
            service_config=service_config,
            api_key=api_key,
        )

        # Initialize Anthropic client
        self.client = AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)

    async def _generate_impl(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> LLMResponse:
        """Anthropic-specific generation implementation."""
        start_time = time.time()
        request_time = datetime.utcnow()

        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            system_message = None

            for msg in messages:
                role = msg.role.value if hasattr(msg.role, "value") else msg.role
                if role == "system":
                    system_message = msg.content
                else:
                    anthropic_messages.append({
                        "role": role,
                        "content": msg.content,
                    })

            # Build request kwargs
            request_kwargs = {
                "model": self.model_config.name,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
                **kwargs,
            }

            if system_message:
                request_kwargs["system"] = system_message

            # Call Anthropic API
            response = await self.client.messages.create(**request_kwargs)

            # Calculate metrics
            response_time = datetime.utcnow()
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract content
            content = ""
            if response.content:
                content = response.content[0].text if response.content else ""

            # Map finish reason
            finish_reason_map = {
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
            }
            finish_reason = finish_reason_map.get(response.stop_reason or "end_turn", "stop")

            # Calculate cost
            usage = self.calculate_cost(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )

            # Build performance metrics
            metrics = PerformanceMetrics(
                latency_ms=latency_ms,
                request_time=request_time,
                response_time=response_time,
            )

            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.provider,
                usage=usage,
                metrics=metrics,
                finish_reason=FinishReason(finish_reason),
            )

        except Exception as e:
            raise map_anthropic_error(e, self.provider.value, self.model_config.name)

    async def _generate_stream_impl(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> AsyncIterator[str]:
        """Anthropic-specific streaming implementation."""
        try:
            # Convert messages
            anthropic_messages = []
            system_message = None

            for msg in messages:
                role = msg.role.value if hasattr(msg.role, "value") else msg.role
                if role == "system":
                    system_message = msg.content
                else:
                    anthropic_messages.append({
                        "role": role,
                        "content": msg.content,
                    })

            # Build request kwargs
            request_kwargs = {
                "model": self.model_config.name,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
                "stream": True,
                **kwargs,
            }

            if system_message:
                request_kwargs["system"] = system_message

            # Stream from Anthropic
            async with self.client.messages.stream(**request_kwargs) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            raise map_anthropic_error(e, self.provider.value, self.model_config.name)

