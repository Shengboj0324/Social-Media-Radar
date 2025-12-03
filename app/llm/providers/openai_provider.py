"""Enhanced OpenAI provider with industrial-grade reliability.

This module provides comprehensive OpenAI integration:
- Latest models (GPT-4o, GPT-4 Turbo, GPT-4o-mini)
- Comprehensive error handling
- Cost tracking
- Performance metrics
- Streaming support
- Function calling support
"""

import logging
import time
from datetime import datetime
from typing import AsyncIterator, List, Optional

import openai
from openai import AsyncOpenAI, OpenAI

from app.core.config import settings
from app.llm.base_client import EnhancedBaseEmbeddingClient, EnhancedBaseLLMClient
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
    EmbeddingResponse,
    FinishReason,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    PerformanceMetrics,
    TokenUsage,
)

logger = logging.getLogger(__name__)


def map_openai_error(error: Exception, provider: str, model: str) -> Exception:
    """Map OpenAI errors to LLM exceptions.

    Args:
        error: Original OpenAI error
        provider: Provider name
        model: Model name

    Returns:
        Mapped LLM exception
    """
    error_str = str(error).lower()

    # Authentication errors
    if isinstance(error, openai.AuthenticationError):
        return LLMAuthenticationError(
            f"OpenAI authentication failed: {str(error)}",
            provider=provider,
            model=model,
            original_error=error,
        )

    # Rate limit errors
    if isinstance(error, openai.RateLimitError):
        # Extract retry-after if available
        retry_after = None
        if hasattr(error, "response") and error.response:
            retry_after = error.response.headers.get("retry-after")
            if retry_after:
                retry_after = int(retry_after)

        # Check if it's quota exceeded
        if "quota" in error_str or "billing" in error_str:
            return LLMQuotaExceededError(
                f"OpenAI quota exceeded: {str(error)}",
                provider=provider,
                model=model,
                original_error=error,
            )

        return LLMRateLimitError(
            f"OpenAI rate limit exceeded: {str(error)}",
            retry_after=retry_after,
            provider=provider,
            model=model,
            original_error=error,
        )

    # Invalid request errors
    if isinstance(error, openai.BadRequestError):
        # Context length errors
        if "context_length" in error_str or "maximum context" in error_str:
            return LLMContextLengthError(
                f"OpenAI context length exceeded: {str(error)}",
                provider=provider,
                model=model,
                original_error=error,
            )

        # Content filter errors
        if "content_filter" in error_str or "content policy" in error_str:
            return LLMContentFilterError(
                f"OpenAI content filtered: {str(error)}",
                provider=provider,
                model=model,
                original_error=error,
            )

        return LLMInvalidRequestError(
            f"OpenAI invalid request: {str(error)}",
            provider=provider,
            model=model,
            original_error=error,
        )

    # Server errors
    if isinstance(error, openai.InternalServerError):
        return LLMServerError(
            f"OpenAI server error: {str(error)}",
            provider=provider,
            model=model,
            original_error=error,
        )

    # Service unavailable
    if isinstance(error, openai.APIConnectionError):
        return LLMServiceUnavailableError(
            f"OpenAI service unavailable: {str(error)}",
            provider=provider,
            model=model,
            original_error=error,
        )

    # Generic provider error
    return LLMProviderError(
        f"OpenAI error: {str(error)}",
        provider=provider,
        model=model,
        retryable=True,
        original_error=error,
    )


class OpenAILLMClient(EnhancedBaseLLMClient):
    """Enhanced OpenAI LLM client with full reliability stack."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        service_config: Optional[LLMServiceConfig] = None,
    ):
        """Initialize OpenAI LLM client.

        Args:
            model_name: Model name (e.g., "gpt-4o", "gpt-4-turbo-2024-04-09")
            api_key: OpenAI API key
            service_config: Service configuration
        """
        # Get model config from registry
        model_config = MODEL_REGISTRY.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown OpenAI model: {model_name}")

        super().__init__(
            provider=LLMProvider.OPENAI,
            model_config=model_config,
            service_config=service_config,
            api_key=api_key,
        )

        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    async def _generate_impl(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> LLMResponse:
        """OpenAI-specific generation implementation."""
        start_time = time.time()
        request_time = datetime.utcnow()

        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role.value if hasattr(msg.role, "value") else msg.role, "content": msg.content}
                for msg in messages
            ]

            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model_config.name,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Calculate metrics
            response_time = datetime.utcnow()
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response data
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"

            # Calculate cost
            usage = self.calculate_cost(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
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
            raise map_openai_error(e, self.provider.value, self.model_config.name)

    async def _generate_stream_impl(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> AsyncIterator[str]:
        """OpenAI-specific streaming implementation."""
        try:
            # Convert messages
            openai_messages = [
                {"role": msg.role.value if hasattr(msg.role, "value") else msg.role, "content": msg.content}
                for msg in messages
            ]

            # Stream from OpenAI
            stream = await self.client.chat.completions.create(
                model=self.model_config.name,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise map_openai_error(e, self.provider.value, self.model_config.name)


class OpenAIEmbeddingClient(EnhancedBaseEmbeddingClient):
    """Enhanced OpenAI embedding client."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
    ):
        """Initialize OpenAI embedding client.

        Args:
            model_name: Embedding model name
            api_key: OpenAI API key
        """
        super().__init__(
            provider=LLMProvider.OPENAI,
            model_name=model_name,
            api_key=api_key,
        )

        self.client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    async def _embed_impl(self, texts: List[str]) -> List[EmbeddingResponse]:
        """OpenAI-specific embedding implementation."""
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model_name,
            )

            # Calculate cost (approximate)
            total_tokens = response.usage.total_tokens
            cost_per_1m = 0.13 if "large" in self.model_name else 0.02  # text-embedding-3-large vs small
            total_cost = (total_tokens / 1_000_000) * cost_per_1m
            tokens_per_text = total_tokens // len(texts)
            cost_per_text = total_cost / len(texts)

            return [
                EmbeddingResponse(
                    embedding=item.embedding,
                    model=response.model,
                    provider=self.provider,
                    usage=TokenUsage(
                        prompt_tokens=tokens_per_text,
                        completion_tokens=0,
                        total_tokens=tokens_per_text,
                        prompt_cost=cost_per_text,
                        completion_cost=0.0,
                        total_cost=cost_per_text,
                    ),
                )
                for item in response.data
            ]

        except Exception as e:
            raise map_openai_error(e, self.provider.value, self.model_name)


class OpenAISyncEmbeddingClient:
    """Synchronous OpenAI embedding client for Celery tasks."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
    ):
        """Initialize sync OpenAI embedding client.

        Args:
            model_name: Embedding model name
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model_name = model_name

    def embed_text(self, text: str) -> EmbeddingResponse:
        """Embed a single text (synchronous).

        Args:
            text: Text to embed

        Returns:
            Embedding response
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name,
            )

            total_tokens = response.usage.total_tokens
            cost_per_1m = 0.13 if "large" in self.model_name else 0.02
            total_cost = (total_tokens / 1_000_000) * cost_per_1m

            return EmbeddingResponse(
                embedding=response.data[0].embedding,
                model=response.model,
                provider=LLMProvider.OPENAI,
                usage=TokenUsage(
                    prompt_tokens=total_tokens,
                    completion_tokens=0,
                    total_tokens=total_tokens,
                    prompt_cost=total_cost,
                    completion_cost=0.0,
                    total_cost=total_cost,
                ),
            )

        except Exception as e:
            raise map_openai_error(e, LLMProvider.OPENAI.value, self.model_name)

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResponse]:
        """Embed multiple texts (synchronous).

        Args:
            texts: Texts to embed

        Returns:
            List of embedding responses
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name,
            )

            total_tokens = response.usage.total_tokens
            cost_per_1m = 0.13 if "large" in self.model_name else 0.02
            total_cost = (total_tokens / 1_000_000) * cost_per_1m
            tokens_per_text = total_tokens // len(texts)
            cost_per_text = total_cost / len(texts)

            return [
                EmbeddingResponse(
                    embedding=item.embedding,
                    model=response.model,
                    provider=LLMProvider.OPENAI,
                    usage=TokenUsage(
                        prompt_tokens=tokens_per_text,
                        completion_tokens=0,
                        total_tokens=tokens_per_text,
                        prompt_cost=cost_per_text,
                        completion_cost=0.0,
                        total_cost=cost_per_text,
                    ),
                )
                for item in response.data
            ]

        except Exception as e:
            raise map_openai_error(e, LLMProvider.OPENAI.value, self.model_name)

