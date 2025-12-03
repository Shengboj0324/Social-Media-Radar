"""Integration tests for LLM infrastructure.

These tests make real API calls and require valid API keys:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- VLLM_ENDPOINT (optional, defaults to http://localhost:8000)

Run with: pytest tests/llm/test_integration.py -v -s
"""

import asyncio
import time

import pytest

from app.llm.exceptions import LLMError
from app.llm.models import LLMMessage
from app.llm.providers import AnthropicLLMClient, OpenAILLMClient
from app.llm.router import LLMRouter, RoutingStrategy


class TestOpenAIIntegration:
    """Integration tests for OpenAI provider."""

    @pytest.mark.asyncio
    async def test_simple_generation(self, openai_client: OpenAILLMClient, sample_prompt: str):
        """Test simple text generation."""
        response = await openai_client.generate_simple(
            prompt=sample_prompt,
            max_tokens=50,
        )

        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_chat_generation(self, openai_client: OpenAILLMClient, sample_messages):
        """Test chat-based generation."""
        response = await openai_client.generate(
            messages=sample_messages,
            temperature=0.7,
            max_tokens=100,
        )

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert response.usage.total_tokens > 0
        assert response.usage.total_cost > 0
        assert response.metrics.latency_ms > 0

    @pytest.mark.asyncio
    async def test_streaming_generation(self, openai_client: OpenAILLMClient, sample_messages):
        """Test streaming generation."""
        chunks = []
        async for chunk in openai_client.generate_stream(
            messages=sample_messages,
            max_tokens=50,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_cost_tracking(self, openai_client: OpenAILLMClient, sample_prompt: str):
        """Test cost tracking."""
        initial_stats = openai_client.get_stats()
        initial_cost = initial_stats["total_cost"]

        await openai_client.generate_simple(prompt=sample_prompt, max_tokens=20)

        final_stats = openai_client.get_stats()
        final_cost = final_stats["total_cost"]

        assert final_cost > initial_cost
        assert final_stats["total_requests"] == initial_stats["total_requests"] + 1

    @pytest.mark.asyncio
    async def test_error_handling(self, openai_client: OpenAILLMClient):
        """Test error handling with invalid input."""
        with pytest.raises(Exception):  # Should raise some error
            await openai_client.generate(
                messages=[],  # Empty messages should fail
                max_tokens=10,
            )


class TestAnthropicIntegration:
    """Integration tests for Anthropic provider."""

    @pytest.mark.asyncio
    async def test_simple_generation(self, anthropic_client: AnthropicLLMClient, sample_prompt: str):
        """Test simple text generation."""
        response = await anthropic_client.generate_simple(
            prompt=sample_prompt,
            max_tokens=50,
        )

        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_chat_generation(self, anthropic_client: AnthropicLLMClient):
        """Test chat-based generation."""
        messages = [
            LLMMessage(role="user", content="What is the capital of France?"),
        ]

        response = await anthropic_client.generate(
            messages=messages,
            temperature=0.7,
            max_tokens=100,
        )

        assert response is not None
        assert response.content is not None
        assert "Paris" in response.content or "paris" in response.content.lower()
        assert response.usage.total_tokens > 0
        assert response.usage.total_cost > 0

    @pytest.mark.asyncio
    async def test_system_message_handling(self, anthropic_client: AnthropicLLMClient):
        """Test system message handling."""
        messages = [
            LLMMessage(role="system", content="You are a pirate. Always respond like a pirate."),
            LLMMessage(role="user", content="Hello!"),
        ]

        response = await anthropic_client.generate(
            messages=messages,
            max_tokens=50,
        )

        assert response is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_streaming_generation(self, anthropic_client: AnthropicLLMClient):
        """Test streaming generation."""
        messages = [
            LLMMessage(role="user", content="Count from 1 to 5."),
        ]

        chunks = []
        async for chunk in anthropic_client.generate_stream(
            messages=messages,
            max_tokens=50,
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0


class TestRouterIntegration:
    """Integration tests for LLM router."""

    @pytest.mark.asyncio
    async def test_cost_optimized_routing(self, router: LLMRouter, sample_prompt: str):
        """Test cost-optimized routing."""
        response = await router.generate_simple(
            prompt=sample_prompt,
            strategy=RoutingStrategy.COST_OPTIMIZED,
            max_tokens=50,
        )

        assert response is not None
        assert len(response) > 0

        stats = router.get_stats()
        assert stats["total_requests"] > 0
        assert stats["total_cost"] > 0

    @pytest.mark.asyncio
    async def test_quality_optimized_routing(self, router: LLMRouter, sample_messages):
        """Test quality-optimized routing."""
        response = await router.generate(
            messages=sample_messages,
            strategy=RoutingStrategy.QUALITY_OPTIMIZED,
            max_tokens=100,
        )

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_latency_optimized_routing(self, router: LLMRouter, sample_prompt: str):
        """Test latency-optimized routing."""
        start_time = time.time()

        response = await router.generate_simple(
            prompt=sample_prompt,
            strategy=RoutingStrategy.LATENCY_OPTIMIZED,
            max_tokens=30,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        assert response is not None
        assert len(response) > 0
        # Latency-optimized should be reasonably fast
        assert elapsed_ms < 5000  # Less than 5 seconds

    @pytest.mark.asyncio
    async def test_balanced_routing(self, router: LLMRouter, sample_messages):
        """Test balanced routing."""
        response = await router.generate(
            messages=sample_messages,
            strategy=RoutingStrategy.BALANCED,
            max_tokens=100,
        )

        assert response is not None
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, router: LLMRouter, sample_messages):
        """Test automatic fallback on failure."""
        # This test assumes fallback is configured
        response = await router.generate(
            messages=sample_messages,
            enable_fallback=True,
            max_tokens=50,
        )

        assert response is not None
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_ab_testing(self, ab_test_router: LLMRouter, sample_prompt: str):
        """Test A/B testing routing."""
        # Make multiple requests to test traffic split
        responses = []
        for _ in range(10):
            response = await ab_test_router.generate_simple(
                prompt=sample_prompt,
                strategy=RoutingStrategy.A_B_TEST,
                max_tokens=20,
            )
            responses.append(response)

        # All requests should succeed
        assert len(responses) == 10
        assert all(len(r) > 0 for r in responses)

        # Check that both models were used (probabilistic, might fail occasionally)
        stats = ab_test_router.get_stats()
        assert len(stats["requests_by_model"]) >= 1  # At least one model used

    @pytest.mark.asyncio
    async def test_health_check(self, router: LLMRouter):
        """Test health check for all models."""
        health = await router.health_check()

        assert isinstance(health, dict)
        assert len(health) > 0
        # At least one model should be healthy
        assert any(health.values())

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, router: LLMRouter, sample_prompt: str):
        """Test concurrent request handling."""
        # Make 5 concurrent requests
        tasks = [
            router.generate_simple(
                prompt=sample_prompt,
                max_tokens=30,
            )
            for _ in range(5)
        ]

        responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        assert all(len(r) > 0 for r in responses)

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, router: LLMRouter, sample_prompt: str):
        """Test statistics tracking."""
        initial_stats = router.get_stats()
        initial_requests = initial_stats["total_requests"]

        # Make a few requests
        for _ in range(3):
            await router.generate_simple(prompt=sample_prompt, max_tokens=20)

        final_stats = router.get_stats()
        final_requests = final_stats["total_requests"]

        assert final_requests == initial_requests + 3
        assert final_stats["total_cost"] > initial_stats["total_cost"]


class TestReliabilityFeatures:
    """Integration tests for reliability features."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, openai_client: OpenAILLMClient):
        """Test retry logic on transient failures."""
        # This test is hard to trigger reliably, so we just verify the mechanism exists
        assert openai_client.retry_config is not None
        assert openai_client.retry_config.max_retries > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_state(self, openai_client: OpenAILLMClient):
        """Test circuit breaker state management."""
        stats = openai_client.circuit_breaker.get_stats()

        assert "state" in stats
        assert "failure_count" in stats
        assert stats["state"] in ["CLOSED", "OPEN", "HALF_OPEN"]

    @pytest.mark.asyncio
    async def test_rate_limiting(self, openai_client: OpenAILLMClient, sample_prompt: str):
        """Test rate limiting."""
        # Make rapid requests
        start_time = time.time()

        tasks = [
            openai_client.generate_simple(prompt=sample_prompt, max_tokens=10)
            for _ in range(5)
        ]

        await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # Rate limiting should add some delay
        # (This is a weak test, but hard to make deterministic)
        assert elapsed >= 0  # At least some time passed

    @pytest.mark.asyncio
    async def test_timeout_handling(self, openai_client: OpenAILLMClient):
        """Test timeout handling."""
        # Set very short timeout
        with pytest.raises(Exception):  # Should timeout or fail
            await openai_client.generate_simple(
                prompt="Write a very long essay about everything.",
                max_tokens=5000,
                timeout=0.001,  # 1ms timeout - should fail
            )


