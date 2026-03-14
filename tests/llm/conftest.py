"""Pytest fixtures for LLM tests."""

import os
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from app.llm.config import LLMServiceConfig, MODEL_REGISTRY
from app.llm.models import LLMProvider
from app.llm.providers import AnthropicLLMClient, OpenAILLMClient, VLLMClient
from app.llm.router import ABTestConfig, LLMRouter


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return api_key


@pytest.fixture(scope="session")
def anthropic_api_key() -> str:
    """Get Anthropic API key from environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return api_key


@pytest.fixture(scope="session")
def vllm_endpoint() -> str:
    """Get vLLM endpoint from environment."""
    endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8000")
    return endpoint


@pytest.fixture
def service_config() -> LLMServiceConfig:
    """Create test service configuration."""
    return LLMServiceConfig(
        primary_model="gpt-4o-mini",
        fallback_models=["gpt-3.5-turbo"],
        enable_cost_optimization=True,
        max_cost_per_request=0.10,
        min_quality_tier=3,
        max_retries=2,
        retry_delay_seconds=0.5,
        circuit_breaker_threshold=3,
    )


@pytest_asyncio.fixture
async def openai_client(
    openai_api_key: str,
    service_config: LLMServiceConfig,
) -> AsyncGenerator[OpenAILLMClient, None]:
    """Create OpenAI client for testing."""
    client = OpenAILLMClient(
        model_name="gpt-4o-mini",
        service_config=service_config,
    )
    yield client


@pytest_asyncio.fixture
async def anthropic_client(
    anthropic_api_key: str,
    service_config: LLMServiceConfig,
) -> AsyncGenerator[AnthropicLLMClient, None]:
    """Create Anthropic client for testing."""
    client = AnthropicLLMClient(
        model_name="claude-3-5-haiku-20241022",
        service_config=service_config,
    )
    yield client


@pytest_asyncio.fixture
async def vllm_client(
    vllm_endpoint: str,
    service_config: LLMServiceConfig,
) -> AsyncGenerator[VLLMClient, None]:
    """Create vLLM client for testing."""
    client = VLLMClient(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        endpoint=vllm_endpoint,
        service_config=service_config,
    )
    yield client


@pytest_asyncio.fixture
async def router(
    openai_api_key: str,  # Ensures OPENAI_API_KEY is set; skips if not
    service_config: LLMServiceConfig,
) -> AsyncGenerator[LLMRouter, None]:
    """Create router for testing."""
    router = LLMRouter(service_config=service_config)
    yield router


@pytest_asyncio.fixture
async def ab_test_router(
    openai_api_key: str,  # Ensures OPENAI_API_KEY is set; skips if not
    service_config: LLMServiceConfig,
) -> AsyncGenerator[LLMRouter, None]:
    """Create router with A/B testing for testing."""
    ab_config = ABTestConfig(
        model_a="gpt-4o-mini",
        model_b="gpt-3.5-turbo",
        traffic_split=0.5,
        enabled=True,
    )
    router = LLMRouter(
        service_config=service_config,
        ab_test_config=ab_config,
    )
    yield router


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    from app.llm.models import LLMMessage

    return [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="What is 2+2?"),
    ]


@pytest.fixture
def sample_prompt():
    """Sample prompt for testing."""
    return "Explain quantum computing in one sentence."


@pytest.fixture
def sample_system_prompt():
    """Sample system prompt for testing."""
    return "You are a concise technical expert."

