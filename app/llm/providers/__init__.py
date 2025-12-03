"""LLM provider implementations.

This package contains all LLM provider implementations:
- OpenAI (GPT-4o, GPT-4 Turbo, GPT-4o-mini, GPT-3.5 Turbo)
- Anthropic (Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus)
- vLLM (Llama 3.1, Mixtral, Qwen, any HuggingFace model)
"""

import logging

logger = logging.getLogger(__name__)

# Import providers conditionally based on available dependencies
__all__ = []

# OpenAI provider (requires openai package)
try:
    from app.llm.providers.openai_provider import (
        OpenAIEmbeddingClient,
        OpenAILLMClient,
        OpenAISyncEmbeddingClient,
    )
    __all__.extend(["OpenAILLMClient", "OpenAIEmbeddingClient", "OpenAISyncEmbeddingClient"])
except ImportError as e:
    logger.warning(f"OpenAI provider not available: {e}")
    OpenAILLMClient = None
    OpenAIEmbeddingClient = None
    OpenAISyncEmbeddingClient = None

# Anthropic provider (requires anthropic package)
try:
    from app.llm.providers.anthropic_provider import AnthropicLLMClient
    __all__.append("AnthropicLLMClient")
except ImportError as e:
    logger.warning(f"Anthropic provider not available: {e}")
    AnthropicLLMClient = None

# vLLM provider (requires httpx for API calls)
try:
    from app.llm.providers.vllm_provider import VLLMClient
    __all__.append("VLLMClient")
except ImportError as e:
    logger.warning(f"vLLM provider not available: {e}")
    VLLMClient = None

