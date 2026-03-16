"""Anthropic Claude implementation of LLM client."""

from typing import List, Optional

import anthropic

from app.core.config import settings
from app.llm.client_base import BaseLLMClient, LLMMessage, LLMResponse


class AnthropicLLMClient(BaseLLMClient):
    """Anthropic Claude LLM client.
    
    Claude is known for:
    - Superior factual accuracy
    - Better instruction following
    - Longer context windows (200K tokens)
    - More nuanced understanding
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key (defaults to settings)
            model: Model name (defaults to claude-3-sonnet)
        """
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or getattr(settings, "anthropic_api_key", None)
        )
        self.model = model or "claude-3-sonnet-20240229"
    
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate completion from messages.
        
        Args:
            messages: List of messages
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response
        """
        # Convert messages to Anthropic format
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        
        # Call Anthropic API
        kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        response = await self.client.messages.create(**kwargs)
        
        # Extract content
        content = ""
        if response.content:
            content = response.content[0].text if response.content else ""
        
        return LLMResponse(
            content=content,
            model=response.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason or "stop",
        )
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Generate completion with streaming.
        
        Args:
            messages: List of messages
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Content chunks
        """
        # Convert messages
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        
        # Call Anthropic API with streaming.
        # NOTE: messages.stream() is a context-manager — do NOT pass stream=True as kwarg.
        stream_kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }

        if system_message:
            stream_kwargs["system"] = system_message

        async with self.client.messages.stream(**stream_kwargs) as stream:
            async for text in stream.text_stream:
                yield text

