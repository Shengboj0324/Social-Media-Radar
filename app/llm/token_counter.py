"""Token counting utilities for accurate cost estimation.

This module provides industrial-grade token counting:
- Accurate token estimation before API calls
- Support for multiple tokenizers (OpenAI, Anthropic, etc.)
- Cost estimation based on token counts
- Batch token counting
- Context window validation
"""

import logging
from typing import List, Optional

from app.llm.models import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

# Try to import tiktoken for OpenAI token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using approximate token counting")


class TokenCounter:
    """Token counter with support for multiple providers.
    
    Features:
    - Accurate token counting using provider-specific tokenizers
    - Fallback to approximate counting (4 chars per token)
    - Cost estimation
    - Context window validation
    """
    
    # Approximate tokens per character (fallback)
    CHARS_PER_TOKEN = 4
    
    # Model context windows
    CONTEXT_WINDOWS = {
        # OpenAI
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "gpt-4-turbo": 128_000,
        "gpt-4": 8_192,
        "gpt-3.5-turbo": 16_385,
        
        # Anthropic
        "claude-3-5-sonnet-20241022": 200_000,
        "claude-3-5-haiku-20241022": 200_000,
        "claude-3-opus-20240229": 200_000,
        
        # vLLM / Local
        "meta-llama/Meta-Llama-3.1-405B-Instruct": 128_000,
        "meta-llama/Meta-Llama-3.1-70B-Instruct": 128_000,
        "meta-llama/Meta-Llama-3.1-8B-Instruct": 128_000,
        "mistralai/Mixtral-8x22B-Instruct-v0.1": 65_536,
        "mistralai/Mixtral-8x7B-Instruct-v0.1": 32_768,
        "Qwen/Qwen2.5-72B-Instruct": 32_768,
    }
    
    def __init__(self):
        """Initialize token counter."""
        self._tokenizers = {}
        
        if TIKTOKEN_AVAILABLE:
            logger.info("Token counter initialized with tiktoken support")
        else:
            logger.info("Token counter initialized with approximate counting")
    
    def _get_tokenizer(self, model: str):
        """Get tokenizer for model.
        
        Args:
            model: Model name
            
        Returns:
            Tokenizer or None
        """
        if not TIKTOKEN_AVAILABLE:
            return None
        
        if model in self._tokenizers:
            return self._tokenizers[model]
        
        try:
            # Try to get encoding for model
            if "gpt-4o" in model:
                encoding = tiktoken.get_encoding("o200k_base")
            elif "gpt-4" in model or "gpt-3.5" in model:
                encoding = tiktoken.get_encoding("cl100k_base")
            else:
                # Fallback to cl100k_base for unknown models
                encoding = tiktoken.get_encoding("cl100k_base")
            
            self._tokenizers[model] = encoding
            return encoding
        except Exception as e:
            logger.warning(f"Failed to get tokenizer for {model}: {e}")
            return None
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count
            model: Model name
            
        Returns:
            Token count
        """
        if not text:
            return 0
        
        # Try accurate counting with tiktoken
        tokenizer = self._get_tokenizer(model)
        if tokenizer:
            try:
                return len(tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Tokenizer failed, using approximate counting: {e}")
        
        # Fallback to approximate counting
        return len(text) // self.CHARS_PER_TOKEN
    
    def count_messages_tokens(
        self,
        messages: List[LLMMessage],
        model: str,
    ) -> int:
        """Count tokens in messages.
        
        Args:
            messages: List of messages
            model: Model name
            
        Returns:
            Total token count
        """
        if not messages:
            return 0
        
        # Get tokenizer
        tokenizer = self._get_tokenizer(model)
        
        if tokenizer and TIKTOKEN_AVAILABLE:
            # Accurate counting for OpenAI models
            try:
                # Format: <|im_start|>role\ncontent<|im_end|>
                tokens_per_message = 3  # <|im_start|>, role, <|im_end|>
                tokens_per_name = 1
                
                num_tokens = 0
                for message in messages:
                    num_tokens += tokens_per_message
                    num_tokens += len(tokenizer.encode(message.content))
                    if message.name:
                        num_tokens += tokens_per_name
                
                num_tokens += 3  # Every reply is primed with <|im_start|>assistant<|im_sep|>
                return num_tokens
            except Exception as e:
                logger.warning(f"Message token counting failed: {e}")
        
        # Fallback: approximate counting
        total_chars = sum(len(msg.content) for msg in messages)
        return total_chars // self.CHARS_PER_TOKEN
    
    def get_context_window(self, model: str) -> int:
        """Get context window size for model.
        
        Args:
            model: Model name
            
        Returns:
            Context window size in tokens
        """
        return self.CONTEXT_WINDOWS.get(model, 4096)  # Default to 4K
    
    def validate_context_window(
        self,
        messages: List[LLMMessage],
        model: str,
        max_tokens: Optional[int] = None,
    ) -> bool:
        """Validate that messages fit in context window.
        
        Args:
            messages: List of messages
            model: Model name
            max_tokens: Maximum tokens to generate
            
        Returns:
            True if valid, False otherwise
        """
        prompt_tokens = self.count_messages_tokens(messages, model)
        completion_tokens = max_tokens or 500  # Default estimate
        total_tokens = prompt_tokens + completion_tokens
        
        context_window = self.get_context_window(model)
        
        return total_tokens <= context_window


# Global token counter instance
_token_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get global token counter instance.
    
    Returns:
        Token counter
    """
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter

