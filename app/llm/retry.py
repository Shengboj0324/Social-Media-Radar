"""Retry logic with exponential backoff and jitter.

This module provides industrial-grade retry mechanisms:
- Exponential backoff with jitter
- Configurable retry strategies
- Error-specific retry logic
- Maximum retry limits
"""

import asyncio
import logging
import random
from typing import Callable, Optional, Type, TypeVar

from app.llm.exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_factor: float = 0.1,
    ):
        """Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add jitter to delays
            jitter_factor: Jitter factor (0.0 to 1.0)
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_factor = jitter_factor

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = min(
            self.initial_delay * (self.exponential_base**attempt),
            self.max_delay,
        )

        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_range = delay * self.jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0.0, delay)


# Default retry configuration
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
    jitter_factor=0.1,
)

# Retryable error types
RETRYABLE_ERRORS = (
    LLMRateLimitError,
    LLMServerError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)


async def retry_async(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    retryable_exceptions: tuple = RETRYABLE_ERRORS,
    **kwargs,
) -> T:
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        config: Retry configuration
        retryable_exceptions: Tuple of retryable exception types
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        Exception: Last exception if all retries fail
    """
    config = config or DEFAULT_RETRY_CONFIG
    last_exception: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)

        except retryable_exceptions as e:
            last_exception = e

            # Don't retry on last attempt
            if attempt >= config.max_retries:
                logger.error(
                    f"All {config.max_retries} retries exhausted for {func.__name__}",
                    exc_info=True,
                )
                raise

            # Calculate delay
            delay = config.calculate_delay(attempt)

            # Special handling for rate limit errors
            if isinstance(e, LLMRateLimitError) and e.retry_after:
                delay = max(delay, e.retry_after)

            logger.warning(
                f"Retry attempt {attempt + 1}/{config.max_retries} for {func.__name__} "
                f"after {delay:.2f}s delay. Error: {str(e)}"
            )

            # Wait before retrying
            await asyncio.sleep(delay)

        except Exception as e:
            # Non-retryable error
            logger.error(
                f"Non-retryable error in {func.__name__}: {str(e)}",
                exc_info=True,
            )
            raise

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed unexpectedly")


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable.

    Args:
        error: Exception to check

    Returns:
        True if error is retryable
    """
    if isinstance(error, LLMError):
        return error.retryable
    return isinstance(error, RETRYABLE_ERRORS)

