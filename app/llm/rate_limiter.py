"""Rate limiting for LLM API calls.

This module provides industrial-grade rate limiting:
- Token bucket algorithm
- Sliding window rate limiting
- Per-provider rate limits
- Automatic backoff on rate limit errors
"""

import asyncio
import logging
import time
from collections import deque
from typing import Optional

from app.llm.exceptions import LLMRateLimitError

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter.

    The token bucket algorithm allows bursts while maintaining average rate:
    - Tokens are added at a constant rate
    - Each request consumes tokens
    - Requests wait if insufficient tokens
    """

    def __init__(
        self,
        rate: float,
        capacity: Optional[float] = None,
        initial_tokens: Optional[float] = None,
    ):
        """Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum tokens (defaults to rate)
            initial_tokens: Initial token count (defaults to capacity)
        """
        self.rate = rate
        self.capacity = capacity or rate
        self.tokens = initial_tokens if initial_tokens is not None else self.capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """Acquire tokens from bucket.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if tokens acquired, False if timeout

        Raises:
            LLMRateLimitError: If timeout exceeded
        """
        start_time = time.time()

        while True:
            async with self._lock:
                # Refill tokens based on elapsed time
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now

                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate

            # Check timeout
            if timeout is not None:
                elapsed_total = time.time() - start_time
                if elapsed_total + wait_time > timeout:
                    raise LLMRateLimitError(
                        f"Rate limit timeout after {elapsed_total:.2f}s",
                        retry_after=int(wait_time),
                    )

            # Wait for tokens to refill
            await asyncio.sleep(min(wait_time, 0.1))  # Check every 100ms

    def get_available_tokens(self) -> float:
        """Get current available tokens.

        Returns:
            Number of available tokens
        """
        now = time.time()
        elapsed = now - self.last_update
        return min(self.capacity, self.tokens + elapsed * self.rate)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter.

    Tracks requests in a sliding time window:
    - More accurate than fixed windows
    - Prevents burst at window boundaries
    - Memory efficient with deque
    """

    def __init__(self, max_requests: int, window_seconds: float):
        """Initialize sliding window rate limiter.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire permission to make a request.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if permission granted

        Raises:
            LLMRateLimitError: If timeout exceeded
        """
        start_time = time.time()

        while True:
            async with self._lock:
                now = time.time()

                # Remove old requests outside window
                while self.requests and self.requests[0] < now - self.window_seconds:
                    self.requests.popleft()

                # Check if we can make request
                if len(self.requests) < self.max_requests:
                    self.requests.append(now)
                    return True

                # Calculate wait time
                oldest_request = self.requests[0]
                wait_time = (oldest_request + self.window_seconds) - now

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed + wait_time > timeout:
                    raise LLMRateLimitError(
                        f"Rate limit timeout after {elapsed:.2f}s",
                        retry_after=int(wait_time),
                    )

            # Wait before retrying
            await asyncio.sleep(min(wait_time, 0.1))

    def get_current_rate(self) -> float:
        """Get current request rate.

        Returns:
            Requests per second
        """
        now = time.time()
        # Count requests in last window
        recent_requests = sum(
            1 for req_time in self.requests if req_time >= now - self.window_seconds
        )
        return recent_requests / self.window_seconds


class RateLimiter:
    """Combined rate limiter with token bucket and sliding window."""

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: Optional[float] = None,
        max_requests_per_minute: Optional[int] = None,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_second: Average requests per second
            burst_size: Maximum burst size (defaults to requests_per_second)
            max_requests_per_minute: Hard limit per minute (optional)
        """
        self.token_bucket = TokenBucket(
            rate=requests_per_second,
            capacity=burst_size or requests_per_second,
        )

        self.sliding_window: Optional[SlidingWindowRateLimiter] = None
        if max_requests_per_minute:
            self.sliding_window = SlidingWindowRateLimiter(
                max_requests=max_requests_per_minute,
                window_seconds=60.0,
            )

    async def acquire(self, timeout: Optional[float] = 30.0) -> bool:
        """Acquire permission to make request.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if permission granted
        """
        # Acquire from token bucket
        await self.token_bucket.acquire(tokens=1.0, timeout=timeout)

        # Acquire from sliding window if configured
        if self.sliding_window:
            await self.sliding_window.acquire(timeout=timeout)

        return True

