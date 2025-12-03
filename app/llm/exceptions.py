"""LLM-specific exceptions with comprehensive error classification.

This module provides industrial-grade error handling for LLM operations:
- Detailed error classification
- Retry-ability detection
- Cost tracking on errors
- Provider-specific error mapping
"""

from typing import Optional


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        retryable: bool = False,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        """Initialize LLM error.

        Args:
            message: Error message
            provider: LLM provider name
            model: Model name
            retryable: Whether error is retryable
            status_code: HTTP status code if applicable
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.retryable = retryable
        self.status_code = status_code
        self.original_error = original_error

    def __str__(self) -> str:
        """String representation."""
        parts = [self.message]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.model:
            parts.append(f"model={self.model}")
        if self.status_code:
            parts.append(f"status={self.status_code}")
        return " | ".join(parts)


class LLMAuthenticationError(LLMError):
    """Authentication failed - invalid API key or credentials."""

    def __init__(self, message: str, **kwargs):
        """Initialize authentication error."""
        super().__init__(message, retryable=False, status_code=401, **kwargs)


class LLMRateLimitError(LLMError):
    """Rate limit exceeded - too many requests."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        """Initialize rate limit error.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            **kwargs: Additional error parameters
        """
        super().__init__(message, retryable=True, status_code=429, **kwargs)
        self.retry_after = retry_after


class LLMQuotaExceededError(LLMError):
    """Quota exceeded - billing limit reached."""

    def __init__(self, message: str, **kwargs):
        """Initialize quota exceeded error."""
        super().__init__(message, retryable=False, status_code=429, **kwargs)


class LLMInvalidRequestError(LLMError):
    """Invalid request - bad parameters or malformed input."""

    def __init__(self, message: str, **kwargs):
        """Initialize invalid request error."""
        super().__init__(message, retryable=False, status_code=400, **kwargs)


class LLMContextLengthError(LLMError):
    """Context length exceeded - input too long."""

    def __init__(
        self,
        message: str,
        max_tokens: Optional[int] = None,
        actual_tokens: Optional[int] = None,
        **kwargs,
    ):
        """Initialize context length error.

        Args:
            message: Error message
            max_tokens: Maximum allowed tokens
            actual_tokens: Actual token count
            **kwargs: Additional error parameters
        """
        super().__init__(message, retryable=False, status_code=400, **kwargs)
        self.max_tokens = max_tokens
        self.actual_tokens = actual_tokens


class LLMContentFilterError(LLMError):
    """Content filtered - violated content policy."""

    def __init__(self, message: str, **kwargs):
        """Initialize content filter error."""
        super().__init__(message, retryable=False, status_code=400, **kwargs)


class LLMTimeoutError(LLMError):
    """Request timeout - took too long to complete."""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        """Initialize timeout error.

        Args:
            message: Error message
            timeout_seconds: Timeout duration
            **kwargs: Additional error parameters
        """
        super().__init__(message, retryable=True, status_code=408, **kwargs)
        self.timeout_seconds = timeout_seconds


class LLMServerError(LLMError):
    """Server error - provider-side issue."""

    def __init__(self, message: str, **kwargs):
        """Initialize server error."""
        super().__init__(message, retryable=True, status_code=500, **kwargs)


class LLMServiceUnavailableError(LLMError):
    """Service unavailable - provider is down."""

    def __init__(self, message: str, **kwargs):
        """Initialize service unavailable error."""
        super().__init__(message, retryable=True, status_code=503, **kwargs)


class LLMCircuitBreakerError(LLMError):
    """Circuit breaker open - too many failures."""

    def __init__(
        self,
        message: str,
        failure_count: int,
        threshold: int,
        **kwargs,
    ):
        """Initialize circuit breaker error.

        Args:
            message: Error message
            failure_count: Number of failures
            threshold: Failure threshold
            **kwargs: Additional error parameters
        """
        super().__init__(message, retryable=False, **kwargs)
        self.failure_count = failure_count
        self.threshold = threshold


class LLMModelNotFoundError(LLMError):
    """Model not found - invalid model name."""

    def __init__(self, message: str, **kwargs):
        """Initialize model not found error."""
        super().__init__(message, retryable=False, status_code=404, **kwargs)


class LLMProviderError(LLMError):
    """Generic provider error - catch-all for provider-specific issues."""

    pass

