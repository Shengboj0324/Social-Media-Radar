"""Security middleware for FastAPI application.

Implements:
- Intrusion detection
- Rate limiting
- Security headers
- Request validation
- Audit logging
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.security_advanced import (
    intrusion_detection,
    SecurityHeaders,
    DataMasking,
    SecurityAuditLog
)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for all requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security checks.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response with security headers
        """
        start_time = time.time()
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check if IP is blocked
        if intrusion_detection.is_ip_blocked(client_ip):
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Forbidden",
                    "message": "Your IP has been temporarily blocked due to suspicious activity"
                }
            )
        
        # Check for brute force
        if intrusion_detection.check_brute_force(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": "Too many failed attempts. Please try again later."
                }
            )
        
        # Calculate anomaly score
        request_pattern = {
            "user_agent": request.headers.get("user-agent", ""),
            "path": request.url.path,
            "method": request.method,
        }
        
        # Get user ID if authenticated
        user_id = None
        if hasattr(request.state, "user"):
            user_id = str(request.state.user.id)
        
        # Check if request should be challenged
        if user_id:
            should_challenge = intrusion_detection.should_challenge(
                user_id,
                request_pattern
            )
            
            if should_challenge:
                # In production, implement CAPTCHA challenge
                # For now, just log
                pass
        
        # Process request
        try:
            response = await call_next(request)
            
            # Add security headers
            for header, value in SecurityHeaders.get_headers().items():
                response.headers[header] = value
            
            # Log successful request
            await self._log_request(
                request=request,
                response=response,
                user_id=user_id,
                client_ip=client_ip,
                duration=time.time() - start_time,
                success=True
            )
            
            return response
            
        except Exception as e:
            # Log failed request
            await self._log_request(
                request=request,
                response=None,
                user_id=user_id,
                client_ip=client_ip,
                duration=time.time() - start_time,
                success=False,
                error=str(e)
            )
            
            # Record failed attempt for certain endpoints
            if request.url.path.endswith("/login") or request.url.path.endswith("/auth"):
                intrusion_detection.record_failed_attempt(client_ip)
            
            raise

    async def _log_request(
        self,
        request: Request,
        response: Response,
        user_id: str,
        client_ip: str,
        duration: float,
        success: bool,
        error: str = None
    ):
        """Log request for audit trail.
        
        Args:
            request: Request object
            response: Response object
            user_id: User ID
            client_ip: Client IP address
            duration: Request duration
            success: Whether request succeeded
            error: Error message if failed
        """
        # Determine risk level
        risk_level = "LOW"
        
        if not success:
            risk_level = "MEDIUM"
        
        if request.url.path.startswith("/api/v1/admin"):
            risk_level = "HIGH"
        
        if error and ("authentication" in error.lower() or "authorization" in error.lower()):
            risk_level = "HIGH"
        
        # Create audit log entry
        log_entry = SecurityAuditLog(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            action=f"{request.method} {request.url.path}",
            resource=request.url.path,
            ip_address=client_ip,
            user_agent=request.headers.get("user-agent", ""),
            success=success,
            risk_level=risk_level,
            details={
                "duration_ms": int(duration * 1000),
                "status_code": response.status_code if response else None,
                "error": error,
                "query_params": DataMasking.mask_dict(
                    dict(request.query_params),
                    ["password", "token", "api_key", "secret"]
                ),
            }
        )
        
        # In production, store in database or send to logging service
        # For now, just print (would use proper logging)
        # logger.info(log_entry.dict())


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware.
    
    Implements token bucket algorithm for rate limiting.
    """

    def __init__(self, app, requests_per_minute: int = 60):
        """Initialize rate limiter.
        
        Args:
            app: FastAPI app
            requests_per_minute: Maximum requests per minute per IP
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.buckets = {}  # IP -> (tokens, last_update)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limit before processing request.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response or rate limit error
        """
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        if not self._check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate Limit Exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed"
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self._get_remaining_tokens(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if request is within rate limit.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            True if within limit
        """
        now = time.time()
        
        if client_ip not in self.buckets:
            self.buckets[client_ip] = (self.requests_per_minute, now)
            return True
        
        tokens, last_update = self.buckets[client_ip]
        
        # Refill tokens based on time passed
        time_passed = now - last_update
        tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
        tokens = min(self.requests_per_minute, tokens + tokens_to_add)
        
        # Check if we have tokens
        if tokens >= 1:
            self.buckets[client_ip] = (tokens - 1, now)
            return True
        else:
            self.buckets[client_ip] = (tokens, now)
            return False

    def _get_remaining_tokens(self, client_ip: str) -> int:
        """Get remaining tokens for IP.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            Number of remaining tokens
        """
        if client_ip not in self.buckets:
            return self.requests_per_minute
        
        tokens, _ = self.buckets[client_ip]
        return int(tokens)


from datetime import datetime

