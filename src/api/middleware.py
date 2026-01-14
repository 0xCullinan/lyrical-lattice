"""
File: src/api/middleware.py
Purpose: Custom middleware for rate limiting, request ID, security headers per REQ-API-*, REQ-SEC-007
"""

import time
import uuid
import hashlib
from collections import defaultdict
from datetime import datetime
from typing import Callable, Awaitable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import settings
from src.utils.logger import get_logger
from src.utils.metrics import metrics

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that adds X-Request-ID header to all responses per REQ-API-012."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Add request ID to request state and response headers.
        
        Args:
            request: Incoming request.
            call_next: Next middleware/handler.
            
        Returns:
            Response with X-Request-ID header.
        """
        # Generate or use existing request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response
        response.headers["X-Request-ID"] = request_id
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware per REQ-API-005, REQ-API-006.
    
    Limits requests to RATE_LIMIT_PER_MINUTE per IP address.
    """
    
    def __init__(self, app, limit_per_minute: int = 100):
        """Initialize rate limiter.
        
        Args:
            app: FastAPI app.
            limit_per_minute: Maximum requests per minute per IP.
        """
        super().__init__(app)
        self.limit_per_minute = limit_per_minute
        self.request_counts: dict[str, list[float]] = defaultdict(list)
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Check rate limit and process or reject request.
        
        Args:
            request: Incoming request.
            call_next: Next middleware/handler.
            
        Returns:
            Response or 429 Too Many Requests.
        """
        # Get client IP (handle proxies)
        client_ip = self._get_client_ip(request)
        
        # Clean old entries and check limit
        now = time.time()
        window_start = now - 60.0  # 1 minute window
        
        # Remove old entries
        self.request_counts[client_ip] = [
            t for t in self.request_counts[client_ip] if t > window_start
        ]
        
        # Check limit
        if len(self.request_counts[client_ip]) >= self.limit_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded. Maximum {self.limit_per_minute} requests per minute.",
                },
                headers={"Retry-After": "60"},
            )
        
        # Record this request
        self.request_counts[client_ip].append(now)
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP, handling proxies.
        
        Args:
            request: Incoming request.
            
        Returns:
            Client IP address.
        """
        # Check X-Forwarded-For header
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # First IP in the list is the client
            return forwarded_for.split(",")[0].strip()
        
        # Fall back to direct client
        if request.client:
            return request.client.host
        
        return "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds security headers per REQ-SEC-007."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Add security headers to response.
        
        Args:
            request: Incoming request.
            call_next: Next middleware/handler.
            
        Returns:
            Response with security headers.
        """
        response = await call_next(request)
        
        # Add security headers per REQ-SEC-007
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Additional security headers
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging per REQ-API-010."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Log request details.
        
        Args:
            request: Incoming request.
            call_next: Next middleware/handler.
            
        Returns:
            Response.
        """
        start_time = time.perf_counter()
        
        # Get request details
        request_id = getattr(request.state, "request_id", "unknown")
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path
        method = request.method
        
        # Hash IP for privacy per RULE-027
        if settings.is_production:
            client_ip = hashlib.sha256(client_ip.encode()).hexdigest()[:16]
        
        # Process request
        response = await call_next(request)
        
        # Calculate latency
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        
        # Log request per REQ-API-010
        logger.info(
            f"{method} {endpoint} - {response.status_code} - {latency_ms}ms",
            extra={
                "request_id": request_id,
                "ip_address": client_ip,
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "response_time_ms": latency_ms,
                "user_agent": request.headers.get("User-Agent", ""),
            },
        )
        
        # Record metrics
        metrics.record_request(method, endpoint, response.status_code, latency_ms / 1000)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for Prometheus metrics collection."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Collect request metrics.
        
        Args:
            request: Incoming request.
            call_next: Next middleware/handler.
            
        Returns:
            Response.
        """
        start_time = time.perf_counter()
        
        response = await call_next(request)
        
        latency = time.perf_counter() - start_time
        metrics.record_request(
            request.method,
            request.url.path,
            response.status_code,
            latency,
        )
        
        return response
