"""
Enhanced error handling module with comprehensive exception management.

This module provides:
- Custom exception hierarchy for different error types
- Graceful error handling and recovery mechanisms
- Detailed error logging and reporting
- Circuit breaker pattern for external services
- Retry mechanisms with exponential backoff
- Error rate limiting and throttling
"""

import asyncio
import functools
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging
from contextlib import asynccontextmanager, contextmanager

from fastapi import HTTPException, status
from starlette.responses import JSONResponse

from .logging import logger, audit


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for different types of failures."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    INTERNAL_SERVER = "internal_server"
    TELEMETRY_PROCESSING = "telemetry_processing"
    MISSION_CRITICAL = "mission_critical"


@dataclass
class ErrorDetails:
    """Detailed error information for logging and debugging."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: Dict[str, Any]
    context: Dict[str, Any]
    traceback: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None


class BaseApplicationError(Exception):
    """Base class for all application-specific errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.INTERNAL_SERVER,
        user_message: Optional[str] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.severity = severity
        self.category = category
        self.user_message = user_message or message
        self.error_code = error_code or self.__class__.__name__
        self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "severity": self.severity.value,
            "category": self.category.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class ValidationError(BaseApplicationError):
    """Error for data validation failures."""

    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)

        super().__init__(
            message=message,
            details=details,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            **kwargs
        )


class AuthenticationError(BaseApplicationError):
    """Error for authentication failures."""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AUTHENTICATION,
            user_message="Authentication required",
            **kwargs
        )


class AuthorizationError(BaseApplicationError):
    """Error for authorization/permission failures."""

    def __init__(self, message: str = "Access denied", resource: str = None, **kwargs):
        details = kwargs.pop("details", {})
        if resource:
            details["resource"] = resource

        super().__init__(
            message=message,
            details=details,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AUTHORIZATION,
            user_message="Access denied",
            **kwargs
        )


class NotFoundError(BaseApplicationError):
    """Error for resource not found."""

    def __init__(self, message: str, resource_type: str = None, resource_id: str = None, **kwargs):
        details = kwargs.pop("details", {})
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        super().__init__(
            message=message,
            details=details,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.NOT_FOUND,
            **kwargs
        )


class ConflictError(BaseApplicationError):
    """Error for resource conflicts (e.g., duplicate resources)."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.CONFLICT,
            **kwargs
        )


class RateLimitError(BaseApplicationError):
    """Error for rate limiting violations."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None, **kwargs):
        details = kwargs.pop("details", {})
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(
            message=message,
            details=details,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.RATE_LIMIT,
            user_message="Too many requests. Please try again later.",
            **kwargs
        )


class ExternalServiceError(BaseApplicationError):
    """Error for external service failures."""

    def __init__(self, message: str, service_name: str = None, status_code: int = None, **kwargs):
        details = kwargs.pop("details", {})
        if service_name:
            details["service_name"] = service_name
        if status_code:
            details["status_code"] = status_code

        super().__init__(
            message=message,
            details=details,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXTERNAL_SERVICE,
            user_message="External service temporarily unavailable",
            **kwargs
        )


class DatabaseError(BaseApplicationError):
    """Error for database-related failures."""

    def __init__(self, message: str, operation: str = None, **kwargs):
        details = kwargs.pop("details", {})
        if operation:
            details["operation"] = operation

        super().__init__(
            message=message,
            details=details,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATABASE,
            user_message="Database operation failed",
            **kwargs
        )


class TelemetryProcessingError(BaseApplicationError):
    """Error for telemetry processing failures."""

    def __init__(self, message: str, vehicle_id: str = None, packet_type: str = None, **kwargs):
        details = kwargs.pop("details", {})
        if vehicle_id:
            details["vehicle_id"] = vehicle_id
        if packet_type:
            details["packet_type"] = packet_type

        super().__init__(
            message=message,
            details=details,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TELEMETRY_PROCESSING,
            **kwargs
        )


class MissionCriticalError(BaseApplicationError):
    """Error for mission-critical system failures."""

    def __init__(self, message: str, mission_id: str = None, **kwargs):
        details = kwargs.pop("details", {})
        if mission_id:
            details["mission_id"] = mission_id

        super().__init__(
            message=message,
            details=details,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.MISSION_CRITICAL,
            **kwargs
        )


class CircuitBreakerError(BaseApplicationError):
    """Error when circuit breaker is open."""

    def __init__(self, message: str = "Circuit breaker is open", service: str = None, **kwargs):
        details = kwargs.pop("details", {})
        if service:
            details["service"] = service

        super().__init__(
            message=message,
            details=details,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXTERNAL_SERVICE,
            user_message="Service temporarily unavailable",
            **kwargs
        )


class CircuitBreaker:
    """Circuit breaker implementation for external service calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise CircuitBreakerError(f"Circuit breaker open for {func.__name__}")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class RetryManager:
    """Retry mechanism with exponential backoff."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions

    def execute(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except self.exceptions as e:
                last_exception = e

                if attempt == self.max_attempts - 1:
                    # Last attempt, don't wait
                    break

                delay = min(
                    self.base_delay * (self.backoff_factor ** attempt),
                    self.max_delay
                )

                logger.warning(
                    f"Attempt {attempt + 1} failed for {func.__name__}, retrying in {delay}s",
                    extra={
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "max_attempts": self.max_attempts,
                        "delay": delay,
                        "error": str(e)
                    }
                )

                time.sleep(delay)

        # All attempts failed
        raise last_exception

    async def execute_async(self, func: Callable, *args, **kwargs):
        """Execute async function with retry logic."""
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)
            except self.exceptions as e:
                last_exception = e

                if attempt == self.max_attempts - 1:
                    break

                delay = min(
                    self.base_delay * (self.backoff_factor ** attempt),
                    self.max_delay
                )

                logger.warning(
                    f"Async attempt {attempt + 1} failed for {func.__name__}, retrying in {delay}s",
                    extra={
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "max_attempts": self.max_attempts,
                        "delay": delay,
                        "error": str(e)
                    }
                )

                await asyncio.sleep(delay)

        raise last_exception


class ErrorHandler:
    """Central error handling and logging."""

    def __init__(self):
        self.error_counts = {}
        self.error_rate_window = 300  # 5 minutes
        self.max_error_rate = 10  # errors per window

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> ErrorDetails:
        """Handle and log error with comprehensive details."""
        error_id = f"err_{int(time.time() * 1000)}_{id(error)}"

        # Determine error details based on type
        if isinstance(error, BaseApplicationError):
            severity = error.severity
            category = error.category
            message = error.message
            details = error.details
        elif isinstance(error, HTTPException):
            severity = self._map_http_status_to_severity(error.status_code)
            category = self._map_http_status_to_category(error.status_code)
            message = str(error.detail)
            details = {"status_code": error.status_code}
        else:
            severity = ErrorSeverity.HIGH
            category = ErrorCategory.INTERNAL_SERVER
            message = str(error)
            details = {"error_type": type(error).__name__}

        # Create error details
        error_details = ErrorDetails(
            error_id=error_id,
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            category=category,
            message=message,
            details=details,
            context=context or {},
            traceback=traceback.format_exc(),
            user_id=user_id,
            request_id=request_id
        )

        # Log error appropriately
        self._log_error(error_details, error)

        # Track error rates
        self._track_error_rate(category)

        # Alert for critical errors
        if severity == ErrorSeverity.CRITICAL:
            self._send_critical_alert(error_details)

        return error_details

    def _map_http_status_to_severity(self, status_code: int) -> ErrorSeverity:
        """Map HTTP status code to error severity."""
        if status_code >= 500:
            return ErrorSeverity.HIGH
        elif status_code >= 400:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _map_http_status_to_category(self, status_code: int) -> ErrorCategory:
        """Map HTTP status code to error category."""
        status_map = {
            400: ErrorCategory.VALIDATION,
            401: ErrorCategory.AUTHENTICATION,
            403: ErrorCategory.AUTHORIZATION,
            404: ErrorCategory.NOT_FOUND,
            409: ErrorCategory.CONFLICT,
            429: ErrorCategory.RATE_LIMIT,
            500: ErrorCategory.INTERNAL_SERVER,
            502: ErrorCategory.EXTERNAL_SERVICE,
            503: ErrorCategory.EXTERNAL_SERVICE,
            504: ErrorCategory.TIMEOUT
        }
        return status_map.get(status_code, ErrorCategory.INTERNAL_SERVER)

    def _log_error(self, error_details: ErrorDetails, original_error: Exception):
        """Log error with appropriate level and context."""
        log_data = {
            "error_id": error_details.error_id,
            "severity": error_details.severity.value,
            "category": error_details.category.value,
            "user_id": error_details.user_id,
            "request_id": error_details.request_id,
            "details": error_details.details,
            "context": error_details.context
        }

        if error_details.severity == ErrorSeverity.CRITICAL:
            logger.critical(error_details.message, extra=log_data)
        elif error_details.severity == ErrorSeverity.HIGH:
            logger.error(error_details.message, extra=log_data)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            logger.warning(error_details.message, extra=log_data)
        else:
            logger.info(error_details.message, extra=log_data)

        # Log to audit trail for security-related errors
        if error_details.category in [ErrorCategory.AUTHENTICATION, ErrorCategory.AUTHORIZATION]:
            audit.log_system_event(
                event_type="security_error",
                severity=error_details.severity.value,
                description=error_details.message,
                details=log_data
            )

    def _track_error_rate(self, category: ErrorCategory):
        """Track error rates and alert on threshold breaches."""
        now = time.time()
        window_start = now - self.error_rate_window

        # Initialize category tracking if needed
        if category not in self.error_counts:
            self.error_counts[category] = []

        # Add current error
        self.error_counts[category].append(now)

        # Remove old errors outside window
        self.error_counts[category] = [
            timestamp for timestamp in self.error_counts[category]
            if timestamp > window_start
        ]

        # Check rate threshold
        error_count = len(self.error_counts[category])
        if error_count >= self.max_error_rate:
            logger.error(
                f"Error rate threshold exceeded for {category.value}: {error_count} errors in {self.error_rate_window}s",
                extra={
                    "event_type": "error_rate_alert",
                    "category": category.value,
                    "error_count": error_count,
                    "window_seconds": self.error_rate_window
                }
            )

    def _send_critical_alert(self, error_details: ErrorDetails):
        """Send alerts for critical errors."""
        # This would integrate with alerting systems like PagerDuty, Slack, etc.
        logger.critical(
            "CRITICAL ERROR ALERT",
            extra={
                "event_type": "critical_alert",
                "error_id": error_details.error_id,
                "message": error_details.message,
                "details": error_details.details,
                "context": error_details.context
            }
        )


def handle_errors(
    reraise: bool = True,
    default_return: Any = None,
    log_level: str = "ERROR",
    include_traceback: bool = True
):
    """Decorator for comprehensive error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = ErrorHandler()
                error_details = error_handler.handle_error(
                    error=e,
                    context={"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
                )

                if reraise:
                    raise
                return default_return

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler = ErrorHandler()
                error_details = error_handler.handle_error(
                    error=e,
                    context={"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
                )

                if reraise:
                    raise
                return default_return

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@contextmanager
def graceful_degradation(
    fallback_value: Any = None,
    log_error: bool = True,
    error_message: str = "Operation failed, using fallback"
):
    """Context manager for graceful degradation when operations fail."""
    try:
        yield
    except Exception as e:
        if log_error:
            logger.warning(f"{error_message}: {str(e)}")
        return fallback_value


# Global error handler instance
error_handler = ErrorHandler()

# Export commonly used items
__all__ = [
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorDetails",
    "BaseApplicationError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError",
    "ExternalServiceError",
    "DatabaseError",
    "TelemetryProcessingError",
    "MissionCriticalError",
    "CircuitBreakerError",
    "CircuitBreaker",
    "RetryManager",
    "ErrorHandler",
    "error_handler",
    "handle_errors",
    "graceful_degradation"
]
