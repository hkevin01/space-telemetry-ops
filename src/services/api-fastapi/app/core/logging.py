"""
Enhanced logging utilities with structured logging, audit trails, and performance monitoring.

This module provides comprehensive logging capabilities including:
- Structured JSON logging
- Performance metrics and timing
- Audit trail logging for security events
- Context-aware logging with request tracing
- Error tracking and alerting
- Memory and resource monitoring
"""

import json
import time
import functools
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
import logging.config
import sys
import os
import psutil
from pathlib import Path

from loguru import logger


@dataclass
class LogContext:
    """Context information for enhanced logging."""
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    mission_id: Optional[str] = None
    spacecraft_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    start_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceTimer:
    """High-precision performance timer for measuring execution time."""

    def __init__(self, name: str = "operation", logger_instance: Any = None):
        self.name = name
        self.logger = logger_instance or logger
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_ns: Optional[int] = None
        self.duration_ms: Optional[float] = None
        self.duration_s: Optional[float] = None

    def __enter__(self):
        """Start timing when entering context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log results when exiting context manager."""
        self.stop()
        self.log_performance()

    def start(self) -> None:
        """Start the performance timer."""
        self.start_time = time.perf_counter_ns()

    def stop(self) -> None:
        """Stop the performance timer and calculate durations."""
        if self.start_time is None:
            raise ValueError("Timer not started")

        self.end_time = time.perf_counter_ns()
        self.duration_ns = self.end_time - self.start_time
        self.duration_ms = self.duration_ns / 1_000_000  # nanoseconds to milliseconds
        self.duration_s = self.duration_ns / 1_000_000_000  # nanoseconds to seconds

    def get_duration(self, unit: str = "ms") -> float:
        """
        Get duration in specified unit.

        Args:
            unit: Time unit - 'ns', 'μs', 'ms', 's'

        Returns:
            Duration in specified unit
        """
        if self.duration_ns is None:
            raise ValueError("Timer not stopped")

        if unit == "ns":
            return float(self.duration_ns)
        elif unit == "μs" or unit == "us":
            return self.duration_ns / 1_000
        elif unit == "ms":
            return self.duration_ms
        elif unit == "s":
            return self.duration_s
        else:
            raise ValueError(f"Invalid unit: {unit}. Use 'ns', 'μs', 'ms', or 's'")

    def log_performance(self) -> None:
        """Log performance metrics."""
        if self.duration_ms is None:
            return

        self.logger.info(
            f"Performance: {self.name} completed",
            extra={
                "event_type": "performance",
                "operation": self.name,
                "duration_ns": self.duration_ns,
                "duration_ms": self.duration_ms,
                "duration_s": self.duration_s,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


class AuditLogger:
    """Specialized logger for security and audit events."""

    def __init__(self):
        self.logger = logger.bind(component="audit")

    def log_authentication(self, user_id: str, success: bool, ip_address: str = None,
                          user_agent: str = None, details: Dict[str, Any] = None) -> None:
        """Log authentication events."""
        event_data = {
            "event_type": "authentication",
            "user_id": user_id,
            "success": success,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        }

        level = "info" if success else "warning"
        message = f"Authentication {'succeeded' if success else 'failed'} for user {user_id}"

        self.logger.log(level.upper(), message, extra=event_data)

    def log_authorization(self, user_id: str, resource: str, action: str,
                         granted: bool, details: Dict[str, Any] = None) -> None:
        """Log authorization events."""
        event_data = {
            "event_type": "authorization",
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "granted": granted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        }

        level = "info" if granted else "warning"
        message = f"Authorization {'granted' if granted else 'denied'} for user {user_id} on {resource}:{action}"

        self.logger.log(level.upper(), message, extra=event_data)

    def log_data_access(self, user_id: str, data_type: str, operation: str,
                       record_count: int = None, details: Dict[str, Any] = None) -> None:
        """Log data access events."""
        event_data = {
            "event_type": "data_access",
            "user_id": user_id,
            "data_type": data_type,
            "operation": operation,
            "record_count": record_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        }

        message = f"Data access: {user_id} performed {operation} on {data_type}"
        if record_count:
            message += f" ({record_count} records)"

        self.logger.info(message, extra=event_data)

    def log_system_event(self, event_type: str, severity: str, description: str,
                        component: str = None, details: Dict[str, Any] = None) -> None:
        """Log system events."""
        event_data = {
            "event_type": f"system_{event_type}",
            "severity": severity,
            "component": component,
            "description": description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        }

        self.logger.log(severity.upper(), f"System event: {description}", extra=event_data)


class ResourceMonitor:
    """Monitor system resources and log warnings when thresholds are exceeded."""

    def __init__(self, memory_threshold_mb: int = 1024, cpu_threshold_percent: float = 80.0):
        self.memory_threshold_mb = memory_threshold_mb
        self.cpu_threshold_percent = cpu_threshold_percent
        self.logger = logger.bind(component="resource_monitor")

    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB
        memory_percent = process.memory_percent()

        system_memory = psutil.virtual_memory()

        memory_data = {
            "process_memory_mb": round(memory_mb, 2),
            "process_memory_percent": round(memory_percent, 2),
            "system_memory_total_gb": round(system_memory.total / (1024**3), 2),
            "system_memory_available_gb": round(system_memory.available / (1024**3), 2),
            "system_memory_percent": system_memory.percent,
            "threshold_exceeded": memory_mb > self.memory_threshold_mb
        }

        if memory_data["threshold_exceeded"]:
            self.logger.warning(
                f"Memory usage threshold exceeded: {memory_mb:.2f}MB > {self.memory_threshold_mb}MB",
                extra={"event_type": "resource_warning", **memory_data}
            )

        return memory_data

    def check_cpu_usage(self) -> Dict[str, Any]:
        """Check current CPU usage."""
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        system_cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        cpu_data = {
            "process_cpu_percent": round(cpu_percent, 2),
            "system_cpu_percent": round(system_cpu_percent, 2),
            "cpu_count": cpu_count,
            "threshold_exceeded": cpu_percent > self.cpu_threshold_percent
        }

        if cpu_data["threshold_exceeded"]:
            self.logger.warning(
                f"CPU usage threshold exceeded: {cpu_percent:.2f}% > {self.cpu_threshold_percent}%",
                extra={"event_type": "resource_warning", **cpu_data}
            )

        return cpu_data

    def check_disk_usage(self, path: str = "/") -> Dict[str, Any]:
        """Check disk usage for specified path."""
        disk_usage = psutil.disk_usage(path)

        disk_data = {
            "path": path,
            "total_gb": round(disk_usage.total / (1024**3), 2),
            "used_gb": round(disk_usage.used / (1024**3), 2),
            "free_gb": round(disk_usage.free / (1024**3), 2),
            "percent_used": round((disk_usage.used / disk_usage.total) * 100, 2)
        }

        # Warn if less than 10% free space
        if disk_data["percent_used"] > 90:
            self.logger.warning(
                f"Disk space critically low: {disk_data['percent_used']:.1f}% used on {path}",
                extra={"event_type": "resource_warning", **disk_data}
            )

        return disk_data

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        return {
            "memory": self.check_memory_usage(),
            "cpu": self.check_cpu_usage(),
            "disk": self.check_disk_usage(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None,
    max_size_mb: int = 100,
    backup_count: int = 5,
    enable_audit: bool = True
) -> None:
    """
    Set up comprehensive logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type (json or text)
        log_file: Optional file path for logging
        max_size_mb: Maximum log file size in MB
        backup_count: Number of backup log files to keep
        enable_audit: Enable audit logging
    """
    # Remove default handler
    logger.remove()

    # Configure format
    if format_type == "json":
        log_format = (
            "{"
            "\"timestamp\": \"{time:YYYY-MM-DD HH:mm:ss.SSS}\", "
            "\"level\": \"{level}\", "
            "\"module\": \"{name}\", "
            "\"function\": \"{function}\", "
            "\"line\": {line}, "
            "\"message\": \"{message}\", "
            "\"extra\": {extra}"
            "}"
        )
    else:
        log_format = (
            "[{time:YYYY-MM-DD HH:mm:ss.SSS}] {level: <8} | "
            "{name}:{function}:{line} | {message} | {extra}"
        )

    # Add console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        colorize=(format_type != "json")
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation=f"{max_size_mb} MB",
            retention=f"{backup_count} files",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )

    # Configure standard library logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def log_function_call(
    include_args: bool = True,
    include_result: bool = False,
    log_level: str = "DEBUG",
    max_arg_length: int = 100
):
    """
    Decorator to log function calls with arguments and timing.

    Args:
        include_args: Whether to include function arguments in logs
        include_result: Whether to include function result in logs
        log_level: Log level for the function call logs
        max_arg_length: Maximum length of argument values in logs
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"

            # Prepare arguments for logging
            log_data = {
                "event_type": "function_call",
                "function": func_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            if include_args:
                # Safely serialize arguments
                try:
                    safe_args = []
                    for arg in args:
                        arg_str = str(arg)
                        if len(arg_str) > max_arg_length:
                            arg_str = arg_str[:max_arg_length] + "..."
                        safe_args.append(arg_str)

                    safe_kwargs = {}
                    for k, v in kwargs.items():
                        v_str = str(v)
                        if len(v_str) > max_arg_length:
                            v_str = v_str[:max_arg_length] + "..."
                        safe_kwargs[k] = v_str

                    log_data["args"] = safe_args
                    log_data["kwargs"] = safe_kwargs
                except Exception as e:
                    log_data["args_error"] = str(e)

            # Execute function with timing
            with PerformanceTimer(func_name) as timer:
                try:
                    result = func(*args, **kwargs)
                    log_data["success"] = True

                    if include_result:
                        try:
                            result_str = str(result)
                            if len(result_str) > max_arg_length:
                                result_str = result_str[:max_arg_length] + "..."
                            log_data["result"] = result_str
                        except Exception as e:
                            log_data["result_error"] = str(e)

                    return result

                except Exception as e:
                    log_data["success"] = False
                    log_data["error"] = str(e)
                    log_data["error_type"] = type(e).__name__
                    log_data["traceback"] = traceback.format_exc()

                    logger.error(f"Function {func_name} failed: {str(e)}", extra=log_data)
                    raise
                finally:
                    log_data["duration_ms"] = timer.duration_ms
                    if log_data.get("success", False):
                        logger.log(log_level.upper(), f"Function {func_name} completed", extra=log_data)

        return wrapper
    return decorator


@contextmanager
def log_context(context: LogContext):
    """Context manager for maintaining log context across operations."""
    old_context = getattr(logger, "_context", {})
    new_context = {
        **old_context,
        "request_id": context.request_id,
        "user_id": context.user_id,
        "session_id": context.session_id,
        "mission_id": context.mission_id,
        "spacecraft_id": context.spacecraft_id,
        "component": context.component,
        "operation": context.operation,
        **context.metadata
    }

    logger.configure(extra=new_context)
    try:
        yield logger
    finally:
        logger.configure(extra=old_context)


# Global instances
audit = AuditLogger()
resource_monitor = ResourceMonitor()

# Export commonly used items
__all__ = [
    "logger",
    "audit",
    "resource_monitor",
    "PerformanceTimer",
    "AuditLogger",
    "ResourceMonitor",
    "LogContext",
    "setup_logging",
    "log_function_call",
    "log_context"
]
