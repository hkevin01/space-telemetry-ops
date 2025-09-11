"""
Enhanced settings module with comprehensive configuration management.

This module provides robust configuration management with:
- Environment variable validation
- Type safety with Pydantic
- Default value handling
- Configuration caching
- Security-focused defaults
"""

import os
import secrets
from typing import Optional, List, Dict, Any
from datetime import timedelta
from pathlib import Path

from pydantic import BaseSettings, Field, validator
from pydantic.networks import PostgresDsn, RedisDsn


class DatabaseSettings(BaseSettings):
    """Database configuration with connection management."""

    host: str = Field(default="localhost", env="POSTGRES_HOST")
    port: int = Field(default=5432, env="POSTGRES_PORT", ge=1, le=65535)
    user: str = Field(default="telemetry", env="POSTGRES_USER")
    password: str = Field(default="telemetrypw", env="POSTGRES_PASSWORD")
    database: str = Field(default="telemetrydb", env="POSTGRES_DB")

    # Connection pool settings
    pool_size: int = Field(default=20, env="DB_POOL_SIZE", ge=5, le=100)
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW", ge=0, le=100)
    pool_timeout: float = Field(default=30.0, env="DB_POOL_TIMEOUT", gt=0, le=300)
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE", ge=300)

    # Performance settings
    statement_timeout: int = Field(default=30000, env="DB_STATEMENT_TIMEOUT", ge=1000)  # milliseconds
    query_timeout: float = Field(default=30.0, env="DB_QUERY_TIMEOUT", gt=0, le=300)  # seconds

    @property
    def dsn(self) -> str:
        """Generate PostgreSQL DSN with connection parameters."""
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_dsn(self) -> str:
        """Generate PostgreSQL async DSN."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisSettings(BaseSettings):
    """Redis configuration for caching and message queuing."""

    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT", ge=1, le=65535)
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB", ge=0, le=15)

    # Connection settings
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS", ge=5, le=100)
    socket_timeout: float = Field(default=5.0, env="REDIS_SOCKET_TIMEOUT", gt=0, le=30)
    socket_connect_timeout: float = Field(default=5.0, env="REDIS_CONNECT_TIMEOUT", gt=0, le=30)
    retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")

    # Cache settings
    default_ttl: int = Field(default=3600, env="REDIS_DEFAULT_TTL", ge=60)  # seconds

    @property
    def url(self) -> str:
        """Generate Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class SecuritySettings(BaseSettings):
    """Security and authentication configuration."""

    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32), env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=60, env="ACCESS_TOKEN_EXPIRE_MINUTES", ge=5, le=1440)
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS", ge=1, le=30)

    # Password policy
    password_min_length: int = Field(default=12, env="PASSWORD_MIN_LENGTH", ge=8)
    password_require_uppercase: bool = Field(default=True, env="PASSWORD_REQUIRE_UPPERCASE")
    password_require_lowercase: bool = Field(default=True, env="PASSWORD_REQUIRE_LOWERCASE")
    password_require_numbers: bool = Field(default=True, env="PASSWORD_REQUIRE_NUMBERS")
    password_require_symbols: bool = Field(default=True, env="PASSWORD_REQUIRE_SYMBOLS")

    # Rate limiting
    rate_limit_requests: int = Field(default=1000, env="RATE_LIMIT_REQUESTS", ge=100)
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW", ge=60)  # seconds

    # Security headers
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    cors_origins: List[str] = Field(default=["http://localhost:3000", "http://localhost:5173"], env="CORS_ORIGINS")
    enable_csrf: bool = Field(default=True, env="ENABLE_CSRF")

    @validator("secret_key")
    def validate_secret_key(cls, v: str) -> str:
        """Ensure secret key meets minimum security requirements."""
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v


class TelemetrySettings(BaseSettings):
    """Telemetry-specific configuration."""

    # Ingestion settings
    max_message_size: int = Field(default=1048576, env="MAX_MESSAGE_SIZE", ge=1024)  # 1MB
    max_batch_size: int = Field(default=1000, env="MAX_BATCH_SIZE", ge=1, le=10000)
    batch_timeout: float = Field(default=5.0, env="BATCH_TIMEOUT", gt=0, le=60)  # seconds

    # Processing settings
    max_concurrent_workers: int = Field(default=10, env="MAX_CONCURRENT_WORKERS", ge=1, le=100)
    message_retention_days: int = Field(default=90, env="MESSAGE_RETENTION_DAYS", ge=1, le=3650)

    # Quality control
    enable_validation: bool = Field(default=True, env="ENABLE_VALIDATION")
    enable_deduplication: bool = Field(default=True, env="ENABLE_DEDUPLICATION")
    anomaly_detection_threshold: float = Field(default=0.95, env="ANOMALY_THRESHOLD", ge=0.1, le=1.0)

    # Time synchronization
    time_sync_tolerance: float = Field(default=1.0, env="TIME_SYNC_TOLERANCE", ge=0.1, le=3600)  # seconds
    enable_ntp_sync: bool = Field(default=True, env="ENABLE_NTP_SYNC")
    ntp_servers: List[str] = Field(default=["pool.ntp.org"], env="NTP_SERVERS")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""

    # Metrics settings
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=8090, env="METRICS_PORT", ge=1024, le=65535)

    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    log_max_size: int = Field(default=100, env="LOG_MAX_SIZE", ge=10)  # MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT", ge=1, le=10)

    # Health check settings
    health_check_interval: float = Field(default=30.0, env="HEALTH_CHECK_INTERVAL", ge=5, le=300)  # seconds
    health_check_timeout: float = Field(default=5.0, env="HEALTH_CHECK_TIMEOUT", ge=1, le=30)  # seconds

    # Alerting settings
    enable_alerts: bool = Field(default=True, env="ENABLE_ALERTS")
    alert_webhook_url: Optional[str] = Field(default=None, env="ALERT_WEBHOOK_URL")

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


class PerformanceSettings(BaseSettings):
    """Performance and resource management configuration."""

    # Memory limits
    max_memory_usage: int = Field(default=2048, env="MAX_MEMORY_MB", ge=512, le=16384)  # MB
    memory_check_interval: float = Field(default=60.0, env="MEMORY_CHECK_INTERVAL", ge=10, le=300)  # seconds

    # CPU limits
    max_cpu_percent: float = Field(default=80.0, env="MAX_CPU_PERCENT", ge=10.0, le=95.0)
    cpu_check_interval: float = Field(default=30.0, env="CPU_CHECK_INTERVAL", ge=5, le=300)  # seconds

    # Disk space limits
    min_disk_space_gb: int = Field(default=10, env="MIN_DISK_SPACE_GB", ge=1, le=1000)
    disk_check_interval: float = Field(default=300.0, env="DISK_CHECK_INTERVAL", ge=60, le=3600)  # seconds

    # Cache settings
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    cache_max_size: int = Field(default=1000, env="CACHE_MAX_SIZE", ge=100, le=10000)
    cache_ttl: int = Field(default=3600, env="CACHE_TTL", ge=60, le=86400)  # seconds


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""

    # Application metadata
    app_name: str = Field(default="Space Telemetry Operations", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # Configuration sections
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    security: SecuritySettings = SecuritySettings()
    telemetry: TelemetrySettings = TelemetrySettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    performance: PerformanceSettings = PerformanceSettings()

    # API settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT", ge=1024, le=65535)
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")

    # External services
    minio_endpoint: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="telemetry-raw", env="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, env="MINIO_SECURE")

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True

    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_environments = ["development", "testing", "staging", "production"]
        if v.lower() not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v.lower()

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"

    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration dict."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "[{asctime}] {levelname} in {module}: {message}",
                    "style": "{",
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json" if self.monitoring.log_format == "json" else "default",
                },
            },
            "root": {
                "level": self.monitoring.log_level,
                "handlers": ["console"],
            },
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings
