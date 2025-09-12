"""
Configuration for Mission Control Dashboard Enhancement

This module provides configuration management for the enhanced mission control
dashboard, including widget settings, layout preferences, performance tuning,
and integration parameters.
"""

from pydantic import BaseSettings, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class DashboardTheme(str, Enum):
    """Available dashboard themes"""
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"


class ChartAnimationLevel(str, Enum):
    """Chart animation performance levels"""
    NONE = "none"
    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"


class DashboardConfig(BaseSettings):
    """Main dashboard configuration"""

    # Basic settings
    enabled: bool = Field(True, description="Enable dashboard enhancement features")
    theme: DashboardTheme = Field(DashboardTheme.DARK, description="Default dashboard theme")

    # Performance settings
    max_data_points_per_widget: int = Field(1000, description="Maximum data points per widget")
    widget_update_interval_seconds: int = Field(30, description="Widget update interval in seconds")
    cache_ttl_minutes: int = Field(5, description="Widget data cache TTL in minutes")
    max_concurrent_widgets: int = Field(50, description="Maximum concurrent widgets per dashboard")

    # WebSocket settings
    websocket_ping_interval: int = Field(30, description="WebSocket ping interval in seconds")
    websocket_timeout_seconds: int = Field(60, description="WebSocket connection timeout")
    max_websocket_connections: int = Field(1000, description="Maximum concurrent WebSocket connections")

    # Chart settings
    chart_animation_level: ChartAnimationLevel = Field(
        ChartAnimationLevel.STANDARD,
        description="Chart animation performance level"
    )
    default_color_scheme: str = Field("blue", description="Default widget color scheme")
    enable_chart_zoom: bool = Field(True, description="Enable chart zoom/pan functionality")
    enable_data_export: bool = Field(True, description="Enable data export from widgets")

    # Grid layout settings
    grid_columns: int = Field(12, description="Number of grid columns")
    grid_rows: int = Field(20, description="Number of grid rows")
    widget_margin: int = Field(16, description="Widget margin in pixels")
    min_widget_width: int = Field(2, description="Minimum widget width in grid units")
    min_widget_height: int = Field(2, description="Minimum widget height in grid units")

    # Alert settings
    max_alerts_displayed: int = Field(50, description="Maximum alerts displayed in panel")
    alert_auto_dismiss_minutes: int = Field(60, description="Auto-dismiss alerts after minutes")
    enable_alert_sound: bool = Field(True, description="Enable audio alerts")
    critical_alert_blink: bool = Field(True, description="Blink critical alerts")

    # Integration settings
    anomaly_detection_integration: bool = Field(True, description="Enable anomaly detection integration")
    performance_monitoring_integration: bool = Field(True, description="Enable performance monitoring integration")
    telemetry_realtime_updates: bool = Field(True, description="Enable real-time telemetry updates")

    # Template settings
    auto_create_mission_dashboards: bool = Field(True, description="Auto-create dashboards for new missions")
    default_satellite_widgets: List[str] = Field(
        ["altitude", "velocity", "power_level", "temperature", "attitude"],
        description="Default widgets for satellite missions"
    )
    default_rover_widgets: List[str] = Field(
        ["position", "battery_level", "motor_current", "env_temperature"],
        description="Default widgets for rover missions"
    )
    default_probe_widgets: List[str] = Field(
        ["distance_earth", "rtg_power", "comm_delay", "instrument_health"],
        description="Default widgets for probe missions"
    )

    # Security settings
    require_authentication: bool = Field(True, description="Require user authentication")
    enable_layout_sharing: bool = Field(True, description="Enable dashboard layout sharing")
    max_layouts_per_user: int = Field(20, description="Maximum layouts per user")

    class Config:
        env_prefix = "DASHBOARD_"
        case_sensitive = False


class WidgetTypeConfig(BaseSettings):
    """Configuration for specific widget types"""

    # Line chart settings
    line_chart_max_points: int = Field(500, description="Maximum points for line charts")
    line_chart_smoothing: bool = Field(True, description="Enable line chart smoothing")
    line_chart_fill: bool = Field(False, description="Fill area under line charts")

    # Gauge settings
    gauge_animation_duration: int = Field(750, description="Gauge animation duration in ms")
    gauge_color_transitions: bool = Field(True, description="Enable gauge color transitions")
    gauge_show_percentage: bool = Field(True, description="Show percentage in gauges")

    # Status indicator settings
    status_blink_critical: bool = Field(True, description="Blink critical status indicators")
    status_show_timestamp: bool = Field(True, description="Show timestamp on status indicators")
    status_auto_refresh: bool = Field(True, description="Auto-refresh status indicators")

    # Bar chart settings
    bar_chart_animation: bool = Field(True, description="Enable bar chart animations")
    bar_chart_3d_effect: bool = Field(False, description="Enable 3D effect on bar charts")

    # Scatter plot settings
    scatter_point_size: int = Field(3, description="Default scatter plot point size")
    scatter_show_trend_line: bool = Field(False, description="Show trend lines on scatter plots")

    class Config:
        env_prefix = "WIDGET_"
        case_sensitive = False


class AlertConfig(BaseSettings):
    """Configuration for alert management"""

    # Alert levels
    info_color: str = Field("#3b82f6", description="Info alert color")
    warning_color: str = Field("#f59e0b", description="Warning alert color")
    critical_color: str = Field("#ef4444", description="Critical alert color")

    # Sound settings
    enable_sound: bool = Field(True, description="Enable alert sounds")
    info_sound_file: Optional[str] = Field(None, description="Info alert sound file")
    warning_sound_file: Optional[str] = Field(None, description="Warning alert sound file")
    critical_sound_file: Optional[str] = Field(None, description="Critical alert sound file")

    # Notification settings
    enable_browser_notifications: bool = Field(True, description="Enable browser notifications")
    enable_email_notifications: bool = Field(False, description="Enable email notifications")
    email_smtp_server: Optional[str] = Field(None, description="SMTP server for email alerts")
    email_sender: Optional[str] = Field(None, description="Email sender address")

    # Alert aggregation
    group_similar_alerts: bool = Field(True, description="Group similar alerts together")
    alert_grouping_window_minutes: int = Field(5, description="Time window for alert grouping")
    max_grouped_alerts: int = Field(10, description="Maximum alerts in a group")

    class Config:
        env_prefix = "ALERT_"
        case_sensitive = False


class PerformanceConfig(BaseSettings):
    """Performance optimization configuration"""

    # Rendering performance
    enable_virtualization: bool = Field(True, description="Enable widget virtualization")
    lazy_load_widgets: bool = Field(True, description="Lazy load off-screen widgets")
    debounce_updates_ms: int = Field(100, description="Debounce widget updates in ms")

    # Data processing
    enable_data_sampling: bool = Field(True, description="Enable data point sampling for large datasets")
    sampling_threshold: int = Field(1000, description="Data point threshold for sampling")
    sampling_algorithm: str = Field("lttb", description="Sampling algorithm (lttb, average, min-max)")

    # Memory management
    max_memory_usage_mb: int = Field(512, description="Maximum memory usage in MB")
    garbage_collection_interval: int = Field(300, description="Garbage collection interval in seconds")
    clear_cache_on_memory_pressure: bool = Field(True, description="Clear caches under memory pressure")

    # Network optimization
    batch_websocket_updates: bool = Field(True, description="Batch WebSocket updates")
    websocket_batch_size: int = Field(10, description="WebSocket update batch size")
    websocket_batch_delay_ms: int = Field(50, description="WebSocket batch delay in ms")

    class Config:
        env_prefix = "PERFORMANCE_"
        case_sensitive = False


class SecurityConfig(BaseSettings):
    """Security configuration for dashboard"""

    # Authentication
    jwt_secret_key: str = Field("your-secret-key-change-in-production", description="JWT secret key")
    jwt_expiration_hours: int = Field(24, description="JWT token expiration in hours")
    require_https: bool = Field(False, description="Require HTTPS for all connections")

    # Authorization
    enable_rbac: bool = Field(False, description="Enable role-based access control")
    default_user_role: str = Field("viewer", description="Default user role")
    admin_users: List[str] = Field([], description="List of admin user IDs")

    # Data protection
    encrypt_websocket_data: bool = Field(False, description="Encrypt WebSocket communications")
    log_user_actions: bool = Field(True, description="Log user actions for audit")
    anonymize_telemetry_data: bool = Field(False, description="Anonymize sensitive telemetry data")

    # Rate limiting
    enable_rate_limiting: bool = Field(True, description="Enable API rate limiting")
    api_requests_per_minute: int = Field(1000, description="API requests per minute per user")
    websocket_messages_per_minute: int = Field(6000, description="WebSocket messages per minute")

    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = False


# Global configuration instances
dashboard_config = DashboardConfig()
widget_config = WidgetTypeConfig()
alert_config = AlertConfig()
performance_config = PerformanceConfig()
security_config = SecurityConfig()


# Configuration helper functions
def get_widget_defaults(chart_type: str) -> Dict[str, Any]:
    """Get default configuration for a specific widget type"""

    defaults = {
        "line": {
            "max_points": widget_config.line_chart_max_points,
            "smoothing": widget_config.line_chart_smoothing,
            "fill": widget_config.line_chart_fill,
            "animation_duration": 750
        },
        "gauge": {
            "animation_duration": widget_config.gauge_animation_duration,
            "color_transitions": widget_config.gauge_color_transitions,
            "show_percentage": widget_config.gauge_show_percentage
        },
        "status": {
            "blink_critical": widget_config.status_blink_critical,
            "show_timestamp": widget_config.status_show_timestamp,
            "auto_refresh": widget_config.status_auto_refresh
        },
        "bar": {
            "animation": widget_config.bar_chart_animation,
            "3d_effect": widget_config.bar_chart_3d_effect
        },
        "scatter": {
            "point_size": widget_config.scatter_point_size,
            "show_trend_line": widget_config.scatter_show_trend_line
        }
    }

    return defaults.get(chart_type, {})


def get_mission_template_config(mission_type: str) -> Dict[str, Any]:
    """Get default widget configuration for a mission type"""

    templates = {
        "satellite": {
            "widgets": dashboard_config.default_satellite_widgets,
            "theme": "dark",
            "update_interval": 30,
            "grid_density": "normal"
        },
        "rover": {
            "widgets": dashboard_config.default_rover_widgets,
            "theme": "dark",
            "update_interval": 15,
            "grid_density": "compact"
        },
        "probe": {
            "widgets": dashboard_config.default_probe_widgets,
            "theme": "dark",
            "update_interval": 60,
            "grid_density": "sparse"
        },
        "station": {
            "widgets": ["oxygen_level", "co2_level", "cabin_pressure", "power_generation"],
            "theme": "light",
            "update_interval": 10,
            "grid_density": "dense"
        }
    }

    return templates.get(mission_type, templates["satellite"])


def validate_dashboard_config() -> List[str]:
    """Validate dashboard configuration and return any warnings"""
    warnings = []

    # Check performance settings
    if dashboard_config.max_data_points_per_widget > 2000:
        warnings.append("High max_data_points_per_widget may impact performance")

    if dashboard_config.widget_update_interval_seconds < 5:
        warnings.append("Very frequent widget updates may cause high CPU usage")

    if dashboard_config.max_concurrent_widgets > 100:
        warnings.append("High widget count may impact browser performance")

    # Check WebSocket settings
    if dashboard_config.max_websocket_connections > 5000:
        warnings.append("High WebSocket connection limit may require server tuning")

    # Check security settings
    if not security_config.require_https and security_config.encrypt_websocket_data:
        warnings.append("WebSocket encryption enabled without HTTPS requirement")

    if security_config.jwt_secret_key == "your-secret-key-change-in-production":
        warnings.append("Default JWT secret key detected - change in production")

    # Check memory settings
    if performance_config.max_memory_usage_mb < 256:
        warnings.append("Low memory limit may cause frequent garbage collection")

    return warnings


# Export configuration objects
__all__ = [
    "DashboardConfig",
    "WidgetTypeConfig",
    "AlertConfig",
    "PerformanceConfig",
    "SecurityConfig",
    "dashboard_config",
    "widget_config",
    "alert_config",
    "performance_config",
    "security_config",
    "get_widget_defaults",
    "get_mission_template_config",
    "validate_dashboard_config"
]
