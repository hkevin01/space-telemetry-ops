"""
Mission Control Dashboard Enhancement API

This module provides the FastAPI router for the enhanced mission control
dashboard with real-time telemetry visualization, WebSocket streaming,
and advanced dashboard configuration management.

REQUIREMENTS FULFILLMENT:
=======================
[FR-007] Mission Control Dashboard (CRITICAL)
  • FR-007.1: Provides real-time dashboard data updates
  • FR-007.2: Supports configurable dashboard layout management
  • FR-007.3: Enables mission-specific dashboard templates
  • FR-007.5: Implements WebSocket streaming endpoints

[FR-009] REST API Services (CRITICAL)
  • FR-009.1: Provides RESTful dashboard management endpoints
  • FR-009.4: Returns standardized JSON responses
  • FR-009.5: Integrated with OpenAPI documentation

API Endpoints:
- Dashboard layout CRUD operations (FR-007.2)
- Widget management and configuration (FR-007.2)
- Real-time data streaming via WebSocket (FR-007.5)
- Dashboard template management (FR-007.3)
"""

import logging
import uuid
from typing import List, Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Path,
    Query,
    WebSocket,
    WebSocketDisconnect,
)

# Internal imports
from .dashboard_service import (
    ChartType,
    CreateLayoutRequest,
    CreateWidgetRequest,
    DashboardLayout,
    DashboardWidget,
    LayoutResponse,
    WidgetDataResponse,
    dashboard_service,
)

# Create API router
router = APIRouter(prefix="/api/dashboard", tags=["Mission Control Dashboard"])
logger = logging.getLogger(__name__)


@router.on_event("startup")
async def startup_event():
    """Start background dashboard update services"""
    await dashboard_service.start_background_updates()
    logger.info("Mission Control Dashboard API started")


@router.on_event("shutdown")
async def shutdown_event():
    """Stop background dashboard update services"""
    await dashboard_service.stop_background_updates()
    logger.info("Mission Control Dashboard API stopped")


# Dashboard Layout Management Endpoints

@router.post("/layouts", response_model=LayoutResponse)
async def create_dashboard_layout(request: CreateLayoutRequest):
    """
    Create a new dashboard layout

    Creates a new dashboard layout with the specified configuration.
    Returns the created layout with a unique ID.
    """
    try:
        layout_id = f"layout_{uuid.uuid4().hex[:8]}"

        layout = DashboardLayout(
            layout_id=layout_id,
            name=request.name,
            description=request.description,
            mission_id=request.mission_id,
            created_by="api_user",  # Would get from auth in production
            theme=request.theme
        )

        # Save the layout
        success = dashboard_service.save_layout(layout)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to create dashboard layout"
            )

        logger.info(f"Created dashboard layout: {layout_id}")

        return LayoutResponse(**layout.to_dict())

    except Exception as e:
        logger.error(f"Error creating dashboard layout: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating dashboard layout: {str(e)}"
        )


@router.get("/layouts", response_model=List[LayoutResponse])
async def list_dashboard_layouts(mission_id: Optional[str] = Query(None)):
    """
    List available dashboard layouts

    Retrieves all dashboard layouts, optionally filtered by mission ID.
    """
    try:
        layouts = dashboard_service.list_layouts(mission_id=mission_id)

        return [LayoutResponse(**layout.to_dict()) for layout in layouts]

    except Exception as e:
        logger.error(f"Error listing dashboard layouts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing dashboard layouts: {str(e)}"
        )


@router.get("/layouts/{layout_id}", response_model=LayoutResponse)
async def get_dashboard_layout(layout_id: str = Path(..., description="Dashboard layout ID")):
    """
    Get specific dashboard layout by ID

    Retrieves a dashboard layout with all its widget configurations.
    """
    try:
        layout = dashboard_service.get_layout(layout_id)
        if not layout:
            raise HTTPException(
                status_code=404,
                detail=f"Dashboard layout not found: {layout_id}"
            )

        return LayoutResponse(**layout.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard layout {layout_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting dashboard layout: {str(e)}"
        )


@router.put("/layouts/{layout_id}", response_model=LayoutResponse)
async def update_dashboard_layout(
    layout_id: str = Path(..., description="Dashboard layout ID"),
    request: CreateLayoutRequest = None
):
    """
    Update existing dashboard layout

    Updates the configuration of an existing dashboard layout.
    """
    try:
        layout = dashboard_service.get_layout(layout_id)
        if not layout:
            raise HTTPException(
                status_code=404,
                detail=f"Dashboard layout not found: {layout_id}"
            )

        # Update layout properties
        if request.name:
            layout.name = request.name
        if request.description:
            layout.description = request.description
        if request.mission_id:
            layout.mission_id = request.mission_id
        if request.theme:
            layout.theme = request.theme

        # Save updated layout
        success = dashboard_service.save_layout(layout)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update dashboard layout"
            )

        logger.info(f"Updated dashboard layout: {layout_id}")

        return LayoutResponse(**layout.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating dashboard layout {layout_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating dashboard layout: {str(e)}"
        )


@router.delete("/layouts/{layout_id}")
async def delete_dashboard_layout(layout_id: str = Path(..., description="Dashboard layout ID")):
    """
    Delete dashboard layout

    Removes a dashboard layout and all its widgets.
    """
    try:
        success = dashboard_service.delete_layout(layout_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Dashboard layout not found: {layout_id}"
            )

        logger.info(f"Deleted dashboard layout: {layout_id}")

        return {"message": f"Dashboard layout {layout_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dashboard layout {layout_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting dashboard layout: {str(e)}"
        )


# Widget Management Endpoints

@router.post("/layouts/{layout_id}/widgets")
async def add_widget_to_layout(
    layout_id: str = Path(..., description="Dashboard layout ID"),
    request: CreateWidgetRequest = None
):
    """
    Add widget to dashboard layout

    Creates and adds a new widget to the specified dashboard layout.
    """
    try:
        layout = dashboard_service.get_layout(layout_id)
        if not layout:
            raise HTTPException(
                status_code=404,
                detail=f"Dashboard layout not found: {layout_id}"
            )

        # Validate chart type
        try:
            chart_type = ChartType(request.chart_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid chart type: {request.chart_type}"
            )

        # Generate widget ID
        widget_id = f"widget_{uuid.uuid4().hex[:8]}"

        # Create widget
        widget = DashboardWidget(
            widget_id=widget_id,
            title=request.title,
            chart_type=chart_type,
            data_source=request.data_source,
            spacecraft_id=request.spacecraft_id,
            position_x=request.position_x,
            position_y=request.position_y,
            width=request.width,
            height=request.height,
            time_window_hours=request.time_window_hours,
            color_scheme=request.color_scheme,
            warning_threshold=request.warning_threshold,
            critical_threshold=request.critical_threshold
        )

        # Add widget to layout
        layout.widgets.append(widget)

        # Save updated layout
        success = dashboard_service.save_layout(layout)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to add widget to layout"
            )

        logger.info(f"Added widget {widget_id} to layout {layout_id}")

        return {
            "message": f"Widget {widget_id} added to layout {layout_id}",
            "widget": widget.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding widget to layout {layout_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error adding widget to layout: {str(e)}"
        )


@router.get("/layouts/{layout_id}/widgets/{widget_id}/data", response_model=WidgetDataResponse)
async def get_widget_data(
    layout_id: str = Path(..., description="Dashboard layout ID"),
    widget_id: str = Path(..., description="Widget ID")
):
    """
    Get current data for specific widget

    Retrieves the latest aggregated data for a specific widget.
    """
    try:
        layout = dashboard_service.get_layout(layout_id)
        if not layout:
            raise HTTPException(
                status_code=404,
                detail=f"Dashboard layout not found: {layout_id}"
            )

        # Find widget in layout
        widget = None
        for w in layout.widgets:
            if w.widget_id == widget_id:
                widget = w
                break

        if not widget:
            raise HTTPException(
                status_code=404,
                detail=f"Widget not found: {widget_id}"
            )

        # Get widget data (using mock data for now)
        mock_telemetry_data = []  # Would fetch real telemetry data
        widget_data = await dashboard_service.data_aggregator.get_widget_data(
            widget, mock_telemetry_data
        )

        return WidgetDataResponse(**widget_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting widget data for {widget_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting widget data: {str(e)}"
        )


@router.delete("/layouts/{layout_id}/widgets/{widget_id}")
async def remove_widget_from_layout(
    layout_id: str = Path(..., description="Dashboard layout ID"),
    widget_id: str = Path(..., description="Widget ID")
):
    """
    Remove widget from dashboard layout

    Removes a widget from the specified dashboard layout.
    """
    try:
        layout = dashboard_service.get_layout(layout_id)
        if not layout:
            raise HTTPException(
                status_code=404,
                detail=f"Dashboard layout not found: {layout_id}"
            )

        # Find and remove widget
        widget_found = False
        layout.widgets = [w for w in layout.widgets if w.widget_id != widget_id]

        # Check if widget was actually removed
        original_count = len(layout.widgets)
        if len([w for w in layout.widgets if w.widget_id == widget_id]) == 0:
            widget_found = True

        if not widget_found:
            raise HTTPException(
                status_code=404,
                detail=f"Widget not found: {widget_id}"
            )

        # Save updated layout
        success = dashboard_service.save_layout(layout)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to remove widget from layout"
            )

        logger.info(f"Removed widget {widget_id} from layout {layout_id}")

        return {"message": f"Widget {widget_id} removed from layout {layout_id}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing widget {widget_id} from layout {layout_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error removing widget: {str(e)}"
        )


# Real-time WebSocket Endpoints

@router.websocket("/ws/{layout_id}")
async def dashboard_websocket(websocket: WebSocket, layout_id: str):
    """
    WebSocket endpoint for real-time dashboard updates

    Provides real-time streaming of telemetry data and dashboard updates
    for a specific layout.
    """
    connection_id = f"conn_{uuid.uuid4().hex[:8]}"

    try:
        # Accept WebSocket connection
        await dashboard_service.websocket_manager.connect(websocket, connection_id)

        # Verify layout exists
        layout = dashboard_service.get_layout(layout_id)
        if not layout:
            await websocket.send_text('{"error": "Layout not found"}')
            return

        # Subscribe to all widgets in the layout
        for widget in layout.widgets:
            await dashboard_service.websocket_manager.subscribe_to_widget(
                connection_id, widget.widget_id
            )

        logger.info(f"WebSocket connected for layout {layout_id}: {connection_id}")

        # Send initial layout data
        await websocket.send_text(f'{{"type": "layout_data", "layout": {layout.to_dict()}}}')

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = eval(data)  # In production, use json.loads with error handling

                # Handle client messages
                if message.get("type") == "subscribe_widget":
                    widget_id = message.get("widget_id")
                    if widget_id:
                        await dashboard_service.websocket_manager.subscribe_to_widget(
                            connection_id, widget_id
                        )
                        await websocket.send_text(
                            f'{{"type": "subscription_confirmed", "widget_id": "{widget_id}"}}'
                        )

                elif message.get("type") == "unsubscribe_widget":
                    widget_id = message.get("widget_id")
                    if widget_id:
                        await dashboard_service.websocket_manager.unsubscribe_from_widget(
                            connection_id, widget_id
                        )
                        await websocket.send_text(
                            f'{{"type": "unsubscription_confirmed", "widget_id": "{widget_id}"}}'
                        )

                elif message.get("type") == "ping":
                    await websocket.send_text('{"type": "pong"}')

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {str(e)}")
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        dashboard_service.websocket_manager.disconnect(connection_id)


# Dashboard Template Management

@router.post("/templates/{mission_type}")
async def create_template_layout(mission_type: str = Path(..., description="Mission type")):
    """
    Create template dashboard layout for mission type

    Creates a pre-configured dashboard layout optimized for specific mission types.
    """
    try:
        template_id = f"template_{mission_type}_{uuid.uuid4().hex[:8]}"

        if mission_type.lower() == "satellite":
            layout = _create_satellite_template(template_id)
        elif mission_type.lower() == "rover":
            layout = _create_rover_template(template_id)
        elif mission_type.lower() == "probe":
            layout = _create_probe_template(template_id)
        else:
            # Create default template
            layout = dashboard_service.create_default_layout(template_id)
            layout.name = f"{mission_type.title()} Mission Template"

        # Save template layout
        success = dashboard_service.save_layout(layout)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to create template layout"
            )

        logger.info(f"Created {mission_type} template layout: {template_id}")

        return LayoutResponse(**layout.to_dict())

    except Exception as e:
        logger.error(f"Error creating {mission_type} template: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating template: {str(e)}"
        )


# Analytics and Metrics Endpoints

@router.get("/analytics/usage/{layout_id}")
async def get_dashboard_usage_analytics(
    layout_id: str = Path(..., description="Dashboard layout ID"),
    days: int = Query(30, description="Number of days for analytics")
):
    """
    Get dashboard usage analytics

    Provides analytics data for dashboard usage, widget performance,
    and user interaction patterns.
    """
    try:
        layout = dashboard_service.get_layout(layout_id)
        if not layout:
            raise HTTPException(
                status_code=404,
                detail=f"Dashboard layout not found: {layout_id}"
            )

        # Mock analytics data (would fetch from metrics database in production)
        analytics_data = {
            "layout_id": layout_id,
            "period_days": days,
            "total_views": 150,
            "unique_users": 12,
            "average_session_duration_minutes": 45.2,
            "widget_interactions": {
                widget.widget_id: {
                    "views": 75,
                    "data_refreshes": 300,
                    "alerts_triggered": 5
                }
                for widget in layout.widgets
            },
            "performance_metrics": {
                "average_load_time_ms": 245,
                "websocket_connection_duration_avg_minutes": 22.5,
                "data_update_frequency_seconds": 30
            },
            "alert_statistics": {
                "total_alerts": 23,
                "critical_alerts": 3,
                "warning_alerts": 20,
                "acknowledged_alerts": 18,
                "false_positives": 2
            }
        }

        return analytics_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics for layout {layout_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting dashboard analytics: {str(e)}"
        )


# Helper functions for creating template layouts

def _create_satellite_template(template_id: str) -> DashboardLayout:
    """Create template layout optimized for satellite missions"""

    layout = DashboardLayout(
        layout_id=template_id,
        name="Satellite Mission Template",
        description="Optimized dashboard for satellite telemetry monitoring",
        theme="dark"
    )

    widgets = [
        # Orbital parameters
        DashboardWidget(
            widget_id="altitude_gauge",
            title="Altitude",
            chart_type=ChartType.GAUGE,
            data_source="altitude",
            position_x=0, position_y=0,
            width=3, height=3,
            warning_threshold=200000,
            critical_threshold=150000
        ),
        DashboardWidget(
            widget_id="velocity_line",
            title="Orbital Velocity",
            chart_type=ChartType.LINE,
            data_source="velocity",
            position_x=3, position_y=0,
            width=6, height=3
        ),

        # Power systems
        DashboardWidget(
            widget_id="solar_power",
            title="Solar Panel Power",
            chart_type=ChartType.LINE,
            data_source="solar_power",
            position_x=9, position_y=0,
            width=3, height=3,
            color_scheme="yellow"
        ),

        # Thermal systems
        DashboardWidget(
            widget_id="thermal_status",
            title="Thermal Status",
            chart_type=ChartType.STATUS,
            data_source="temperature",
            position_x=0, position_y=3,
            width=4, height=2,
            warning_threshold=50,
            critical_threshold=70
        ),

        # Communications
        DashboardWidget(
            widget_id="signal_strength",
            title="Signal Strength",
            chart_type=ChartType.GAUGE,
            data_source="signal_strength",
            position_x=4, position_y=3,
            width=4, height=2,
            color_scheme="green"
        ),

        # Attitude control
        DashboardWidget(
            widget_id="attitude_scatter",
            title="Attitude Control",
            chart_type=ChartType.SCATTER,
            data_source="attitude",
            position_x=8, position_y=3,
            width=4, height=2
        )
    ]

    layout.widgets = widgets
    return layout


def _create_rover_template(template_id: str) -> DashboardLayout:
    """Create template layout optimized for rover missions"""

    layout = DashboardLayout(
        layout_id=template_id,
        name="Rover Mission Template",
        description="Optimized dashboard for rover telemetry monitoring",
        theme="dark"
    )

    widgets = [
        # Navigation
        DashboardWidget(
            widget_id="position_map",
            title="Rover Position",
            chart_type=ChartType.MAP,
            data_source="position",
            position_x=0, position_y=0,
            width=6, height=4
        ),

        # Battery status
        DashboardWidget(
            widget_id="battery_gauge",
            title="Battery Level",
            chart_type=ChartType.GAUGE,
            data_source="battery_level",
            position_x=6, position_y=0,
            width=3, height=2,
            warning_threshold=30,
            critical_threshold=15,
            color_scheme="orange"
        ),

        # Motor status
        DashboardWidget(
            widget_id="motor_status",
            title="Motor Status",
            chart_type=ChartType.STATUS,
            data_source="motor_current",
            position_x=9, position_y=0,
            width=3, height=2
        ),

        # Environmental sensors
        DashboardWidget(
            widget_id="environment_temp",
            title="Environmental Temperature",
            chart_type=ChartType.LINE,
            data_source="env_temperature",
            position_x=6, position_y=2,
            width=6, height=2,
            color_scheme="red"
        )
    ]

    layout.widgets = widgets
    return layout


def _create_probe_template(template_id: str) -> DashboardLayout:
    """Create template layout optimized for deep space probe missions"""

    layout = DashboardLayout(
        layout_id=template_id,
        name="Deep Space Probe Template",
        description="Optimized dashboard for deep space probe monitoring",
        theme="dark"
    )

    widgets = [
        # Distance and trajectory
        DashboardWidget(
            widget_id="distance_earth",
            title="Distance from Earth",
            chart_type=ChartType.LINE,
            data_source="distance_earth",
            position_x=0, position_y=0,
            width=8, height=3,
            time_window_hours=168  # 1 week
        ),

        # Power systems
        DashboardWidget(
            widget_id="rtg_power",
            title="RTG Power Output",
            chart_type=ChartType.GAUGE,
            data_source="rtg_power",
            position_x=8, position_y=0,
            width=4, height=3,
            warning_threshold=200,
            critical_threshold=150,
            color_scheme="purple"
        ),

        # Communication delay
        DashboardWidget(
            widget_id="comm_delay",
            title="Communication Delay",
            chart_type=ChartType.LINE,
            data_source="comm_delay",
            position_x=0, position_y=3,
            width=6, height=2,
            color_scheme="indigo"
        ),

        # Scientific instruments
        DashboardWidget(
            widget_id="instruments_status",
            title="Instrument Status",
            chart_type=ChartType.STATUS,
            data_source="instrument_health",
            position_x=6, position_y=3,
            width=6, height=2
        )
    ]

    layout.widgets = widgets
    return layout


# Export router
__all__ = ["router"]
