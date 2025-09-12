"""
Basic test to ensure FastAPI application starts correctly.
"""
import pytest

def test_root_endpoint(client):
    """Test the root endpoint returns expected response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Space Telemetry Operations API"
    assert data["status"] == "operational"
    assert "timestamp" in data

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "space-telemetry-api"

def test_telemetry_status(client):
    """Test telemetry status endpoint."""
    response = client.get("/telemetry/status")
    assert response.status_code == 200
    data = response.json()
    assert "ingestion_rate" in data
    assert "active_spacecraft" in data
    assert "system_health" in data
