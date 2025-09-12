# Space Telemetry Operations - Quick Start Guide

## Prerequisites

- Docker & Docker Compose
- Node.js 18+ (for development)
- Python 3.10+ (for development)
- Redis, PostgreSQL, MinIO (via Docker)

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/your-org/space-telemetry-ops
cd space-telemetry-ops
cp .env.example .env
# Edit .env with your configuration
```

### 2. Start All Services
```bash
# Production deployment
docker-compose -f docker/docker-compose.yml up -d

# Development environment
docker-compose -f docker/docker-compose.dev.yml up -d
```

### 3. Access Applications

| Service | URL | Description |
|---------|-----|-------------|
| Frontend Dashboard | http://localhost:3000 | Mission Control Interface |
| FastAPI Backend | http://localhost:8083 | REST API & Documentation |
| Ingestion Service | http://localhost:8080 | Telemetry Data Ingestion |
| Airflow ETL | http://localhost:8082 | Data Pipeline Management |
| Grafana Monitoring | http://localhost:3001 | System Monitoring |
| MinIO Console | http://localhost:9001 | Object Storage Management |

## Default Credentials

| Service | Username | Password |
|---------|----------|----------|
| Airflow | admin | admin123 |
| Grafana | admin | admin123 |
| MinIO | minio | minio123456 |
| PostgreSQL | telemetry_user | telemetry_pass123 |

## Development Workflow

### Frontend Development
```bash
cd src/frontend
npm install
npm start
# Open http://localhost:3000
```

### Backend Development
```bash
cd src/backend
poetry install
poetry shell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8083
# Open http://localhost:8083/docs
```

### Ingestion Service Development
```bash
cd src/services/ingest-node
npm install
npm run dev
# Service runs on http://localhost:8080
```

## Testing

### Run All Tests
```bash
# Frontend tests
cd src/frontend && npm test

# Backend tests
cd src/backend && poetry run pytest

# Ingestion service tests
cd src/services/ingest-node && npm test

# Integration tests
cd tests && python -m pytest integration/
```

### Load Testing
```bash
# Test ingestion service performance
cd tests/load && node telemetry-load-test.js
```

## Data Flow Test

### 1. Send Test Telemetry
```bash
curl -X POST http://localhost:8080/api/telemetry \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": '$(date +%s000)',
    "satelliteId": "SAT-001",
    "missionId": "MISSION-Alpha",
    "telemetryType": "sensor",
    "data": {
      "temperature": 23.5,
      "pressure": 101.3,
      "voltage": 12.7
    }
  }'
```

### 2. Check Redis Storage
```bash
redis-cli -h localhost -p 6379 keys "telemetry:*"
redis-cli -h localhost -p 6379 get "telemetry:SAT-001:latest"
```

### 3. Verify ETL Processing
- Open Airflow UI: http://localhost:8082
- Check `space_telemetry_etl_pipeline` DAG status
- Verify data in PostgreSQL and MinIO

### 4. Monitor Real-time Updates
```javascript
// Connect to WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:8081');
ws.onmessage = (event) => {
  console.log('Real-time telemetry:', JSON.parse(event.data));
};
```

## Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check Docker logs
docker-compose -f docker/docker-compose.yml logs [service-name]

# Restart specific service
docker-compose -f docker/docker-compose.yml restart [service-name]
```

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8080
lsof -i :8080

# Kill process using port
kill -9 $(lsof -t -i:8080)
```

#### Database Connection Issues
```bash
# Test PostgreSQL connection
docker exec -it space-telemetry-postgres psql -U telemetry_user -d telemetry

# Test Redis connection
docker exec -it space-telemetry-redis redis-cli ping
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check service health
curl http://localhost:8080/health
curl http://localhost:8083/health
```

## Performance Benchmarks

### Expected Performance
- **Ingestion Rate**: 50,000+ messages/second
- **API Response Time**: <100ms (95th percentile)
- **WebSocket Latency**: <10ms
- **ETL Processing**: 1M records/hour
- **Dashboard Update**: Real-time (<1 second)

### Load Test Results
```bash
# Generate load test report
cd tests/performance
./run-performance-tests.sh
```

## Security Checklist

- [ ] Change all default passwords
- [ ] Configure SSL/TLS certificates
- [ ] Set up firewall rules
- [ ] Enable audit logging
- [ ] Configure backup schedules
- [ ] Review security policies
- [ ] Run security scan: `npm audit` and `safety check`

## Next Steps

1. **Configure Production Environment**
   - Set up SSL certificates
   - Configure load balancers
   - Set up monitoring and alerting
   - Configure backup and disaster recovery

2. **Customize for Your Mission**
   - Modify telemetry schemas
   - Add custom data transformations
   - Create mission-specific dashboards
   - Configure alert thresholds

3. **Scale Deployment**
   - Set up Kubernetes orchestration
   - Configure auto-scaling
   - Implement distributed caching
   - Set up multi-region deployment

## Support

- **Documentation**: See `docs/` directory
- **Issues**: GitHub Issues
- **Security**: security@space-telemetry.com
- **Community**: [Discussion Forum](https://github.com/your-org/space-telemetry-ops/discussions)

---

**Ready to process satellite telemetry at scale!** 🚀
