# Node.js Ingestion Service

High-performance telemetry ingestion service built with Node.js for processing satellite telemetry data at scale.

## Features

- **High Throughput**: Handles 50,000+ messages per second
- **Real-time Processing**: WebSocket broadcasting for live updates
- **Redis Integration**: Fast caching and message queuing
- **Batch Processing**: Efficient bulk data ingestion
- **Rate Limiting**: Built-in protection against overload
- **Health Monitoring**: Comprehensive health checks and metrics
- **Validation**: Joi schema validation for telemetry data
- **Logging**: Structured logging with Winston
- **Docker Ready**: Production-ready containerization

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Satellites    │───▶│ Ingestion API   │───▶│     Redis       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │                        │
                               ▼                        ▼
                      ┌─────────────────┐    ┌─────────────────┐
                      │   WebSocket     │    │   ETL Pipeline  │
                      │   Broadcasting  │    │   (Airflow)     │
                      └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites
- Node.js 18+
- Redis server
- Docker (optional)

### Installation

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Redis configuration
   ```

3. **Start the Service**
   ```bash
   npm start
   ```

### Docker Deployment

1. **Build Image**
   ```bash
   docker build -t telemetry-ingestion .
   ```

2. **Run Container**
   ```bash
   docker run -d \
     --name telemetry-ingestion \
     -p 8080:8080 \
     -p 8081:8081 \
     -e REDIS_HOST=redis-server \
     telemetry-ingestion
   ```

## API Endpoints

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": 1703123456789,
  "uptime": 3600.123,
  "redis": "connected",
  "memory": {
    "used": "45.2 MB",
    "total": "128.0 MB"
  }
}
```

### Single Telemetry Ingestion
```http
POST /api/telemetry
Content-Type: application/json

{
  "timestamp": 1703123456789,
  "satelliteId": "SAT-001",
  "missionId": "MISSION-Alpha",
  "telemetryType": "sensor",
  "data": {
    "temperature": 23.5,
    "pressure": 101.3,
    "voltage": 12.7
  }
}
```

### Batch Telemetry Ingestion
```http
POST /api/telemetry/batch
Content-Type: application/json

{
  "messages": [
    {
      "timestamp": 1703123456789,
      "satelliteId": "SAT-001",
      "missionId": "MISSION-Alpha",
      "telemetryType": "sensor",
      "data": { "temperature": 23.5 }
    }
  ]
}
```

### Metrics (Prometheus Format)
```http
GET /metrics
```

## WebSocket Real-time Updates

Connect to WebSocket server for real-time telemetry updates:

```javascript
const ws = new WebSocket('ws://localhost:8081');

ws.on('message', (data) => {
  const telemetry = JSON.parse(data);
  console.log('Real-time telemetry:', telemetry);
});
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NODE_ENV` | Environment mode | `development` |
| `PORT` | HTTP server port | `8080` |
| `WS_PORT` | WebSocket server port | `8081` |
| `REDIS_HOST` | Redis server host | `localhost` |
| `REDIS_PORT` | Redis server port | `6379` |
| `REDIS_PASSWORD` | Redis password | `` |
| `RATE_LIMIT_MAX_REQUESTS` | Max requests per window | `1000` |
| `BATCH_SIZE` | Batch processing size | `100` |
| `LOG_LEVEL` | Logging level | `info` |

### Telemetry Data Schema

```javascript
{
  timestamp: Number,     // Unix timestamp (required)
  satelliteId: String,   // Satellite identifier (required)
  missionId: String,     // Mission identifier (required)
  telemetryType: String, // Type: sensor, status, command, etc. (required)
  data: Object          // Telemetry payload (required)
}
```

## Performance Metrics

- **Throughput**: 50,000+ messages/second
- **Latency**: Sub-millisecond processing
- **Memory**: ~128MB baseline usage
- **CPU**: Optimized for multi-core utilization
- **Connections**: Supports 1,000+ concurrent WebSocket connections

## Monitoring

### Health Checks
- `/health` endpoint for service status
- Redis connectivity monitoring
- Memory usage tracking
- Uptime reporting

### Metrics Collection
- Prometheus-compatible metrics at `/metrics`
- Message throughput counters
- Error rate tracking
- Response time histograms
- WebSocket connection counts

### Logging
- Structured JSON logging with Winston
- Configurable log levels
- File rotation support
- Request/response logging
- Error tracking with stack traces

## Development

### Running Tests
```bash
npm test
```

### Code Coverage
```bash
npm run test:coverage
```

### Linting
```bash
npm run lint
```

### Development Mode
```bash
npm run dev
```

## Production Deployment

### Security Considerations
- CORS configuration
- Rate limiting enabled
- Helmet security headers
- Input validation
- Error sanitization

### Scaling
- Horizontal scaling with load balancer
- Redis clustering for high availability
- Container orchestration with Kubernetes
- Auto-scaling based on metrics

### Monitoring
- Health check endpoints
- Prometheus metrics integration
- Log aggregation (ELK stack compatible)
- Alert integration (webhook support)

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Check Redis server status
   - Verify connection parameters
   - Check network connectivity

2. **High Memory Usage**
   - Monitor batch sizes
   - Check Redis memory usage
   - Review log retention settings

3. **Rate Limiting Triggered**
   - Adjust `RATE_LIMIT_MAX_REQUESTS`
   - Implement client-side throttling
   - Consider batch processing

### Debug Mode
```bash
DEBUG=* npm start
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details
