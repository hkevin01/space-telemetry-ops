const request = require('supertest');
const app = require('../index');
const redis = require('redis');

// Mock Redis
jest.mock('redis', () => ({
  createClient: jest.fn(() => ({
    connect: jest.fn(),
    on: jest.fn(),
    set: jest.fn(),
    get: jest.fn(),
    disconnect: jest.fn(),
    isReady: true
  }))
}));

describe('Telemetry Ingestion Service', () => {
  let server;

  beforeAll(() => {
    server = app.listen(0);
  });

  afterAll(async () => {
    await server.close();
  });

  describe('Health Check', () => {
    test('GET /health should return 200', async () => {
      const response = await request(app).get('/health');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('status', 'healthy');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body).toHaveProperty('uptime');
      expect(response.body).toHaveProperty('redis');
    });
  });

  describe('Metrics', () => {
    test('GET /metrics should return metrics in Prometheus format', async () => {
      const response = await request(app).get('/metrics');

      expect(response.status).toBe(200);
      expect(response.headers['content-type']).toMatch(/text\/plain/);
      expect(response.text).toContain('telemetry_messages_total');
    });
  });

  describe('Telemetry Ingestion', () => {
    test('POST /api/telemetry should accept valid telemetry data', async () => {
      const telemetryData = {
        timestamp: Date.now(),
        satelliteId: 'SAT-001',
        missionId: 'MISSION-Alpha',
        telemetryType: 'sensor',
        data: {
          temperature: 23.5,
          pressure: 101.3,
          voltage: 12.7
        }
      };

      const response = await request(app)
        .post('/api/telemetry')
        .send(telemetryData);

      expect(response.status).toBe(202);
      expect(response.body).toHaveProperty('status', 'accepted');
      expect(response.body).toHaveProperty('messageId');
    });

    test('POST /api/telemetry should reject invalid data', async () => {
      const invalidData = {
        timestamp: 'invalid-timestamp',
        satelliteId: '',
        data: 'not-an-object'
      };

      const response = await request(app)
        .post('/api/telemetry')
        .send(invalidData);

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    test('POST /api/telemetry/batch should accept multiple telemetry messages', async () => {
      const batchData = [
        {
          timestamp: Date.now(),
          satelliteId: 'SAT-001',
          missionId: 'MISSION-Alpha',
          telemetryType: 'sensor',
          data: { temperature: 23.5 }
        },
        {
          timestamp: Date.now(),
          satelliteId: 'SAT-002',
          missionId: 'MISSION-Beta',
          telemetryType: 'status',
          data: { status: 'operational' }
        }
      ];

      const response = await request(app)
        .post('/api/telemetry/batch')
        .send({ messages: batchData });

      expect(response.status).toBe(202);
      expect(response.body).toHaveProperty('status', 'accepted');
      expect(response.body).toHaveProperty('processedCount', 2);
    });

    test('POST /api/telemetry/batch should handle mixed valid/invalid data', async () => {
      const batchData = [
        {
          timestamp: Date.now(),
          satelliteId: 'SAT-001',
          missionId: 'MISSION-Alpha',
          telemetryType: 'sensor',
          data: { temperature: 23.5 }
        },
        {
          timestamp: 'invalid',
          satelliteId: '',
          data: 'invalid'
        }
      ];

      const response = await request(app)
        .post('/api/telemetry/batch')
        .send({ messages: batchData });

      expect(response.status).toBe(207);
      expect(response.body).toHaveProperty('processedCount', 1);
      expect(response.body).toHaveProperty('errorCount', 1);
      expect(response.body).toHaveProperty('errors');
    });
  });

  describe('WebSocket Connection', () => {
    test('WebSocket should accept connections', (done) => {
      const WebSocket = require('ws');
      const ws = new WebSocket(`ws://localhost:${server.address().port}`);

      ws.on('open', () => {
        ws.close();
        done();
      });

      ws.on('error', done);
    });
  });

  describe('Rate Limiting', () => {
    test('Should handle high request volume', async () => {
      const telemetryData = {
        timestamp: Date.now(),
        satelliteId: 'SAT-001',
        missionId: 'MISSION-Alpha',
        telemetryType: 'sensor',
        data: { temperature: 23.5 }
      };

      // Send 100 requests rapidly
      const promises = Array(100).fill().map(() =>
        request(app).post('/api/telemetry').send(telemetryData)
      );

      const responses = await Promise.all(promises);

      // Most should succeed (202), some might be rate limited (429)
      const successCount = responses.filter(r => r.status === 202).length;
      const rateLimitedCount = responses.filter(r => r.status === 429).length;

      expect(successCount + rateLimitedCount).toBe(100);
      expect(successCount).toBeGreaterThan(80); // At least 80% should succeed
    });
  });
});
