"""
Warm Path Data Tests - PostgreSQL Operational Analytics

Tests for operational telemetry data processing through PostgreSQL warm path.
Validates analytical queries, data integrity, and performance for mission
operations and real-time decision making.
"""

import asyncio
import pytest
import asyncpg
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random
import statistics
from dataclasses import dataclass

# Test configuration
WARM_PATH_CONFIG = {
    "postgres_host": "localhost",
    "postgres_port": 5432,
    "postgres_db": "telemetry",
    "postgres_user": "telemetry_user",
    "postgres_password": "telemetry_pass123",
    "query_performance_threshold_ms": 50.0,  # <50ms query time
    "bulk_insert_target": 10000,  # 10K records/sec
    "retention_days": 30,  # Warm path retention
    "max_connections": 20
}

@dataclass
class TelemetryRecord:
    """Structured telemetry record for PostgreSQL"""
    timestamp: datetime
    satellite_id: str
    mission_id: str
    telemetry_type: str
    category: str
    data_quality_score: int
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    voltage: Optional[float] = None
    current: Optional[float] = None
    power: Optional[float] = None
    altitude: Optional[float] = None
    velocity: Optional[float] = None
    status: Optional[str] = None

class WarmPathTester:
    """Warm Path PostgreSQL testing framework"""

    def __init__(self):
        self.pool = None
        self.test_data = []
        self.performance_metrics = {}

    async def setup(self):
        """Initialize PostgreSQL connection pool and test environment"""
        self.pool = await asyncpg.create_pool(
            host=WARM_PATH_CONFIG["postgres_host"],
            port=WARM_PATH_CONFIG["postgres_port"],
            database=WARM_PATH_CONFIG["postgres_db"],
            user=WARM_PATH_CONFIG["postgres_user"],
            password=WARM_PATH_CONFIG["postgres_password"],
            min_size=5,
            max_size=WARM_PATH_CONFIG["max_connections"]
        )

        # Create test table
        await self.create_test_schema()
        await self.clean_test_data()

    async def teardown(self):
        """Cleanup test environment"""
        if self.pool:
            await self.clean_test_data()
            await self.pool.close()

    async def create_test_schema(self):
        """Create test database schema"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_processed (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    satellite_id VARCHAR(50) NOT NULL,
                    mission_id VARCHAR(50) NOT NULL,
                    telemetry_type VARCHAR(50) NOT NULL,
                    category VARCHAR(50),
                    data_quality_score INTEGER,
                    processing_time TIMESTAMPTZ DEFAULT NOW(),
                    temperature FLOAT,
                    pressure FLOAT,
                    voltage FLOAT,
                    current FLOAT,
                    power FLOAT,
                    altitude FLOAT,
                    velocity FLOAT,
                    status VARCHAR(100),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # Create optimized indexes for telemetry queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp
                ON telemetry_processed(timestamp);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_satellite
                ON telemetry_processed(satellite_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_mission
                ON telemetry_processed(mission_id);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_composite
                ON telemetry_processed(satellite_id, timestamp DESC);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_quality
                ON telemetry_processed(data_quality_score)
                WHERE data_quality_score < 85;
            """)

    async def clean_test_data(self):
        """Remove test data"""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM telemetry_processed WHERE satellite_id LIKE 'TEST-%'")

    def generate_telemetry_record(self, satellite_id: str = None, mission_id: str = None) -> TelemetryRecord:
        """Generate realistic telemetry record for PostgreSQL"""
        satellite_id = satellite_id or f"TEST-SAT-{random.randint(1, 999):03d}"
        mission_id = mission_id or f"TEST-MISSION-{random.choice(['Alpha', 'Beta', 'Gamma'])}"

        telemetry_types = ["sensor", "status", "health", "position", "power", "communication"]
        telemetry_type = random.choice(telemetry_types)

        # Map telemetry type to category
        category_map = {
            "sensor": "environmental",
            "status": "operational",
            "health": "diagnostic",
            "position": "navigation",
            "power": "subsystem",
            "communication": "telemetry"
        }

        return TelemetryRecord(
            timestamp=datetime.utcnow() - timedelta(seconds=random.randint(0, 86400)),
            satellite_id=satellite_id,
            mission_id=mission_id,
            telemetry_type=telemetry_type,
            category=category_map[telemetry_type],
            data_quality_score=random.randint(70, 100),
            temperature=round(random.uniform(-50, 100), 2) if random.random() > 0.3 else None,
            pressure=round(random.uniform(0, 200), 3) if random.random() > 0.4 else None,
            voltage=round(random.uniform(10, 15), 2) if random.random() > 0.2 else None,
            current=round(random.uniform(0, 5), 2) if random.random() > 0.5 else None,
            power=round(random.uniform(100, 1500), 1) if random.random() > 0.4 else None,
            altitude=random.randint(400000, 450000) if telemetry_type == "position" else None,
            velocity=round(random.uniform(27000, 28000), 1) if telemetry_type == "position" else None,
            status=random.choice(["NOMINAL", "WARNING", "CRITICAL", "UNKNOWN"]) if random.random() > 0.6 else None
        )

@pytest.fixture
async def warm_path_tester():
    """Pytest fixture for warm path testing"""
    tester = WarmPathTester()
    await tester.setup()
    yield tester
    await tester.teardown()

class TestWarmPathPerformance:
    """Test warm path performance requirements"""

    @pytest.mark.asyncio
    async def test_single_insert_performance(self, warm_path_tester):
        """Test single record insert performance"""
        record = warm_path_tester.generate_telemetry_record("TEST-PERF-001")

        async with warm_path_tester.pool.acquire() as conn:
            start_time = time.perf_counter()

            await conn.execute("""
                INSERT INTO telemetry_processed
                (timestamp, satellite_id, mission_id, telemetry_type, category,
                 data_quality_score, temperature, pressure, voltage, current,
                 power, altitude, velocity, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """,
            record.timestamp, record.satellite_id, record.mission_id,
            record.telemetry_type, record.category, record.data_quality_score,
            record.temperature, record.pressure, record.voltage, record.current,
            record.power, record.altitude, record.velocity, record.status)

            end_time = time.perf_counter()

        insert_latency_ms = (end_time - start_time) * 1000

        assert insert_latency_ms < WARM_PATH_CONFIG["query_performance_threshold_ms"], \
            f"Insert latency {insert_latency_ms:.3f}ms exceeds threshold"

        print(f"Single insert latency: {insert_latency_ms:.3f}ms")

    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, warm_path_tester):
        """Test bulk insert throughput"""
        num_records = 1000
        records = []

        # Generate test records
        for i in range(num_records):
            record = warm_path_tester.generate_telemetry_record(f"TEST-BULK-{i:04d}")
            records.append((
                record.timestamp, record.satellite_id, record.mission_id,
                record.telemetry_type, record.category, record.data_quality_score,
                record.temperature, record.pressure, record.voltage, record.current,
                record.power, record.altitude, record.velocity, record.status
            ))

        # Measure bulk insert performance
        async with warm_path_tester.pool.acquire() as conn:
            start_time = time.perf_counter()

            await conn.executemany("""
                INSERT INTO telemetry_processed
                (timestamp, satellite_id, mission_id, telemetry_type, category,
                 data_quality_score, temperature, pressure, voltage, current,
                 power, altitude, velocity, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """, records)

            end_time = time.perf_counter()

        duration_seconds = end_time - start_time
        throughput = num_records / duration_seconds

        # Target at least 5K records/sec for test environment
        min_throughput = 5000
        assert throughput >= min_throughput, \
            f"Bulk insert throughput {throughput:.0f} records/sec below minimum {min_throughput}"

        print(f"Bulk insert throughput: {throughput:.0f} records/sec")

    @pytest.mark.asyncio
    async def test_query_performance(self, warm_path_tester):
        """Test analytical query performance"""
        # Insert test data for queries
        num_records = 500
        satellites = [f"TEST-QUERY-{i:03d}" for i in range(10)]
        missions = ["TEST-MISSION-Alpha", "TEST-MISSION-Beta"]

        records = []
        for i in range(num_records):
            satellite_id = random.choice(satellites)
            mission_id = random.choice(missions)
            record = warm_path_tester.generate_telemetry_record(satellite_id, mission_id)
            records.append((
                record.timestamp, record.satellite_id, record.mission_id,
                record.telemetry_type, record.category, record.data_quality_score,
                record.temperature, record.pressure, record.voltage, record.current,
                record.power, record.altitude, record.velocity, record.status
            ))

        async with warm_path_tester.pool.acquire() as conn:
            # Bulk insert test data
            await conn.executemany("""
                INSERT INTO telemetry_processed
                (timestamp, satellite_id, mission_id, telemetry_type, category,
                 data_quality_score, temperature, pressure, voltage, current,
                 power, altitude, velocity, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """, records)

            # Test various analytical queries
            test_queries = [
                # Latest telemetry by satellite
                ("Latest telemetry query", """
                    SELECT DISTINCT ON (satellite_id)
                           satellite_id, timestamp, temperature, voltage, status
                    FROM telemetry_processed
                    WHERE satellite_id LIKE 'TEST-QUERY-%'
                    ORDER BY satellite_id, timestamp DESC
                """),

                # Mission summary statistics
                ("Mission statistics", """
                    SELECT mission_id,
                           COUNT(*) as message_count,
                           AVG(data_quality_score) as avg_quality,
                           AVG(temperature) as avg_temp,
                           MAX(timestamp) as last_seen
                    FROM telemetry_processed
                    WHERE mission_id LIKE 'TEST-MISSION-%'
                    GROUP BY mission_id
                """),

                # Time-based trend analysis
                ("Hourly trend analysis", """
                    SELECT date_trunc('hour', timestamp) as hour,
                           satellite_id,
                           AVG(temperature) as avg_temp,
                           AVG(voltage) as avg_voltage,
                           COUNT(*) as message_count
                    FROM telemetry_processed
                    WHERE satellite_id LIKE 'TEST-QUERY-%'
                      AND timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY hour, satellite_id
                    ORDER BY hour DESC, satellite_id
                """),

                # Quality analysis
                ("Data quality analysis", """
                    SELECT category,
                           COUNT(*) as total_messages,
                           COUNT(CASE WHEN data_quality_score >= 95 THEN 1 END) as excellent,
                           COUNT(CASE WHEN data_quality_score BETWEEN 85 AND 94 THEN 1 END) as good,
                           COUNT(CASE WHEN data_quality_score < 85 THEN 1 END) as poor,
                           AVG(data_quality_score) as avg_quality
                    FROM telemetry_processed
                    WHERE satellite_id LIKE 'TEST-QUERY-%'
                    GROUP BY category
                """)
            ]

            query_results = {}

            for query_name, query_sql in test_queries:
                start_time = time.perf_counter()
                result = await conn.fetch(query_sql)
                end_time = time.perf_counter()

                query_latency_ms = (end_time - start_time) * 1000
                query_results[query_name] = {
                    "latency_ms": query_latency_ms,
                    "row_count": len(result)
                }

                # Performance assertion
                assert query_latency_ms < WARM_PATH_CONFIG["query_performance_threshold_ms"], \
                    f"{query_name} latency {query_latency_ms:.3f}ms exceeds threshold"

                print(f"{query_name}: {query_latency_ms:.3f}ms ({len(result)} rows)")

class TestWarmPathAnalytics:
    """Test analytical capabilities of warm path"""

    @pytest.mark.asyncio
    async def test_telemetry_trend_analysis(self, warm_path_tester):
        """Test telemetry trend analysis over time"""
        satellite_id = "TEST-TREND-SAT"
        mission_id = "TEST-TREND-MISSION"

        # Generate time-series data with trends
        base_time = datetime.utcnow() - timedelta(hours=24)
        records = []

        for i in range(100):
            # Create trending data (temperature increasing, voltage decreasing)
            timestamp = base_time + timedelta(minutes=i * 15)  # 15-minute intervals

            record = TelemetryRecord(
                timestamp=timestamp,
                satellite_id=satellite_id,
                mission_id=mission_id,
                telemetry_type="sensor",
                category="environmental",
                data_quality_score=random.randint(90, 100),
                temperature=20.0 + (i * 0.5) + random.uniform(-2, 2),  # Trending up
                voltage=13.0 - (i * 0.01) + random.uniform(-0.1, 0.1),  # Trending down
                pressure=101.3 + random.uniform(-5, 5)
            )

            records.append((
                record.timestamp, record.satellite_id, record.mission_id,
                record.telemetry_type, record.category, record.data_quality_score,
                record.temperature, record.pressure, record.voltage, record.current,
                record.power, record.altitude, record.velocity, record.status
            ))

        async with warm_path_tester.pool.acquire() as conn:
            # Insert trend data
            await conn.executemany("""
                INSERT INTO telemetry_processed
                (timestamp, satellite_id, mission_id, telemetry_type, category,
                 data_quality_score, temperature, pressure, voltage, current,
                 power, altitude, velocity, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """, records)

            # Analyze temperature trend
            temp_trend = await conn.fetch("""
                SELECT timestamp, temperature,
                       LAG(temperature) OVER (ORDER BY timestamp) as prev_temp,
                       temperature - LAG(temperature) OVER (ORDER BY timestamp) as temp_change
                FROM telemetry_processed
                WHERE satellite_id = $1 AND temperature IS NOT NULL
                ORDER BY timestamp
            """, satellite_id)

            # Analyze voltage trend
            voltage_trend = await conn.fetch("""
                SELECT timestamp, voltage,
                       LAG(voltage) OVER (ORDER BY timestamp) as prev_voltage,
                       voltage - LAG(voltage) OVER (ORDER BY timestamp) as voltage_change
                FROM telemetry_processed
                WHERE satellite_id = $1 AND voltage IS NOT NULL
                ORDER BY timestamp
            """, satellite_id)

            # Calculate trend statistics
            temp_changes = [row['temp_change'] for row in temp_trend if row['temp_change'] is not None]
            voltage_changes = [row['voltage_change'] for row in voltage_trend if row['voltage_change'] is not None]

            avg_temp_change = statistics.mean(temp_changes) if temp_changes else 0
            avg_voltage_change = statistics.mean(voltage_changes) if voltage_changes else 0

            # Verify trends
            assert len(temp_trend) >= 95, "Insufficient temperature data points"
            assert len(voltage_trend) >= 95, "Insufficient voltage data points"

            # Temperature should be trending up (positive average change)
            assert avg_temp_change > 0.2, f"Temperature trend {avg_temp_change:.3f} not increasing as expected"

            # Voltage should be trending down (negative average change)
            assert avg_voltage_change < -0.005, f"Voltage trend {avg_voltage_change:.3f} not decreasing as expected"

            print(f"Temperature trend: {avg_temp_change:.3f}°C per interval")
            print(f"Voltage trend: {avg_voltage_change:.4f}V per interval")

    @pytest.mark.asyncio
    async def test_anomaly_detection_queries(self, warm_path_tester):
        """Test anomaly detection through SQL queries"""
        satellite_id = "TEST-ANOMALY-SAT"

        # Insert normal and anomalous data
        normal_records = []
        anomaly_records = []

        base_time = datetime.utcnow() - timedelta(hours=1)

        # Normal data (95% of records)
        for i in range(95):
            timestamp = base_time + timedelta(seconds=i * 30)
            record = TelemetryRecord(
                timestamp=timestamp,
                satellite_id=satellite_id,
                mission_id="TEST-ANOMALY-MISSION",
                telemetry_type="sensor",
                category="environmental",
                data_quality_score=random.randint(90, 100),
                temperature=random.uniform(18, 25),  # Normal range
                voltage=random.uniform(12.5, 13.5),  # Normal range
                pressure=random.uniform(98, 104)     # Normal range
            )

            normal_records.append((
                record.timestamp, record.satellite_id, record.mission_id,
                record.telemetry_type, record.category, record.data_quality_score,
                record.temperature, record.pressure, record.voltage, record.current,
                record.power, record.altitude, record.velocity, record.status
            ))

        # Anomalous data (5% of records)
        anomaly_times = [base_time + timedelta(seconds=i * 30) for i in [10, 25, 50, 78, 89]]
        for timestamp in anomaly_times:
            record = TelemetryRecord(
                timestamp=timestamp,
                satellite_id=satellite_id,
                mission_id="TEST-ANOMALY-MISSION",
                telemetry_type="sensor",
                category="environmental",
                data_quality_score=random.randint(60, 85),  # Lower quality
                temperature=random.choice([45, -10, 60]),   # Extreme temperatures
                voltage=random.choice([10.0, 15.5]),       # Out of range voltages
                pressure=random.choice([50, 150])          # Extreme pressures
            )

            anomaly_records.append((
                record.timestamp, record.satellite_id, record.mission_id,
                record.telemetry_type, record.category, record.data_quality_score,
                record.temperature, record.pressure, record.voltage, record.current,
                record.power, record.altitude, record.velocity, record.status
            ))

        async with warm_path_tester.pool.acquire() as conn:
            # Insert all test data
            all_records = normal_records + anomaly_records
            await conn.executemany("""
                INSERT INTO telemetry_processed
                (timestamp, satellite_id, mission_id, telemetry_type, category,
                 data_quality_score, temperature, pressure, voltage, current,
                 power, altitude, velocity, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """, all_records)

            # Statistical anomaly detection queries
            anomalies = await conn.fetch("""
                WITH stats AS (
                    SELECT
                        AVG(temperature) as avg_temp,
                        STDDEV(temperature) as stddev_temp,
                        AVG(voltage) as avg_voltage,
                        STDDEV(voltage) as stddev_voltage,
                        AVG(pressure) as avg_pressure,
                        STDDEV(pressure) as stddev_pressure
                    FROM telemetry_processed
                    WHERE satellite_id = $1
                )
                SELECT
                    timestamp, temperature, voltage, pressure, data_quality_score,
                    ABS(temperature - stats.avg_temp) / stats.stddev_temp as temp_z_score,
                    ABS(voltage - stats.avg_voltage) / stats.stddev_voltage as voltage_z_score,
                    ABS(pressure - stats.avg_pressure) / stats.stddev_pressure as pressure_z_score
                FROM telemetry_processed, stats
                WHERE satellite_id = $1
                  AND (
                      ABS(temperature - stats.avg_temp) / stats.stddev_temp > 2.0 OR
                      ABS(voltage - stats.avg_voltage) / stats.stddev_voltage > 2.0 OR
                      ABS(pressure - stats.avg_pressure) / stats.stddev_pressure > 2.0 OR
                      data_quality_score < 90
                  )
                ORDER BY timestamp
            """, satellite_id)

            # Verify anomaly detection
            assert len(anomalies) >= 3, f"Expected at least 3 anomalies, found {len(anomalies)}"

            # Check that detected anomalies have high Z-scores or low quality
            for anomaly in anomalies:
                z_scores = [anomaly['temp_z_score'], anomaly['voltage_z_score'], anomaly['pressure_z_score']]
                max_z_score = max(z for z in z_scores if z is not None)

                assert (max_z_score > 2.0 or anomaly['data_quality_score'] < 90), \
                    f"Anomaly detection false positive: Z-score {max_z_score:.2f}, Quality {anomaly['data_quality_score']}"

            print(f"Detected {len(anomalies)} anomalies from {len(all_records)} records")

class TestWarmPathReliability:
    """Test warm path reliability and data integrity"""

    @pytest.mark.asyncio
    async def test_transaction_integrity(self, warm_path_tester):
        """Test transaction rollback and data consistency"""
        satellite_id = "TEST-TRANSACTION-SAT"

        async with warm_path_tester.pool.acquire() as conn:
            # Test successful transaction
            async with conn.transaction():
                record1 = warm_path_tester.generate_telemetry_record(satellite_id)
                record2 = warm_path_tester.generate_telemetry_record(satellite_id)

                await conn.execute("""
                    INSERT INTO telemetry_processed
                    (timestamp, satellite_id, mission_id, telemetry_type, category, data_quality_score)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, record1.timestamp, record1.satellite_id, record1.mission_id,
                    record1.telemetry_type, record1.category, record1.data_quality_score)

                await conn.execute("""
                    INSERT INTO telemetry_processed
                    (timestamp, satellite_id, mission_id, telemetry_type, category, data_quality_score)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, record2.timestamp, record2.satellite_id, record2.mission_id,
                    record2.telemetry_type, record2.category, record2.data_quality_score)

            # Verify successful transaction
            count_after_success = await conn.fetchval(
                "SELECT COUNT(*) FROM telemetry_processed WHERE satellite_id = $1",
                satellite_id
            )
            assert count_after_success == 2

            # Test failed transaction (should rollback)
            try:
                async with conn.transaction():
                    record3 = warm_path_tester.generate_telemetry_record(satellite_id)

                    await conn.execute("""
                        INSERT INTO telemetry_processed
                        (timestamp, satellite_id, mission_id, telemetry_type, category, data_quality_score)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """, record3.timestamp, record3.satellite_id, record3.mission_id,
                        record3.telemetry_type, record3.category, record3.data_quality_score)

                    # Force an error to trigger rollback
                    await conn.execute("SELECT 1/0")  # Division by zero error

            except Exception:
                # Expected to fail
                pass

            # Verify rollback occurred
            count_after_rollback = await conn.fetchval(
                "SELECT COUNT(*) FROM telemetry_processed WHERE satellite_id = $1",
                satellite_id
            )
            assert count_after_rollback == 2, "Transaction rollback failed"

    @pytest.mark.asyncio
    async def test_concurrent_access(self, warm_path_tester):
        """Test concurrent database access"""
        num_concurrent = 10
        records_per_worker = 20

        async def insert_worker(worker_id):
            """Worker function for concurrent inserts"""
            inserted_ids = []

            async with warm_path_tester.pool.acquire() as conn:
                for i in range(records_per_worker):
                    record = warm_path_tester.generate_telemetry_record(f"TEST-CONCURRENT-{worker_id}-{i}")

                    result = await conn.fetchval("""
                        INSERT INTO telemetry_processed
                        (timestamp, satellite_id, mission_id, telemetry_type, category, data_quality_score)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        RETURNING id
                    """, record.timestamp, record.satellite_id, record.mission_id,
                        record.telemetry_type, record.category, record.data_quality_score)

                    inserted_ids.append(result)

            return inserted_ids

        # Run concurrent insertions
        tasks = [insert_worker(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)

        # Verify all inserts succeeded
        all_ids = [id for worker_results in results for id in worker_results]
        expected_count = num_concurrent * records_per_worker

        assert len(all_ids) == expected_count
        assert len(set(all_ids)) == expected_count, "Duplicate IDs detected - concurrency issue"

        # Verify data integrity
        async with warm_path_tester.pool.acquire() as conn:
            actual_count = await conn.fetchval(
                "SELECT COUNT(*) FROM telemetry_processed WHERE satellite_id LIKE 'TEST-CONCURRENT-%'"
            )
            assert actual_count == expected_count

        print(f"Concurrent access test: {expected_count} records inserted successfully")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
