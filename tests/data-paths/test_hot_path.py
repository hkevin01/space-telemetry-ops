"""
Hot Path Data Tests - Redis Real-time Telemetry Processing

Tests for sub-millisecond telemetry data access and real-time processing
through the Redis hot path. Validates performance, reliability, and
mission-critical real-time operations.
"""

import asyncio
import pytest
import redis.asyncio as redis
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random
import string

# Test configuration
HOT_PATH_CONFIG = {
    "redis_host": "localhost",
    "redis_port": 6379,
    "redis_db": 0,
    "performance_threshold_ms": 1.0,  # <1ms response time requirement
    "throughput_target": 50000,  # 50K msgs/sec target
    "retention_minutes": 15,  # Hot path retention
    "max_connections": 100
}

class HotPathTester:
    """Hot Path Redis testing framework"""
    
    def __init__(self):
        self.redis_client = None
        self.performance_metrics = {}
        self.test_data = []
    
    async def setup(self):
        """Initialize Redis connection and test environment"""
        self.redis_client = redis.Redis(
            host=HOT_PATH_CONFIG["redis_host"],
            port=HOT_PATH_CONFIG["redis_port"],
            db=HOT_PATH_CONFIG["redis_db"],
            decode_responses=True
        )
        
        # Test Redis connectivity
        await self.redis_client.ping()
        
        # Clear test data
        await self.redis_client.flushdb()
        
    async def teardown(self):
        """Cleanup test environment"""
        if self.redis_client:
            # Clean up test keys
            keys = await self.redis_client.keys("test:*")
            if keys:
                await self.redis_client.delete(*keys)
            await self.redis_client.close()
    
    def generate_telemetry_packet(self, satellite_id: str = None, mission_id: str = None) -> Dict[str, Any]:
        """Generate realistic telemetry data packet"""
        satellite_id = satellite_id or f"SAT-{random.randint(1, 999):03d}"
        mission_id = mission_id or f"MISSION-{random.choice(['Alpha', 'Beta', 'Gamma', 'Delta'])}"
        
        return {
            "timestamp": int(time.time() * 1000),  # Millisecond precision
            "satelliteId": satellite_id,
            "missionId": mission_id,
            "telemetryType": random.choice(["sensor", "status", "health", "position", "power"]),
            "data": {
                "temperature": round(random.uniform(-50, 100), 2),
                "pressure": round(random.uniform(0, 200), 3),
                "voltage": round(random.uniform(10, 15), 2),
                "current": round(random.uniform(0, 5), 2),
                "altitude": random.randint(400000, 450000),
                "velocity": round(random.uniform(27000, 28000), 1),
                "orientation": {
                    "pitch": round(random.uniform(-180, 180), 2),
                    "roll": round(random.uniform(-180, 180), 2),
                    "yaw": round(random.uniform(-180, 180), 2)
                }
            },
            "quality": random.randint(85, 100),
            "sequence": random.randint(1, 1000000)
        }

@pytest.fixture
async def hot_path_tester():
    """Pytest fixture for hot path testing"""
    tester = HotPathTester()
    await tester.setup()
    yield tester
    await tester.teardown()

class TestHotPathPerformance:
    """Test hot path performance requirements"""
    
    @pytest.mark.asyncio
    async def test_single_write_performance(self, hot_path_tester):
        """Test single telemetry write performance (<1ms)"""
        telemetry = hot_path_tester.generate_telemetry_packet("SAT-001", "MISSION-Alpha")
        key = f"telemetry:{telemetry['satelliteId']}:latest"
        
        # Measure write performance
        start_time = time.perf_counter()
        await hot_path_tester.redis_client.set(
            key, 
            json.dumps(telemetry),
            ex=HOT_PATH_CONFIG["retention_minutes"] * 60  # TTL in seconds
        )
        end_time = time.perf_counter()
        
        write_latency_ms = (end_time - start_time) * 1000
        
        assert write_latency_ms < HOT_PATH_CONFIG["performance_threshold_ms"], \
            f"Write latency {write_latency_ms:.3f}ms exceeds threshold {HOT_PATH_CONFIG['performance_threshold_ms']}ms"
        
        # Verify data integrity
        stored_data = await hot_path_tester.redis_client.get(key)
        assert stored_data is not None
        parsed_data = json.loads(stored_data)
        assert parsed_data["satelliteId"] == telemetry["satelliteId"]
        assert parsed_data["timestamp"] == telemetry["timestamp"]
    
    @pytest.mark.asyncio
    async def test_single_read_performance(self, hot_path_tester):
        """Test single telemetry read performance (<1ms)"""
        telemetry = hot_path_tester.generate_telemetry_packet("SAT-002", "MISSION-Beta")
        key = f"telemetry:{telemetry['satelliteId']}:latest"
        
        # Store test data
        await hot_path_tester.redis_client.set(key, json.dumps(telemetry))
        
        # Measure read performance
        start_time = time.perf_counter()
        data = await hot_path_tester.redis_client.get(key)
        end_time = time.perf_counter()
        
        read_latency_ms = (end_time - start_time) * 1000
        
        assert read_latency_ms < HOT_PATH_CONFIG["performance_threshold_ms"], \
            f"Read latency {read_latency_ms:.3f}ms exceeds threshold {HOT_PATH_CONFIG['performance_threshold_ms']}ms"
        
        # Verify data integrity
        assert data is not None
        parsed_data = json.loads(data)
        assert parsed_data["satelliteId"] == telemetry["satelliteId"]
    
    @pytest.mark.asyncio
    async def test_bulk_write_throughput(self, hot_path_tester):
        """Test bulk write throughput (50K+ msgs/sec target)"""
        num_messages = 1000  # Reduced for test environment
        telemetry_batch = []
        
        # Generate test batch
        for i in range(num_messages):
            telemetry = hot_path_tester.generate_telemetry_packet(f"SAT-{i:03d}")
            telemetry_batch.append(telemetry)
        
        # Measure bulk write performance
        start_time = time.perf_counter()
        
        # Use pipeline for bulk operations
        pipe = hot_path_tester.redis_client.pipeline()
        for telemetry in telemetry_batch:
            key = f"telemetry:{telemetry['satelliteId']}:latest"
            pipe.set(key, json.dumps(telemetry), ex=900)  # 15 min TTL
        
        await pipe.execute()
        end_time = time.perf_counter()
        
        duration_seconds = end_time - start_time
        throughput = num_messages / duration_seconds
        
        # For test environment, expect at least 10K msgs/sec
        min_throughput = 10000
        assert throughput >= min_throughput, \
            f"Throughput {throughput:.0f} msgs/sec below minimum {min_throughput} msgs/sec"
        
        print(f"Bulk write throughput: {throughput:.0f} msgs/sec")
    
    @pytest.mark.asyncio
    async def test_concurrent_access_performance(self, hot_path_tester):
        """Test concurrent read/write performance"""
        num_concurrent = 50
        operations_per_coroutine = 20
        
        async def write_worker(worker_id):
            """Worker coroutine for concurrent writes"""
            latencies = []
            for i in range(operations_per_coroutine):
                telemetry = hot_path_tester.generate_telemetry_packet(f"SAT-W{worker_id}-{i}")
                key = f"telemetry:{telemetry['satelliteId']}:latest"
                
                start_time = time.perf_counter()
                await hot_path_tester.redis_client.set(key, json.dumps(telemetry))
                end_time = time.perf_counter()
                
                latencies.append((end_time - start_time) * 1000)
            
            return latencies
        
        async def read_worker(worker_id):
            """Worker coroutine for concurrent reads"""
            latencies = []
            # First, ensure we have data to read
            for i in range(operations_per_coroutine):
                telemetry = hot_path_tester.generate_telemetry_packet(f"SAT-R{worker_id}-{i}")
                key = f"telemetry:{telemetry['satelliteId']}:latest"
                await hot_path_tester.redis_client.set(key, json.dumps(telemetry))
            
            # Then read the data
            for i in range(operations_per_coroutine):
                key = f"telemetry:SAT-R{worker_id}-{i}:latest"
                
                start_time = time.perf_counter()
                await hot_path_tester.redis_client.get(key)
                end_time = time.perf_counter()
                
                latencies.append((end_time - start_time) * 1000)
            
            return latencies
        
        # Run concurrent operations
        start_time = time.perf_counter()
        
        write_tasks = [write_worker(i) for i in range(num_concurrent // 2)]
        read_tasks = [read_worker(i) for i in range(num_concurrent // 2)]
        
        write_results = await asyncio.gather(*write_tasks)
        read_results = await asyncio.gather(*read_tasks)
        
        end_time = time.perf_counter()
        
        # Analyze results
        all_write_latencies = [lat for result in write_results for lat in result]
        all_read_latencies = [lat for result in read_results for lat in result]
        
        avg_write_latency = sum(all_write_latencies) / len(all_write_latencies)
        avg_read_latency = sum(all_read_latencies) / len(all_read_latencies)
        max_write_latency = max(all_write_latencies)
        max_read_latency = max(all_read_latencies)
        
        total_operations = len(all_write_latencies) + len(all_read_latencies)
        duration = end_time - start_time
        throughput = total_operations / duration
        
        # Performance assertions
        assert avg_write_latency < 5.0, f"Average write latency {avg_write_latency:.2f}ms too high"
        assert avg_read_latency < 5.0, f"Average read latency {avg_read_latency:.2f}ms too high"
        assert max_write_latency < 20.0, f"Max write latency {max_write_latency:.2f}ms too high"
        assert max_read_latency < 20.0, f"Max read latency {max_read_latency:.2f}ms too high"
        
        print(f"Concurrent throughput: {throughput:.0f} ops/sec")
        print(f"Avg write latency: {avg_write_latency:.2f}ms, max: {max_write_latency:.2f}ms")
        print(f"Avg read latency: {avg_read_latency:.2f}ms, max: {max_read_latency:.2f}ms")

class TestHotPathReliability:
    """Test hot path reliability and fault tolerance"""
    
    @pytest.mark.asyncio
    async def test_data_expiration(self, hot_path_tester):
        """Test TTL (Time To Live) functionality"""
        telemetry = hot_path_tester.generate_telemetry_packet("SAT-TTL", "MISSION-TTL")
        key = f"telemetry:{telemetry['satelliteId']}:latest"
        
        # Store with 2 second TTL
        await hot_path_tester.redis_client.set(key, json.dumps(telemetry), ex=2)
        
        # Verify data exists
        data = await hot_path_tester.redis_client.get(key)
        assert data is not None
        
        # Check TTL
        ttl = await hot_path_tester.redis_client.ttl(key)
        assert ttl <= 2
        assert ttl > 0
        
        # Wait for expiration and verify cleanup
        await asyncio.sleep(3)
        data = await hot_path_tester.redis_client.get(key)
        assert data is None
    
    @pytest.mark.asyncio
    async def test_memory_optimization(self, hot_path_tester):
        """Test memory usage optimization"""
        # Store many keys and monitor memory usage
        num_keys = 1000
        base_memory = await hot_path_tester.redis_client.info('memory')
        base_used = base_memory['used_memory']
        
        # Store telemetry data
        for i in range(num_keys):
            telemetry = hot_path_tester.generate_telemetry_packet(f"SAT-MEM-{i:04d}")
            key = f"telemetry:{telemetry['satelliteId']}:latest"
            await hot_path_tester.redis_client.set(key, json.dumps(telemetry), ex=3600)
        
        # Check memory usage
        current_memory = await hot_path_tester.redis_client.info('memory')
        current_used = current_memory['used_memory']
        memory_increase = current_used - base_used
        
        # Average memory per key should be reasonable
        avg_memory_per_key = memory_increase / num_keys
        max_memory_per_key = 2048  # 2KB per telemetry packet
        
        assert avg_memory_per_key < max_memory_per_key, \
            f"Average memory per key {avg_memory_per_key} bytes exceeds limit {max_memory_per_key} bytes"
        
        print(f"Memory usage: {memory_increase} bytes for {num_keys} keys")
        print(f"Average per key: {avg_memory_per_key:.1f} bytes")

class TestHotPathRealTimeFeatures:
    """Test real-time features specific to hot path"""
    
    @pytest.mark.asyncio
    async def test_pub_sub_notifications(self, hot_path_tester):
        """Test Redis pub/sub for real-time notifications"""
        channel = "telemetry:alerts:critical"
        received_messages = []
        
        # Create subscriber
        pubsub = hot_path_tester.redis_client.pubsub()
        await pubsub.subscribe(channel)
        
        async def message_handler():
            """Handle incoming pub/sub messages"""
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    received_messages.append(json.loads(message['data']))
                    if len(received_messages) >= 3:
                        break
        
        # Start message handler
        handler_task = asyncio.create_task(message_handler())
        
        # Give subscriber time to connect
        await asyncio.sleep(0.1)
        
        # Publish test alerts
        alerts = [
            {"type": "CRITICAL", "satellite": "SAT-001", "message": "Temperature critical"},
            {"type": "WARNING", "satellite": "SAT-002", "message": "Voltage low"},
            {"type": "EMERGENCY", "satellite": "SAT-003", "message": "Communication lost"}
        ]
        
        for alert in alerts:
            await hot_path_tester.redis_client.publish(channel, json.dumps(alert))
            await asyncio.sleep(0.05)  # Small delay between messages
        
        # Wait for messages to be received
        await asyncio.wait_for(handler_task, timeout=5.0)
        
        # Verify all messages were received
        assert len(received_messages) == 3
        
        # Verify message content
        for i, alert in enumerate(alerts):
            assert received_messages[i]["type"] == alert["type"]
            assert received_messages[i]["satellite"] == alert["satellite"]
        
        await pubsub.unsubscribe(channel)
        await pubsub.close()
    
    @pytest.mark.asyncio
    async def test_atomic_operations(self, hot_path_tester):
        """Test atomic operations for data consistency"""
        satellite_id = "SAT-ATOMIC"
        counter_key = f"telemetry:{satellite_id}:message_count"
        latest_key = f"telemetry:{satellite_id}:latest"
        
        # Test atomic increment with telemetry update
        telemetry = hot_path_tester.generate_telemetry_packet(satellite_id)
        
        # Use transaction for atomic operations
        async with hot_path_tester.redis_client.pipeline(transaction=True) as pipe:
            # Increment message counter
            await pipe.incr(counter_key)
            # Update latest telemetry
            await pipe.set(latest_key, json.dumps(telemetry))
            # Set expiration on both keys
            await pipe.expire(counter_key, 3600)
            await pipe.expire(latest_key, 900)
            
            results = await pipe.execute()
        
        # Verify atomic operation results
        assert results[0] == 1  # Counter incremented to 1
        assert results[1] is True  # Set operation successful
        assert results[2] is True  # Expire set successfully
        assert results[3] is True  # Expire set successfully
        
        # Verify final state
        count = await hot_path_tester.redis_client.get(counter_key)
        latest = await hot_path_tester.redis_client.get(latest_key)
        
        assert int(count) == 1
        assert latest is not None
        
        parsed_telemetry = json.loads(latest)
        assert parsed_telemetry["satelliteId"] == satellite_id

class TestHotPathScenarios:
    """Test real-world operational scenarios"""
    
    @pytest.mark.asyncio
    async def test_mission_critical_scenario(self, hot_path_tester):
        """Test mission-critical real-time telemetry scenario"""
        # Simulate ISS docking procedure with high-frequency telemetry
        mission_id = "ISS-DOCKING-001"
        spacecraft_id = "DRAGON-CAPSULE"
        
        # High-frequency telemetry (10 Hz for 30 seconds simulation)
        simulation_duration = 2  # Reduced for testing
        frequency_hz = 10
        total_packets = simulation_duration * frequency_hz
        
        start_time = time.perf_counter()
        latencies = []
        
        for i in range(total_packets):
            # Critical docking telemetry
            telemetry = {
                "timestamp": int(time.time() * 1000),
                "satelliteId": spacecraft_id,
                "missionId": mission_id,
                "telemetryType": "docking_critical",
                "data": {
                    "relative_position": {
                        "x": round(random.uniform(-2, 2), 3),  # meters
                        "y": round(random.uniform(-2, 2), 3),
                        "z": round(random.uniform(0, 10), 3)
                    },
                    "relative_velocity": {
                        "x": round(random.uniform(-0.1, 0.1), 4),  # m/s
                        "y": round(random.uniform(-0.1, 0.1), 4),
                        "z": round(random.uniform(-0.5, 0), 4)
                    },
                    "attitude": {
                        "pitch": round(random.uniform(-5, 5), 2),
                        "roll": round(random.uniform(-5, 5), 2),
                        "yaw": round(random.uniform(-5, 5), 2)
                    },
                    "docking_port_pressure": round(random.uniform(14.5, 14.7), 2)
                },
                "critical": True,
                "sequence": i
            }
            
            # Store with priority (shorter TTL for frequent updates)
            key = f"telemetry:{spacecraft_id}:docking_latest"
            
            op_start = time.perf_counter()
            await hot_path_tester.redis_client.set(key, json.dumps(telemetry), ex=60)
            op_end = time.perf_counter()
            
            latencies.append((op_end - op_start) * 1000)
            
            # Simulate real-time frequency
            await asyncio.sleep(1.0 / frequency_hz)
        
        end_time = time.perf_counter()
        
        # Analyze performance
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        total_duration = end_time - start_time
        actual_frequency = total_packets / total_duration
        
        # Mission-critical performance requirements
        assert avg_latency < 1.0, f"Average latency {avg_latency:.3f}ms exceeds 1ms limit"
        assert max_latency < 10.0, f"Max latency {max_latency:.3f}ms exceeds 10ms limit"
        assert p95_latency < 2.0, f"P95 latency {p95_latency:.3f}ms exceeds 2ms limit"
        
        print(f"Mission-critical scenario completed:")
        print(f"  Packets processed: {total_packets}")
        print(f"  Actual frequency: {actual_frequency:.1f} Hz")
        print(f"  Avg latency: {avg_latency:.3f}ms")
        print(f"  P95 latency: {p95_latency:.3f}ms")
        print(f"  Max latency: {max_latency:.3f}ms")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
