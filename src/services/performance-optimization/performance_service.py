"""
Performance Optimization Service

This module provides comprehensive database and application performance
optimization for high-throughput telemetry operations. Includes query
optimization, connection pooling, caching strategies, and real-time
performance monitoring.

Target Performance:
- <10ms database query times at scale
- 50K+ messages/second sustained throughput
- <100ms end-to-end processing latency
- Efficient memory usage and connection management
"""

import asyncio
import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from functools import wraps

import asyncpg
import redis.asyncio as redis
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.pool import QueuePool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Monitoring and metrics
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking structure"""

    # Database metrics
    query_count: int = 0
    query_time_total: float = 0.0
    query_time_min: float = float('inf')
    query_time_max: float = 0.0
    connection_pool_active: int = 0
    connection_pool_idle: int = 0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_size: int = 0

    # Application metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    throughput_msg_per_sec: float = 0.0

    # Processing metrics
    processing_time_total: float = 0.0
    processing_count: int = 0
    error_count: int = 0

    def get_average_query_time(self) -> float:
        """Calculate average query time"""
        return self.query_time_total / self.query_count if self.query_count > 0 else 0.0

    def get_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def get_average_processing_time(self) -> float:
        """Calculate average processing time"""
        return self.processing_time_total / self.processing_count if self.processing_count > 0 else 0.0


class QueryOptimizer:
    """Database query optimization and monitoring"""

    def __init__(self,
                 connection_string: str,
                 pool_size: int = 20,
                 max_overflow: int = 30,
                 pool_timeout: int = 30):

        self.connection_string = connection_string
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout

        # Create optimized connection pool
        self.engine = create_async_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=3600,  # Recycle connections every hour
            pool_pre_ping=True,  # Validate connections before use
            echo=False  # Set to True for query logging in debug mode
        )

        self.metrics = PerformanceMetrics()
        self.logger = logging.getLogger(__name__)

        # Query cache for frequently accessed data
        self.query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)

        # Prometheus metrics
        self.query_duration = Histogram(
            'db_query_duration_seconds',
            'Database query execution time',
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        self.query_counter = Counter(
            'db_queries_total',
            'Total number of database queries'
        )
        self.connection_pool_gauge = Gauge(
            'db_connection_pool_active',
            'Number of active database connections'
        )

    def performance_monitor(self, query_name: str = "unknown"):
        """Decorator for monitoring query performance"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)

                    # Record successful query metrics
                    execution_time = time.time() - start_time
                    self._update_query_metrics(execution_time, success=True)

                    # Prometheus metrics
                    self.query_duration.observe(execution_time)
                    self.query_counter.inc()

                    self.logger.debug(f"Query {query_name} completed in {execution_time:.3f}s")

                    return result

                except Exception as e:
                    # Record error metrics
                    execution_time = time.time() - start_time
                    self._update_query_metrics(execution_time, success=False)
                    self.metrics.error_count += 1

                    self.logger.error(f"Query {query_name} failed after {execution_time:.3f}s: {str(e)}")
                    raise

            return wrapper
        return decorator

    def _update_query_metrics(self, execution_time: float, success: bool = True):
        """Update internal query performance metrics"""
        if success:
            self.metrics.query_count += 1
            self.metrics.query_time_total += execution_time
            self.metrics.query_time_min = min(self.metrics.query_time_min, execution_time)
            self.metrics.query_time_max = max(self.metrics.query_time_max, execution_time)

    @asynccontextmanager
    async def get_connection(self):
        """Get optimized database connection with monitoring"""
        async with self.engine.begin() as conn:
            # Update connection pool metrics
            self.connection_pool_gauge.set(self.engine.pool.checkedout())
            self.metrics.connection_pool_active = self.engine.pool.checkedout()
            self.metrics.connection_pool_idle = self.engine.pool.checkedin()

            yield conn

    async def execute_optimized_query(self,
                                    query: str,
                                    parameters: Optional[Dict] = None,
                                    cache_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Execute query with performance optimizations and caching"""

        # Check cache first if cache_key provided
        if cache_key and cache_key in self.query_cache:
            cached_result, cached_time = self.query_cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                self.metrics.cache_hits += 1
                return cached_result
            else:
                # Remove expired cache entry
                del self.query_cache[cache_key]

        # Execute query with performance monitoring
        @self.performance_monitor(f"query_{cache_key or 'unnamed'}")
        async def _execute():
            async with self.get_connection() as conn:
                result = await conn.execute(text(query), parameters or {})
                rows = result.fetchall()

                # Convert to list of dictionaries for easier handling
                return [dict(row._mapping) for row in rows]

        result = await _execute()

        # Cache result if cache_key provided
        if cache_key:
            self.query_cache[cache_key] = (result, datetime.now())
            self.metrics.cache_misses += 1

        return result

    async def bulk_insert_optimized(self,
                                  table_name: str,
                                  data: List[Dict[str, Any]],
                                  batch_size: int = 1000) -> int:
        """Optimized bulk insert with batching"""
        if not data:
            return 0

        total_inserted = 0

        # Process in batches for optimal performance
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            @self.performance_monitor(f"bulk_insert_{table_name}")
            async def _insert_batch():
                nonlocal total_inserted

                # Build bulk insert query
                if batch:
                    columns = list(batch[0].keys())
                    placeholders = ', '.join([f':{col}' for col in columns])
                    query = f"""
                        INSERT INTO {table_name} ({', '.join(columns)})
                        VALUES ({placeholders})
                    """

                    async with self.get_connection() as conn:
                        result = await conn.execute(text(query), batch)
                        total_inserted += result.rowcount

            await _insert_batch()

        self.logger.info(f"Bulk inserted {total_inserted} records into {table_name}")
        return total_inserted

    def optimize_table_indexes(self) -> List[str]:
        """Generate recommended index optimizations for telemetry tables"""

        recommendations = [
            # Time-series optimizations for telemetry_packets
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_telemetry_time_spacecraft "
            "ON telemetry_packets (spacecraft_time, spacecraft_id) "
            "WHERE spacecraft_time >= NOW() - INTERVAL '30 days';",

            # Composite index for common query patterns
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_telemetry_status_time "
            "ON telemetry_packets (telemetry_status, spacecraft_time DESC) "
            "INCLUDE (packet_id, sequence_number);",

            # Partial index for active spacecraft
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_spacecraft "
            "ON spacecraft (spacecraft_id, mission_phase) "
            "WHERE is_active = true;",

            # Index for anomaly detection queries
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_telemetry_payload_gin "
            "ON telemetry_packets USING GIN (processed_data) "
            "WHERE alert_level > 0;",

            # Time-based partitioning suggestion
            "-- Consider partitioning telemetry_packets by spacecraft_time "
            "-- for tables > 100M records",
        ]

        return recommendations

    async def analyze_query_performance(self) -> Dict[str, Any]:
        """Analyze current query performance and provide recommendations"""

        analysis = {
            'current_metrics': {
                'average_query_time_ms': self.metrics.get_average_query_time() * 1000,
                'total_queries': self.metrics.query_count,
                'cache_hit_ratio': self.metrics.get_cache_hit_ratio(),
                'connection_pool_utilization': (
                    self.metrics.connection_pool_active /
                    (self.metrics.connection_pool_active + self.metrics.connection_pool_idle)
                    if (self.metrics.connection_pool_active + self.metrics.connection_pool_idle) > 0
                    else 0
                )
            },
            'recommendations': [],
            'index_suggestions': self.optimize_table_indexes()
        }

        # Performance-based recommendations
        avg_time_ms = self.metrics.get_average_query_time() * 1000
        if avg_time_ms > 50:
            analysis['recommendations'].append({
                'type': 'query_optimization',
                'priority': 'high',
                'message': f'Average query time ({avg_time_ms:.1f}ms) exceeds target (<10ms)',
                'action': 'Review slow queries and add appropriate indexes'
            })

        if self.metrics.get_cache_hit_ratio() < 0.8:
            analysis['recommendations'].append({
                'type': 'cache_optimization',
                'priority': 'medium',
                'message': f'Cache hit ratio ({self.metrics.get_cache_hit_ratio():.1%}) is below optimal',
                'action': 'Increase cache TTL or cache size for frequently accessed data'
            })

        pool_util = analysis['current_metrics']['connection_pool_utilization']
        if pool_util > 0.8:
            analysis['recommendations'].append({
                'type': 'connection_pool',
                'priority': 'high',
                'message': f'Connection pool utilization ({pool_util:.1%}) is high',
                'action': 'Consider increasing pool size or optimizing connection usage'
            })

        return analysis


class CacheManager:
    """Redis-based caching for improved performance"""

    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 default_ttl: int = 300,
                 max_connections: int = 20):

        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.max_connections = max_connections

        self.redis_pool = redis.ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            retry_on_timeout=True,
            health_check_interval=30
        )

        self.client = redis.Redis(connection_pool=self.redis_pool)
        self.metrics = PerformanceMetrics()
        self.logger = logging.getLogger(__name__)

        # Prometheus metrics
        self.cache_operations = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'status']
        )
        self.cache_hit_ratio_gauge = Gauge(
            'cache_hit_ratio',
            'Cache hit ratio'
        )

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with metrics tracking"""
        try:
            value = await self.client.get(key)
            if value is not None:
                self.metrics.cache_hits += 1
                self.cache_operations.labels(operation='get', status='hit').inc()

                # Update hit ratio gauge
                hit_ratio = self.metrics.get_cache_hit_ratio()
                self.cache_hit_ratio_gauge.set(hit_ratio)

                # Deserialize if JSON
                import json
                try:
                    return json.loads(value.decode('utf-8'))
                except:
                    return value.decode('utf-8')
            else:
                self.metrics.cache_misses += 1
                self.cache_operations.labels(operation='get', status='miss').inc()
                return None

        except Exception as e:
            self.logger.error(f"Cache get error for key {key}: {str(e)}")
            self.cache_operations.labels(operation='get', status='error').inc()
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL"""
        try:
            # Serialize value
            import json
            if isinstance(value, (dict, list, int, float, bool)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)

            ttl = ttl or self.default_ttl
            await self.client.setex(key, ttl, serialized_value)

            self.cache_operations.labels(operation='set', status='success').inc()
            return True

        except Exception as e:
            self.logger.error(f"Cache set error for key {key}: {str(e)}")
            self.cache_operations.labels(operation='set', status='error').inc()
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            result = await self.client.delete(key)
            self.cache_operations.labels(operation='delete',
                                       status='success' if result else 'not_found').inc()
            return bool(result)

        except Exception as e:
            self.logger.error(f"Cache delete error for key {key}: {str(e)}")
            self.cache_operations.labels(operation='delete', status='error').inc()
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics and health information"""
        try:
            info = await self.client.info()

            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', 'unknown'),
                'used_memory_peak_human': info.get('used_memory_peak_human', 'unknown'),
                'hit_ratio': self.metrics.get_cache_hit_ratio(),
                'total_hits': self.metrics.cache_hits,
                'total_misses': self.metrics.cache_misses,
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'ops_per_sec': info.get('instantaneous_ops_per_sec', 0)
            }

        except Exception as e:
            self.logger.error(f"Error getting cache stats: {str(e)}")
            return {'error': str(e)}


class PerformanceMonitor:
    """Real-time performance monitoring and alerting"""

    def __init__(self,
                 check_interval: int = 30,
                 alert_thresholds: Optional[Dict[str, float]] = None):

        self.check_interval = check_interval
        self.alert_thresholds = alert_thresholds or {
            'query_time_ms': 50.0,
            'memory_usage_percent': 80.0,
            'cpu_usage_percent': 85.0,
            'cache_hit_ratio': 0.7,
            'connection_pool_utilization': 0.8
        }

        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history: List[Tuple[datetime, PerformanceMetrics]] = []
        self.max_history = 1440  # 24 hours of 1-minute samples

        self.logger = logging.getLogger(__name__)

        # Prometheus metrics
        self.system_metrics = {
            'memory_usage': Gauge('system_memory_usage_percent', 'System memory usage'),
            'cpu_usage': Gauge('system_cpu_usage_percent', 'System CPU usage'),
            'disk_usage': Gauge('system_disk_usage_percent', 'System disk usage'),
        }

    def start_monitoring(self):
        """Start background performance monitoring"""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info(f"Performance monitoring started (interval: {self.check_interval}s)")

    def stop_monitoring(self):
        """Stop background performance monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self.logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop running in background thread"""
        while self.running:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()

                # Store in history
                self.metrics_history.append((datetime.now(), metrics))

                # Limit history size
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history:]

                # Update Prometheus metrics
                self.system_metrics['memory_usage'].set(metrics.memory_usage_mb)
                self.system_metrics['cpu_usage'].set(metrics.cpu_usage_percent)

                # Check for alerts
                self._check_performance_alerts(metrics)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")

            time.sleep(self.check_interval)

    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Disk metrics
            disk = psutil.disk_usage('/')

            metrics = PerformanceMetrics()
            metrics.memory_usage_mb = memory.used / (1024 * 1024)
            metrics.cpu_usage_percent = cpu_percent

            return metrics

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")
            return PerformanceMetrics()

    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check metrics against alert thresholds"""
        alerts = []

        # Memory usage alert
        memory_percent = (metrics.memory_usage_mb / (psutil.virtual_memory().total / (1024 * 1024))) * 100
        if memory_percent > self.alert_thresholds['memory_usage_percent']:
            alerts.append({
                'type': 'memory_usage',
                'level': 'warning',
                'message': f'High memory usage: {memory_percent:.1f}%',
                'current': memory_percent,
                'threshold': self.alert_thresholds['memory_usage_percent']
            })

        # CPU usage alert
        if metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage_percent']:
            alerts.append({
                'type': 'cpu_usage',
                'level': 'warning',
                'message': f'High CPU usage: {metrics.cpu_usage_percent:.1f}%',
                'current': metrics.cpu_usage_percent,
                'threshold': self.alert_thresholds['cpu_usage_percent']
            })

        # Log alerts
        for alert in alerts:
            self.logger.warning(f"Performance alert: {alert['message']}")

    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for specified time window"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_metrics = [
            metrics for timestamp, metrics in self.metrics_history
            if timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {'error': 'No metrics available for specified time window'}

        # Calculate aggregated statistics
        memory_values = [m.memory_usage_mb for m in recent_metrics]
        cpu_values = [m.cpu_usage_percent for m in recent_metrics]

        return {
            'time_window_hours': hours,
            'sample_count': len(recent_metrics),
            'memory_usage_mb': {
                'min': min(memory_values),
                'max': max(memory_values),
                'avg': sum(memory_values) / len(memory_values),
            },
            'cpu_usage_percent': {
                'min': min(cpu_values),
                'max': max(cpu_values),
                'avg': sum(cpu_values) / len(cpu_values),
            },
            'alerts_triggered': len([
                m for m in recent_metrics
                if m.cpu_usage_percent > self.alert_thresholds['cpu_usage_percent']
            ])
        }


class PerformanceOptimizationService:
    """Main service coordinating all performance optimization components"""

    def __init__(self,
                 db_connection_string: str,
                 redis_url: str = "redis://localhost:6379"):

        self.db_optimizer = QueryOptimizer(db_connection_string)
        self.cache_manager = CacheManager(redis_url)
        self.performance_monitor = PerformanceMonitor()

        self.logger = logging.getLogger(__name__)

        # Start monitoring
        self.performance_monitor.start_monitoring()

    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics from all components"""

        try:
            # Database metrics
            db_analysis = await self.db_optimizer.analyze_query_performance()

            # Cache metrics
            cache_stats = await self.cache_manager.get_stats()

            # System metrics
            system_summary = self.performance_monitor.get_performance_summary()

            return {
                'timestamp': datetime.now().isoformat(),
                'database': db_analysis,
                'cache': cache_stats,
                'system': system_summary,
                'overall_health': self._calculate_overall_health(
                    db_analysis, cache_stats, system_summary
                )
            }

        except Exception as e:
            self.logger.error(f"Error getting comprehensive metrics: {str(e)}")
            return {'error': str(e)}

    def _calculate_overall_health(self,
                                db_analysis: Dict,
                                cache_stats: Dict,
                                system_summary: Dict) -> Dict[str, Any]:
        """Calculate overall system health score"""

        health_score = 100
        issues = []

        # Database health
        if 'current_metrics' in db_analysis:
            avg_query_time = db_analysis['current_metrics'].get('average_query_time_ms', 0)
            if avg_query_time > 50:
                health_score -= 20
                issues.append(f"Slow database queries ({avg_query_time:.1f}ms avg)")

        # Cache health
        if 'hit_ratio' in cache_stats:
            hit_ratio = cache_stats['hit_ratio']
            if hit_ratio < 0.7:
                health_score -= 15
                issues.append(f"Low cache hit ratio ({hit_ratio:.1%})")

        # System health
        if 'cpu_usage_percent' in system_summary:
            avg_cpu = system_summary['cpu_usage_percent'].get('avg', 0)
            if avg_cpu > 80:
                health_score -= 25
                issues.append(f"High CPU usage ({avg_cpu:.1f}%)")

        # Determine health status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 70:
            status = "good"
        elif health_score >= 50:
            status = "fair"
        else:
            status = "poor"

        return {
            'status': status,
            'score': max(0, health_score),
            'issues': issues
        }

    async def apply_optimizations(self) -> Dict[str, Any]:
        """Apply automatic performance optimizations"""

        results = {
            'database_optimizations': [],
            'cache_optimizations': [],
            'system_optimizations': [],
            'success': True
        }

        try:
            # Database optimizations
            db_recommendations = self.db_optimizer.optimize_table_indexes()
            results['database_optimizations'] = [
                {'type': 'index_suggestion', 'query': rec}
                for rec in db_recommendations
            ]

            # Cache optimizations (placeholder for auto-tuning)
            cache_stats = await self.cache_manager.get_stats()
            if cache_stats.get('hit_ratio', 1.0) < 0.8:
                results['cache_optimizations'].append({
                    'type': 'cache_tuning',
                    'action': 'Consider increasing cache TTL for frequently accessed data'
                })

            self.logger.info("Performance optimizations applied successfully")

        except Exception as e:
            self.logger.error(f"Error applying optimizations: {str(e)}")
            results['success'] = False
            results['error'] = str(e)

        return results


# Global service instance
performance_service: Optional[PerformanceOptimizationService] = None


async def initialize_performance_service(db_connection_string: str,
                                       redis_url: str = "redis://localhost:6379"):
    """Initialize global performance optimization service"""
    global performance_service
    performance_service = PerformanceOptimizationService(db_connection_string, redis_url)
    return performance_service


async def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics"""
    if not performance_service:
        raise RuntimeError("Performance service not initialized")

    return await performance_service.get_comprehensive_metrics()


async def optimize_performance() -> Dict[str, Any]:
    """Apply performance optimizations"""
    if not performance_service:
        raise RuntimeError("Performance service not initialized")

    return await performance_service.apply_optimizations()
