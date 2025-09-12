"""
Analytics Path Tests - Vector DB and ML Pipeline

Tests for predictive analytics, anomaly detection, and machine learning
capabilities through Vector database and ML pipeline integration.
Validates ML model performance, vector similarity search, and predictive analytics.
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
import hashlib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import requests
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp

# Test configuration
ANALYTICS_CONFIG = {
    "vector_db_url": "http://localhost:19530",  # Milvus default port
    "ml_pipeline_url": "http://localhost:8080",
    "prediction_accuracy_threshold": 0.85,
    "anomaly_detection_threshold": 0.05,  # 5% false positive rate
    "vector_similarity_threshold": 0.7,
    "model_training_timeout_s": 300,  # 5 minutes
    "prediction_latency_ms": 100,  # <100ms prediction time
    "batch_size": 1000,
    "vector_dimension": 128,
    "collection_name": "telemetry_embeddings_test"
}

class AnalyticsPathTester:
    """Analytics Path testing framework for ML and Vector DB"""
    
    def __init__(self):
        self.vector_db_client = None
        self.ml_models = {}
        self.test_collections = []
        self.performance_metrics = {}
        self.scaler = StandardScaler()
    
    async def setup(self):
        """Initialize Vector DB and ML pipeline"""
        await self.setup_vector_db()
        await self.setup_ml_pipeline()
        await self.clean_test_collections()
    
    async def teardown(self):
        """Cleanup test environment"""
        await self.clean_test_collections()
    
    async def setup_vector_db(self):
        """Setup Vector DB connection (simulated for testing)"""
        # In a real implementation, this would connect to Milvus/Weaviate/Pinecone
        # For testing, we'll simulate vector operations
        self.vector_db_client = MockVectorDB()
        await self.vector_db_client.connect()
    
    async def setup_ml_pipeline(self):
        """Initialize ML models for testing"""
        # Initialize anomaly detection model
        self.ml_models['anomaly_detector'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Initialize trend prediction model (simplified)
        from sklearn.linear_model import LinearRegression
        self.ml_models['trend_predictor'] = LinearRegression()
        
        # Initialize classification model for health status
        from sklearn.ensemble import RandomForestClassifier
        self.ml_models['health_classifier'] = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    
    async def clean_test_collections(self):
        """Remove test collections"""
        for collection_name in self.test_collections:
            if hasattr(self.vector_db_client, 'drop_collection'):
                await self.vector_db_client.drop_collection(collection_name)
        self.test_collections.clear()
    
    def generate_telemetry_features(self, num_samples: int = 1000, include_anomalies: bool = True) -> pd.DataFrame:
        """Generate realistic telemetry feature data"""
        np.random.seed(42)
        
        # Base satellite telemetry features
        data = []
        
        for i in range(num_samples):
            timestamp = datetime.utcnow() - timedelta(minutes=i)
            
            # Normal operational parameters
            temperature = np.random.normal(25, 15)  # °C
            voltage = np.random.normal(12.5, 0.5)   # V
            current = np.random.normal(2.0, 0.3)    # A
            pressure = np.random.normal(101.3, 5.0) # kPa
            altitude = np.random.normal(425000, 5000) # meters (ISS altitude)
            velocity = np.random.normal(27600, 100)   # m/s (orbital velocity)
            
            # Derived features
            power = voltage * current
            battery_level = max(0, min(100, 
                90 + np.random.normal(0, 10) + np.sin(i * 0.1) * 20))  # Orbital charging cycle
            
            # System health indicators
            cpu_usage = max(0, min(100, np.random.normal(45, 15)))
            memory_usage = max(0, min(100, np.random.normal(60, 20)))
            disk_usage = max(0, min(100, np.random.normal(30, 10)))
            
            # Communication metrics
            signal_strength = np.random.normal(-85, 10)  # dBm
            data_rate = np.random.choice([1024, 2048, 4096, 8192])  # bps
            
            # Attitude control
            gyro_x = np.random.normal(0, 0.1)  # rad/s
            gyro_y = np.random.normal(0, 0.1)
            gyro_z = np.random.normal(0, 0.1)
            
            # Introduce anomalies for 5% of samples
            is_anomaly = False
            if include_anomalies and np.random.random() < 0.05:
                is_anomaly = True
                # Introduce various types of anomalies
                anomaly_type = np.random.choice(['temperature', 'voltage', 'battery', 'system'])
                
                if anomaly_type == 'temperature':
                    temperature += np.random.choice([-30, 40])  # Extreme temperature
                elif anomaly_type == 'voltage':
                    voltage *= np.random.uniform(0.5, 0.7)  # Voltage drop
                elif anomaly_type == 'battery':
                    battery_level *= 0.3  # Battery drain
                elif anomaly_type == 'system':
                    cpu_usage = 95 + np.random.uniform(0, 5)  # High CPU usage
                    memory_usage = 90 + np.random.uniform(0, 10)
            
            # Health status (based on multiple factors)
            health_factors = [
                1 if 10 <= temperature <= 40 else 0,
                1 if 11.0 <= voltage <= 14.0 else 0,
                1 if battery_level > 20 else 0,
                1 if cpu_usage < 80 else 0,
                1 if memory_usage < 85 else 0
            ]
            
            health_score = sum(health_factors) / len(health_factors)
            
            if health_score >= 0.8:
                health_status = 'healthy'
            elif health_score >= 0.6:
                health_status = 'degraded'
            else:
                health_status = 'critical'
            
            record = {
                'timestamp': timestamp,
                'temperature': temperature,
                'voltage': voltage,
                'current': current,
                'power': power,
                'pressure': pressure,
                'altitude': altitude,
                'velocity': velocity,
                'battery_level': battery_level,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'signal_strength': signal_strength,
                'data_rate': data_rate,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z,
                'health_score': health_score,
                'health_status': health_status,
                'is_anomaly': is_anomaly,
                'satellite_id': f"SAT-{np.random.randint(1, 6):03d}",
                'mission_phase': np.random.choice(['launch', 'orbit_insertion', 'operational', 'maintenance'])
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def create_feature_vectors(self, df: pd.DataFrame) -> np.ndarray:
        """Create feature vectors for ML models"""
        # Select numeric features for vector creation
        feature_columns = [
            'temperature', 'voltage', 'current', 'power', 'pressure',
            'altitude', 'velocity', 'battery_level', 'cpu_usage', 
            'memory_usage', 'disk_usage', 'signal_strength', 'gyro_x', 'gyro_y', 'gyro_z'
        ]
        
        feature_matrix = df[feature_columns].values
        
        # Normalize features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        return feature_matrix_scaled
    
    def create_embeddings(self, feature_vectors: np.ndarray, target_dim: int = 128) -> np.ndarray:
        """Create embeddings for vector similarity search"""
        # Simulate dimensionality reduction/embedding creation
        # In production, this would use actual embedding models
        
        from sklearn.decomposition import PCA
        
        if feature_vectors.shape[1] > target_dim:
            pca = PCA(n_components=target_dim)
            embeddings = pca.fit_transform(feature_vectors)
        else:
            # Pad or repeat features to reach target dimension
            repeat_factor = target_dim // feature_vectors.shape[1] + 1
            embeddings = np.tile(feature_vectors, (1, repeat_factor))[:, :target_dim]
        
        # Normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        return embeddings

class MockVectorDB:
    """Mock Vector Database for testing"""
    
    def __init__(self):
        self.collections = {}
        self.connected = False
    
    async def connect(self):
        """Simulate connection to vector database"""
        self.connected = True
    
    async def create_collection(self, name: str, dimension: int) -> bool:
        """Create a collection for vectors"""
        self.collections[name] = {
            'dimension': dimension,
            'vectors': [],
            'metadata': [],
            'ids': []
        }
        return True
    
    async def insert_vectors(self, collection_name: str, vectors: np.ndarray, 
                           metadata: List[Dict], ids: List[str]) -> bool:
        """Insert vectors with metadata"""
        if collection_name not in self.collections:
            return False
        
        collection = self.collections[collection_name]
        collection['vectors'].extend(vectors.tolist())
        collection['metadata'].extend(metadata)
        collection['ids'].extend(ids)
        return True
    
    async def search(self, collection_name: str, query_vector: np.ndarray, 
                    top_k: int = 10) -> List[Dict]:
        """Search for similar vectors"""
        if collection_name not in self.collections:
            return []
        
        collection = self.collections[collection_name]
        
        if not collection['vectors']:
            return []
        
        # Calculate cosine similarity
        vectors = np.array(collection['vectors'])
        query_norm = np.linalg.norm(query_vector)
        vector_norms = np.linalg.norm(vectors, axis=1)
        
        # Avoid division by zero
        similarities = np.dot(vectors, query_vector) / (vector_norms * query_norm + 1e-8)
        
        # Get top_k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'id': collection['ids'][idx],
                'score': float(similarities[idx]),
                'metadata': collection['metadata'][idx]
            })
        
        return results
    
    async def drop_collection(self, name: str) -> bool:
        """Drop a collection"""
        if name in self.collections:
            del self.collections[name]
            return True
        return False

@pytest.fixture
async def analytics_tester():
    """Pytest fixture for analytics testing"""
    tester = AnalyticsPathTester()
    await tester.setup()
    yield tester
    await tester.teardown()

class TestMLPipelineTraining:
    """Test ML model training and validation"""
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_model_training(self, analytics_tester):
        """Test anomaly detection model training and performance"""
        # Generate training data with known anomalies
        df = analytics_tester.generate_telemetry_features(2000, include_anomalies=True)
        
        # Create feature vectors
        feature_vectors = analytics_tester.create_feature_vectors(df)
        
        # Prepare training data (normal samples only for unsupervised learning)
        normal_samples = feature_vectors[~df['is_anomaly'].values]
        
        start_time = time.perf_counter()
        
        # Train anomaly detection model
        analytics_tester.ml_models['anomaly_detector'].fit(normal_samples)
        
        training_time = time.perf_counter() - start_time
        
        # Test on full dataset
        predictions = analytics_tester.ml_models['anomaly_detector'].predict(feature_vectors)
        anomaly_scores = analytics_tester.ml_models['anomaly_detector'].decision_function(feature_vectors)
        
        # Convert predictions (-1 for anomaly, 1 for normal) to boolean
        predicted_anomalies = predictions == -1
        actual_anomalies = df['is_anomaly'].values
        
        # Calculate performance metrics
        true_positives = np.sum(predicted_anomalies & actual_anomalies)
        false_positives = np.sum(predicted_anomalies & ~actual_anomalies)
        false_negatives = np.sum(~predicted_anomalies & actual_anomalies)
        true_negatives = np.sum(~predicted_anomalies & ~actual_anomalies)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        false_positive_rate = false_positives / (false_positives + true_negatives)
        
        print(f"Anomaly Detection Results:")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  False Positive Rate: {false_positive_rate:.3f}")
        print(f"  True Anomalies: {np.sum(actual_anomalies)}")
        print(f"  Detected Anomalies: {np.sum(predicted_anomalies)}")
        
        # Performance assertions
        assert training_time < ANALYTICS_CONFIG["model_training_timeout_s"], \
            f"Training took {training_time:.2f}s, exceeds limit"
        assert false_positive_rate <= ANALYTICS_CONFIG["anomaly_detection_threshold"], \
            f"False positive rate {false_positive_rate:.3f} exceeds threshold"
        assert f1 >= 0.5, f"F1 score {f1:.3f} too low"  # Reasonable threshold for anomaly detection
    
    @pytest.mark.asyncio
    async def test_health_status_classification(self, analytics_tester):
        """Test satellite health status classification"""
        # Generate training data
        df = analytics_tester.generate_telemetry_features(3000, include_anomalies=True)
        
        # Create features and labels
        feature_vectors = analytics_tester.create_feature_vectors(df)
        health_labels = df['health_status'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            feature_vectors, health_labels, 
            test_size=0.2, random_state=42, stratify=health_labels
        )
        
        start_time = time.perf_counter()
        
        # Train classification model
        analytics_tester.ml_models['health_classifier'].fit(X_train, y_train)
        
        training_time = time.perf_counter() - start_time
        
        # Make predictions
        start_pred_time = time.perf_counter()
        y_pred = analytics_tester.ml_models['health_classifier'].predict(X_test)
        prediction_time = time.perf_counter() - start_pred_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Feature importance analysis
        feature_names = [
            'temperature', 'voltage', 'current', 'power', 'pressure',
            'altitude', 'velocity', 'battery_level', 'cpu_usage', 
            'memory_usage', 'disk_usage', 'signal_strength', 'gyro_x', 'gyro_y', 'gyro_z'
        ]
        
        importances = analytics_tester.ml_models['health_classifier'].feature_importances_
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"Health Classification Results:")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Prediction time: {prediction_time * 1000:.2f}ms")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  Test samples: {len(y_test)}")
        print(f"  Top 5 Features:")
        for feature, importance in top_features:
            print(f"    {feature}: {importance:.3f}")
        
        # Performance assertions
        assert training_time < ANALYTICS_CONFIG["model_training_timeout_s"], \
            f"Training took {training_time:.2f}s, exceeds limit"
        assert prediction_time * 1000 < ANALYTICS_CONFIG["prediction_latency_ms"], \
            f"Prediction took {prediction_time * 1000:.2f}ms, exceeds limit"
        assert accuracy >= ANALYTICS_CONFIG["prediction_accuracy_threshold"], \
            f"Accuracy {accuracy:.3f} below threshold"
    
    @pytest.mark.asyncio
    async def test_trend_prediction_model(self, analytics_tester):
        """Test time series trend prediction"""
        # Generate time series data
        df = analytics_tester.generate_telemetry_features(1000, include_anomalies=False)
        df = df.sort_values('timestamp')
        
        # Focus on battery level prediction as an example
        battery_data = df['battery_level'].values
        
        # Create sliding window features (last 10 values predict next value)
        window_size = 10
        X, y = [], []
        
        for i in range(window_size, len(battery_data)):
            X.append(battery_data[i-window_size:i])
            y.append(battery_data[i])
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        start_time = time.perf_counter()
        
        # Train trend prediction model
        analytics_tester.ml_models['trend_predictor'].fit(X_train, y_train)
        
        training_time = time.perf_counter() - start_time
        
        # Make predictions
        start_pred_time = time.perf_counter()
        y_pred = analytics_tester.ml_models['trend_predictor'].predict(X_test)
        prediction_time = time.perf_counter() - start_pred_time
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Calculate percentage accuracy
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        print(f"Trend Prediction Results:")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Prediction time: {prediction_time * 1000:.2f}ms")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Test samples: {len(y_test)}")
        
        # Performance assertions
        assert training_time < ANALYTICS_CONFIG["model_training_timeout_s"], \
            f"Training took {training_time:.2f}s, exceeds limit"
        assert prediction_time * 1000 < ANALYTICS_CONFIG["prediction_latency_ms"], \
            f"Prediction took {prediction_time * 1000:.2f}ms, exceeds limit"
        assert mape < 20, f"MAPE {mape:.2f}% too high for trend prediction"

class TestVectorSimilaritySearch:
    """Test vector database and similarity search"""
    
    @pytest.mark.asyncio
    async def test_vector_collection_creation(self, analytics_tester):
        """Test vector collection creation and management"""
        collection_name = f"test_collection_{int(time.time())}"
        dimension = ANALYTICS_CONFIG["vector_dimension"]
        
        # Create collection
        success = await analytics_tester.vector_db_client.create_collection(
            collection_name, dimension
        )
        
        assert success, "Failed to create vector collection"
        assert collection_name in analytics_tester.vector_db_client.collections
        
        collection = analytics_tester.vector_db_client.collections[collection_name]
        assert collection['dimension'] == dimension
        
        analytics_tester.test_collections.append(collection_name)
        
        print(f"Vector collection created: {collection_name} (dim={dimension})")
    
    @pytest.mark.asyncio
    async def test_bulk_vector_insertion(self, analytics_tester):
        """Test bulk vector insertion performance"""
        # Generate test data
        df = analytics_tester.generate_telemetry_features(1000, include_anomalies=True)
        feature_vectors = analytics_tester.create_feature_vectors(df)
        embeddings = analytics_tester.create_embeddings(
            feature_vectors, ANALYTICS_CONFIG["vector_dimension"]
        )
        
        # Create collection
        collection_name = f"bulk_test_{int(time.time())}"
        await analytics_tester.vector_db_client.create_collection(
            collection_name, ANALYTICS_CONFIG["vector_dimension"]
        )
        
        # Prepare metadata and IDs
        metadata = []
        ids = []
        
        for i, row in df.iterrows():
            metadata.append({
                'timestamp': row['timestamp'].isoformat(),
                'satellite_id': row['satellite_id'],
                'mission_phase': row['mission_phase'],
                'health_status': row['health_status'],
                'is_anomaly': row['is_anomaly'],
                'battery_level': row['battery_level'],
                'temperature': row['temperature']
            })
            ids.append(f"vec_{i:06d}")
        
        # Bulk insert
        start_time = time.perf_counter()
        
        success = await analytics_tester.vector_db_client.insert_vectors(
            collection_name, embeddings, metadata, ids
        )
        
        insertion_time = time.perf_counter() - start_time
        
        assert success, "Failed to insert vectors"
        
        # Verify insertion
        collection = analytics_tester.vector_db_client.collections[collection_name]
        assert len(collection['vectors']) == len(embeddings)
        assert len(collection['metadata']) == len(metadata)
        assert len(collection['ids']) == len(ids)
        
        vectors_per_second = len(embeddings) / insertion_time
        
        print(f"Bulk vector insertion:")
        print(f"  Vectors inserted: {len(embeddings)}")
        print(f"  Insertion time: {insertion_time:.2f}s")
        print(f"  Vectors per second: {vectors_per_second:.0f}")
        print(f"  Vector dimension: {embeddings.shape[1]}")
        
        # Performance assertion
        assert vectors_per_second > 100, f"Insertion rate {vectors_per_second:.0f} too low"
        
        analytics_tester.test_collections.append(collection_name)
    
    @pytest.mark.asyncio
    async def test_similarity_search_performance(self, analytics_tester):
        """Test vector similarity search performance"""
        # Setup test collection with data
        df = analytics_tester.generate_telemetry_features(500, include_anomalies=True)
        feature_vectors = analytics_tester.create_feature_vectors(df)
        embeddings = analytics_tester.create_embeddings(
            feature_vectors, ANALYTICS_CONFIG["vector_dimension"]
        )
        
        collection_name = f"search_test_{int(time.time())}"
        await analytics_tester.vector_db_client.create_collection(
            collection_name, ANALYTICS_CONFIG["vector_dimension"]
        )
        
        # Insert test vectors
        metadata = [{'index': i, 'type': 'test'} for i in range(len(embeddings))]
        ids = [f"test_vec_{i:04d}" for i in range(len(embeddings))]
        
        await analytics_tester.vector_db_client.insert_vectors(
            collection_name, embeddings, metadata, ids
        )
        
        # Perform similarity searches
        search_times = []
        similarity_scores = []
        
        num_searches = 10
        for i in range(num_searches):
            # Use random vector from the collection as query
            query_idx = np.random.randint(0, len(embeddings))
            query_vector = embeddings[query_idx]
            
            start_time = time.perf_counter()
            
            results = await analytics_tester.vector_db_client.search(
                collection_name, query_vector, top_k=10
            )
            
            search_time = time.perf_counter() - start_time
            search_times.append(search_time)
            
            # Verify results
            assert len(results) <= 10, "Too many results returned"
            assert len(results) > 0, "No results returned"
            
            # Check if query vector itself is in top results (should have highest similarity)
            top_result = results[0]
            assert top_result['score'] >= ANALYTICS_CONFIG["vector_similarity_threshold"], \
                f"Top similarity score {top_result['score']:.3f} below threshold"
            
            similarity_scores.extend([r['score'] for r in results])
        
        # Analyze search performance
        avg_search_time = np.mean(search_times)
        max_search_time = np.max(search_times)
        avg_similarity = np.mean(similarity_scores)
        
        print(f"Similarity Search Results:")
        print(f"  Searches performed: {num_searches}")
        print(f"  Average search time: {avg_search_time * 1000:.2f}ms")
        print(f"  Max search time: {max_search_time * 1000:.2f}ms")
        print(f"  Average similarity score: {avg_similarity:.3f}")
        print(f"  Collection size: {len(embeddings)} vectors")
        
        # Performance assertions
        assert avg_search_time * 1000 < ANALYTICS_CONFIG["prediction_latency_ms"], \
            f"Average search time {avg_search_time * 1000:.2f}ms exceeds limit"
        assert avg_similarity >= ANALYTICS_CONFIG["vector_similarity_threshold"], \
            f"Average similarity {avg_similarity:.3f} below threshold"
        
        analytics_tester.test_collections.append(collection_name)
    
    @pytest.mark.asyncio
    async def test_anomaly_similarity_clustering(self, analytics_tester):
        """Test similarity search for anomaly clustering"""
        # Generate data with specific anomaly patterns
        df = analytics_tester.generate_telemetry_features(1000, include_anomalies=True)
        feature_vectors = analytics_tester.create_feature_vectors(df)
        embeddings = analytics_tester.create_embeddings(
            feature_vectors, ANALYTICS_CONFIG["vector_dimension"]
        )
        
        collection_name = f"anomaly_cluster_test_{int(time.time())}"
        await analytics_tester.vector_db_client.create_collection(
            collection_name, ANALYTICS_CONFIG["vector_dimension"]
        )
        
        # Insert vectors with anomaly metadata
        metadata = []
        ids = []
        
        for i, row in df.iterrows():
            metadata.append({
                'is_anomaly': row['is_anomaly'],
                'health_status': row['health_status'],
                'satellite_id': row['satellite_id'],
                'temperature': row['temperature'],
                'voltage': row['voltage'],
                'battery_level': row['battery_level']
            })
            ids.append(f"anomaly_vec_{i:04d}")
        
        await analytics_tester.vector_db_client.insert_vectors(
            collection_name, embeddings, metadata, ids
        )
        
        # Find anomaly vectors and test clustering
        anomaly_indices = df[df['is_anomaly']].index.tolist()
        
        if len(anomaly_indices) > 0:
            # Test similarity between anomalous samples
            anomaly_similarities = []
            
            for i, anomaly_idx in enumerate(anomaly_indices[:5]):  # Test first 5 anomalies
                query_vector = embeddings[anomaly_idx]
                
                results = await analytics_tester.vector_db_client.search(
                    collection_name, query_vector, top_k=20
                )
                
                # Count how many of the top results are also anomalies
                similar_anomalies = 0
                for result in results[:10]:  # Check top 10
                    if result['metadata']['is_anomaly']:
                        similar_anomalies += 1
                
                anomaly_clustering_ratio = similar_anomalies / 10
                anomaly_similarities.append(anomaly_clustering_ratio)
            
            avg_anomaly_clustering = np.mean(anomaly_similarities)
            
            print(f"Anomaly Clustering Analysis:")
            print(f"  Total anomalies in dataset: {len(anomaly_indices)}")
            print(f"  Anomalies tested for clustering: {len(anomaly_similarities)}")
            print(f"  Average anomaly clustering ratio: {avg_anomaly_clustering:.3f}")
            print(f"  Expected clustering (anomalies should cluster together): >0.3")
            
            # Anomalies should cluster together more than random chance
            # With 5% anomaly rate, random clustering would be ~0.05
            # We expect at least 0.2 (20%) for meaningful clustering
            assert avg_anomaly_clustering > 0.2, \
                f"Anomaly clustering ratio {avg_anomaly_clustering:.3f} suggests poor anomaly similarity"
        
        analytics_tester.test_collections.append(collection_name)

class TestRealtimeAnalytics:
    """Test real-time analytics capabilities"""
    
    @pytest.mark.asyncio
    async def test_streaming_anomaly_detection(self, analytics_tester):
        """Test real-time streaming anomaly detection"""
        # Train model on historical data
        historical_df = analytics_tester.generate_telemetry_features(2000, include_anomalies=False)
        historical_features = analytics_tester.create_feature_vectors(historical_df)
        
        # Train on normal data only
        analytics_tester.ml_models['anomaly_detector'].fit(historical_features)
        
        # Simulate streaming data with some anomalies
        streaming_data = []
        detection_times = []
        
        num_streaming_samples = 100
        
        for i in range(num_streaming_samples):
            # Generate single sample (some anomalous)
            single_sample_df = analytics_tester.generate_telemetry_features(1, include_anomalies=(i % 20 == 0))
            single_features = analytics_tester.create_feature_vectors(single_sample_df)
            
            # Measure detection time
            start_time = time.perf_counter()
            
            prediction = analytics_tester.ml_models['anomaly_detector'].predict(single_features)[0]
            anomaly_score = analytics_tester.ml_models['anomaly_detector'].decision_function(single_features)[0]
            
            detection_time = time.perf_counter() - start_time
            detection_times.append(detection_time)
            
            streaming_data.append({
                'sample_index': i,
                'is_anomaly_actual': single_sample_df.iloc[0]['is_anomaly'],
                'is_anomaly_predicted': prediction == -1,
                'anomaly_score': anomaly_score,
                'detection_time_ms': detection_time * 1000
            })
        
        # Analyze streaming performance
        streaming_df = pd.DataFrame(streaming_data)
        
        avg_detection_time = np.mean(detection_times) * 1000  # ms
        max_detection_time = np.max(detection_times) * 1000   # ms
        
        # Calculate accuracy for streaming detection
        accuracy = accuracy_score(
            streaming_df['is_anomaly_actual'], 
            streaming_df['is_anomaly_predicted']
        )
        
        actual_anomalies = streaming_df['is_anomaly_actual'].sum()
        detected_anomalies = streaming_df['is_anomaly_predicted'].sum()
        
        print(f"Streaming Anomaly Detection:")
        print(f"  Samples processed: {num_streaming_samples}")
        print(f"  Average detection time: {avg_detection_time:.2f}ms")
        print(f"  Max detection time: {max_detection_time:.2f}ms")
        print(f"  Detection accuracy: {accuracy:.3f}")
        print(f"  Actual anomalies: {actual_anomalies}")
        print(f"  Detected anomalies: {detected_anomalies}")
        
        # Performance assertions
        assert avg_detection_time < ANALYTICS_CONFIG["prediction_latency_ms"], \
            f"Average detection time {avg_detection_time:.2f}ms exceeds limit"
        assert max_detection_time < ANALYTICS_CONFIG["prediction_latency_ms"] * 2, \
            f"Max detection time {max_detection_time:.2f}ms too high"
        assert accuracy >= 0.7, f"Streaming detection accuracy {accuracy:.3f} too low"
    
    @pytest.mark.asyncio
    async def test_predictive_maintenance_alerts(self, analytics_tester):
        """Test predictive maintenance alert system"""
        # Generate degradation patterns
        df = analytics_tester.generate_telemetry_features(500, include_anomalies=True)
        
        # Simulate component degradation over time
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add degradation trend to battery and temperature
        for i in range(len(df)):
            degradation_factor = i / len(df)  # Gradual degradation
            
            # Battery degrades over time
            if np.random.random() < 0.1:  # 10% chance of degradation event
                df.loc[i, 'battery_level'] *= (1 - degradation_factor * 0.3)
            
            # Temperature increases under stress
            if np.random.random() < 0.05:  # 5% chance of thermal event
                df.loc[i, 'temperature'] += degradation_factor * 20
        
        # Train models on feature data
        feature_vectors = analytics_tester.create_feature_vectors(df)
        
        # Create maintenance labels based on multiple factors
        maintenance_needed = (
            (df['battery_level'] < 30) |
            (df['temperature'] > 45) |
            (df['cpu_usage'] > 90) |
            (df['health_score'] < 0.5)
        )
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            feature_vectors, maintenance_needed.values, 
            test_size=0.3, random_state=42
        )
        
        # Train maintenance prediction model
        from sklearn.ensemble import GradientBoostingClassifier
        maintenance_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        
        start_time = time.perf_counter()
        maintenance_model.fit(X_train, y_train)
        training_time = time.perf_counter() - start_time
        
        # Test prediction performance
        start_pred_time = time.perf_counter()
        maintenance_predictions = maintenance_model.predict(X_test)
        maintenance_probabilities = maintenance_model.predict_proba(X_test)[:, 1]
        prediction_time = time.perf_counter() - start_pred_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, maintenance_predictions)
        f1 = f1_score(y_test, maintenance_predictions)
        
        # Analyze prediction confidence
        high_confidence_predictions = maintenance_probabilities > 0.8
        low_confidence_predictions = maintenance_probabilities < 0.2
        
        confidence_ratio = (np.sum(high_confidence_predictions) + np.sum(low_confidence_predictions)) / len(maintenance_probabilities)
        
        # Simulate alert generation
        alerts_generated = np.sum(maintenance_probabilities > 0.7)  # Alert threshold
        critical_alerts = np.sum(maintenance_probabilities > 0.9)   # Critical threshold
        
        print(f"Predictive Maintenance Results:")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Prediction time: {prediction_time * 1000:.2f}ms")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  Test samples: {len(y_test)}")
        print(f"  Alerts generated: {alerts_generated}")
        print(f"  Critical alerts: {critical_alerts}")
        print(f"  High/Low confidence ratio: {confidence_ratio:.3f}")
        
        # Performance assertions
        assert training_time < ANALYTICS_CONFIG["model_training_timeout_s"], \
            f"Training took {training_time:.2f}s, exceeds limit"
        assert prediction_time * 1000 < ANALYTICS_CONFIG["prediction_latency_ms"], \
            f"Prediction took {prediction_time * 1000:.2f}ms, exceeds limit"
        assert accuracy >= 0.75, f"Maintenance prediction accuracy {accuracy:.3f} too low"
        assert confidence_ratio >= 0.6, f"Model confidence ratio {confidence_ratio:.3f} too low"

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
