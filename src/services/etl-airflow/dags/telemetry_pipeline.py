"""
Space Telemetry ETL Pipeline
============================

Main ETL pipeline for processing space telemetry data through multiple temperature paths:
- Hot Path: Redis → Real-time processing
- Warm Path: PostgreSQL → Batch analytics
- Cold Path: MinIO → Long-term storage
- Analytics Path: Vector DB → ML/AI processing

This DAG orchestrates the complete data flow from ingestion through analytics.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.redis.sensors.redis_key import RedisKeySensor
from airflow.providers.http.operators.http import HttpOperator
from airflow.sensors.filesystem import FileSensor
import pandas as pd
import redis
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'space-ops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2)
}

# Create the DAG
dag = DAG(
    'space_telemetry_etl_pipeline',
    default_args=default_args,
    description='Complete ETL pipeline for space telemetry data processing',
    schedule_interval='@hourly',  # Run every hour
    catchup=False,
    max_active_runs=1,
    tags=['telemetry', 'etl', 'space', 'satellite']
)

def extract_hot_path_data(**context):
    """Extract data from Redis hot path"""
    try:
        redis_client = redis.Redis(
            host='redis',
            port=6379,
            decode_responses=True
        )

        # Get all telemetry keys from the last hour
        execution_date = context['execution_date']
        start_time = int(execution_date.timestamp()) * 1000
        end_time = int((execution_date + timedelta(hours=1)).timestamp()) * 1000

        telemetry_data = []
        keys = redis_client.keys(f"telemetry:*")

        for key in keys:
            data = redis_client.get(key)
            if data:
                telemetry = json.loads(data)
                # Filter by timestamp range
                if start_time <= telemetry.get('timestamp', 0) <= end_time:
                    telemetry_data.append(telemetry)

        logger.info(f"Extracted {len(telemetry_data)} records from hot path")

        # Store extracted data for next task
        context['task_instance'].xcom_push(
            key='hot_path_data',
            value=telemetry_data
        )

        return len(telemetry_data)

    except Exception as e:
        logger.error(f"Error extracting hot path data: {str(e)}")
        raise

def transform_telemetry_data(**context):
    """Transform and clean telemetry data"""
    try:
        # Get data from previous task
        hot_path_data = context['task_instance'].xcom_pull(
            key='hot_path_data',
            task_ids='extract_hot_path'
        )

        if not hot_path_data:
            logger.warning("No data to transform")
            return 0

        # Convert to DataFrame for processing
        df = pd.DataFrame(hot_path_data)

        # Data cleaning and transformation
        # 1. Remove duplicates
        df = df.drop_duplicates(subset=['timestamp', 'satelliteId'])

        # 2. Validate timestamp
        df = df[df['timestamp'].notna()]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 3. Add derived fields
        df['processing_time'] = datetime.utcnow()
        df['data_quality_score'] = df.apply(calculate_quality_score, axis=1)

        # 4. Categorize by telemetry type
        df['category'] = df['telemetryType'].map({
            'sensor': 'environmental',
            'status': 'operational',
            'command': 'control',
            'health': 'diagnostic'
        }).fillna('unknown')

        # 5. Extract nested data fields
        data_fields = df['data'].apply(pd.Series)
        df = pd.concat([df.drop(['data'], axis=1), data_fields], axis=1)

        # Convert back to records
        transformed_data = df.to_dict('records')

        logger.info(f"Transformed {len(transformed_data)} records")

        # Store transformed data
        context['task_instance'].xcom_push(
            key='transformed_data',
            value=transformed_data
        )

        return len(transformed_data)

    except Exception as e:
        logger.error(f"Error transforming data: {str(e)}")
        raise

def calculate_quality_score(row):
    """Calculate data quality score based on completeness and validity"""
    score = 100

    # Check required fields
    required_fields = ['timestamp', 'satelliteId', 'missionId', 'telemetryType']
    for field in required_fields:
        if pd.isna(row.get(field)) or row.get(field) == '':
            score -= 20

    # Check data payload
    if 'data' not in row or not row['data']:
        score -= 30

    # Timestamp validity
    try:
        if row.get('timestamp'):
            ts = pd.to_datetime(row['timestamp'], unit='ms')
            now = datetime.utcnow()
            if abs((now - ts.to_pydatetime()).total_seconds()) > 86400:  # 24 hours
                score -= 15
    except:
        score -= 25

    return max(0, score)

def load_to_warm_path(**context):
    """Load transformed data to PostgreSQL warm path"""
    try:
        from airflow.hooks.postgres_hook import PostgresHook

        # Get transformed data
        transformed_data = context['task_instance'].xcom_pull(
            key='transformed_data',
            task_ids='transform_data'
        )

        if not transformed_data:
            logger.warning("No data to load to warm path")
            return 0

        # Connect to PostgreSQL
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

        # Prepare data for bulk insert
        df = pd.DataFrame(transformed_data)

        # Insert data using pandas to_sql
        df.to_sql(
            'telemetry_processed',
            postgres_hook.get_sqlalchemy_engine(),
            if_exists='append',
            index=False,
            method='multi'
        )

        logger.info(f"Loaded {len(transformed_data)} records to warm path")
        return len(transformed_data)

    except Exception as e:
        logger.error(f"Error loading to warm path: {str(e)}")
        raise

def load_to_cold_path(**context):
    """Load raw data to MinIO cold storage"""
    try:
        import boto3
        from botocore.client import Config

        # Get original hot path data
        hot_path_data = context['task_instance'].xcom_pull(
            key='hot_path_data',
            task_ids='extract_hot_path'
        )

        if not hot_path_data:
            logger.warning("No data to archive to cold path")
            return 0

        # Configure MinIO client
        s3_client = boto3.client(
            's3',
            endpoint_url='http://minio:9000',
            aws_access_key_id='minio',
            aws_secret_access_key='minio123',
            config=Config(signature_version='s3v4')
        )

        # Create archive file
        execution_date = context['execution_date']
        archive_key = f"telemetry/year={execution_date.year}/month={execution_date.month:02d}/day={execution_date.day:02d}/hour={execution_date.hour:02d}/telemetry_data.json"

        # Upload data as JSON
        s3_client.put_object(
            Bucket='telemetry-archive',
            Key=archive_key,
            Body=json.dumps(hot_path_data, indent=2),
            ContentType='application/json'
        )

        logger.info(f"Archived {len(hot_path_data)} records to cold path: {archive_key}")
        return len(hot_path_data)

    except Exception as e:
        logger.error(f"Error loading to cold path: {str(e)}")
        raise

def load_to_analytics_path(**context):
    """Load processed data to Vector DB for ML/AI analytics"""
    try:
        # Get transformed data
        transformed_data = context['task_instance'].xcom_pull(
            key='transformed_data',
            task_ids='transform_data'
        )

        if not transformed_data:
            logger.warning("No data to load to analytics path")
            return 0

        # This would integrate with your vector database
        # For example: Pinecone, Weaviate, or Chroma
        # Here we'll simulate the process

        analytics_records = []
        for record in transformed_data:
            # Create embeddings for numerical data
            if 'temperature' in record or 'pressure' in record or 'voltage' in record:
                vector_data = {
                    'id': f"{record['satelliteId']}_{record['timestamp']}",
                    'metadata': {
                        'satellite_id': record['satelliteId'],
                        'mission_id': record['missionId'],
                        'telemetry_type': record['telemetryType'],
                        'timestamp': record['timestamp'],
                        'quality_score': record.get('data_quality_score', 0)
                    },
                    'values': extract_numerical_features(record)
                }
                analytics_records.append(vector_data)

        # In a real implementation, you would insert into vector DB here
        # vector_db.upsert(vectors=analytics_records, namespace="telemetry")

        logger.info(f"Prepared {len(analytics_records)} records for analytics path")
        return len(analytics_records)

    except Exception as e:
        logger.error(f"Error loading to analytics path: {str(e)}")
        raise

def extract_numerical_features(record):
    """Extract numerical features for vector embedding"""
    features = []

    # Extract common telemetry values
    numerical_fields = ['temperature', 'pressure', 'voltage', 'current', 'power', 'altitude', 'velocity']

    for field in numerical_fields:
        value = record.get(field, 0)
        if isinstance(value, (int, float)):
            features.append(float(value))
        else:
            features.append(0.0)

    # Pad or truncate to fixed size (e.g., 128 dimensions)
    target_size = 128
    if len(features) < target_size:
        features.extend([0.0] * (target_size - len(features)))
    else:
        features = features[:target_size]

    return features

def generate_quality_report(**context):
    """Generate data quality report"""
    try:
        # Get processing results from all paths
        hot_count = context['task_instance'].xcom_pull(task_ids='extract_hot_path')
        warm_count = context['task_instance'].xcom_pull(task_ids='load_warm_path')
        cold_count = context['task_instance'].xcom_pull(task_ids='load_cold_path')
        analytics_count = context['task_instance'].xcom_pull(task_ids='load_analytics_path')

        execution_date = context['execution_date']

        report = {
            'pipeline_run': {
                'execution_date': execution_date.isoformat(),
                'status': 'completed',
                'duration_minutes': 0  # Will be calculated in monitoring
            },
            'data_volumes': {
                'hot_path_extracted': hot_count or 0,
                'warm_path_loaded': warm_count or 0,
                'cold_path_archived': cold_count or 0,
                'analytics_path_prepared': analytics_count or 0
            },
            'data_quality': {
                'extraction_success_rate': (warm_count / hot_count * 100) if hot_count else 0,
                'transformation_success_rate': 100,  # Calculate based on errors
                'loading_success_rate': 100  # Calculate based on errors
            }
        }

        # Store report (could be sent to monitoring system)
        logger.info(f"Pipeline quality report: {json.dumps(report, indent=2)}")

        return report

    except Exception as e:
        logger.error(f"Error generating quality report: {str(e)}")
        raise

# Define tasks
extract_hot_path = PythonOperator(
    task_id='extract_hot_path',
    python_callable=extract_hot_path_data,
    dag=dag,
    doc_md="Extract telemetry data from Redis hot path for the current hour window"
)

transform_data = PythonOperator(
    task_id='transform_data',
    python_callable=transform_telemetry_data,
    dag=dag,
    doc_md="Clean, validate, and transform telemetry data with quality scoring"
)

load_warm_path = PythonOperator(
    task_id='load_warm_path',
    python_callable=load_to_warm_path,
    dag=dag,
    doc_md="Load processed data to PostgreSQL for batch analytics"
)

load_cold_path = PythonOperator(
    task_id='load_cold_path',
    python_callable=load_to_cold_path,
    dag=dag,
    doc_md="Archive raw data to MinIO for long-term storage"
)

load_analytics_path = PythonOperator(
    task_id='load_analytics_path',
    python_callable=load_to_analytics_path,
    dag=dag,
    doc_md="Prepare data for ML/AI analytics in vector database"
)

create_tables = PostgresOperator(
    task_id='create_tables_if_not_exists',
    postgres_conn_id='postgres_default',
    sql="""
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

    CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_processed(timestamp);
    CREATE INDEX IF NOT EXISTS idx_telemetry_satellite ON telemetry_processed(satellite_id);
    CREATE INDEX IF NOT EXISTS idx_telemetry_mission ON telemetry_processed(mission_id);
    """,
    dag=dag,
    doc_md="Create necessary database tables and indexes if they don't exist"
)

quality_report = PythonOperator(
    task_id='generate_quality_report',
    python_callable=generate_quality_report,
    dag=dag,
    doc_md="Generate comprehensive data quality and processing report"
)

# Define task dependencies
create_tables >> extract_hot_path
extract_hot_path >> transform_data
transform_data >> [load_warm_path, load_cold_path, load_analytics_path]
[load_warm_path, load_cold_path, load_analytics_path] >> quality_report

# Add data validation sensors
redis_sensor = RedisKeySensor(
    task_id='check_redis_availability',
    redis_conn_id='redis_default',
    key='telemetry:*',
    dag=dag,
    timeout=300,
    poke_interval=30,
    doc_md="Ensure Redis contains telemetry data before processing"
)

redis_sensor >> extract_hot_path
