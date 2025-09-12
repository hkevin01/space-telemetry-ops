FROM apache/airflow:2.8.0-python3.10

# Switch to root to install system dependencies
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
  gcc \
  g++ \
  && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy DAGs and plugins
COPY dags/ /opt/airflow/dags/
COPY plugins/ /opt/airflow/plugins/
COPY config/ /opt/airflow/config/

# Set environment variables
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True
ENV AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL=30
ENV AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG=1

# Create necessary directories
USER root
RUN mkdir -p /opt/airflow/logs && \
  mkdir -p /opt/airflow/plugins && \
  mkdir -p /opt/airflow/config && \
  chown -R airflow:root /opt/airflow

USER airflow

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}"

# Default command
CMD ["airflow", "webserver"]
