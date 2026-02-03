FROM apache/airflow:2.9.3

# Install system dependencies for LightGBM and other packages
USER root
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch to airflow user before upgrading pip/setuptools
USER airflow

# Upgrade setuptools to fix Python 3.12 compatibility
RUN pip install --upgrade pip setuptools wheel

# Copy and install Python requirements
COPY requirements-airflow.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

# Copy DAGs, source code and scripts
USER root
COPY dags /opt/airflow/dags
COPY src /opt/airflow/src
COPY scripts /opt/airflow/scripts
COPY config.yaml /opt/airflow/config.yaml
RUN chown -R airflow:root /opt/airflow/dags /opt/airflow/src /opt/airflow/scripts /opt/airflow/config.yaml
USER airflow