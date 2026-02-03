"""
Airflow DAG for Fraud Detection Pipeline (Kubernetes Version).

This version uses KubernetesPodOperator for running tasks in Kubernetes pods.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.dates import days_ago
from kubernetes.client import models as k8s

# Default arguments for the DAG
default_args = {
    'owner': 'data-science',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Kubernetes configuration
IMAGE = 'fraud-detection-airflow:v1'
NAMESPACE = 'airflow'

# Volume configuration for shared data
volume = k8s.V1Volume(
    name='fraud-data',
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name='fraud-data-pvc'),
)

volume_mount = k8s.V1VolumeMount(
    name='fraud-data',
    mount_path='/opt/airflow/data',
    sub_path=None,
    read_only=False
)

# Define the DAG
dag = DAG(
    'fraud_detection_pipeline_k8s',
    default_args=default_args,
    description='End-to-end fraud detection pipeline (Kubernetes)',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['fraud-detection', 'machine-learning', 'kubernetes'],
)

# Task 1: Setup
setup_task = KubernetesPodOperator(
    task_id='setup_directories',
    name='setup-directories',
    namespace=NAMESPACE,
    image=IMAGE,
    image_pull_policy='Never',
    cmds=['python3'],
    arguments=['/opt/airflow/scripts/run_setup.py'],
    volumes=[volume],
    volume_mounts=[volume_mount],
    get_logs=True,
    is_delete_operator_pod=True,
    dag=dag,
)

# Task 2: Data Ingestion
data_ingestion_task = KubernetesPodOperator(
    task_id='data_ingestion',
    name='data-ingestion',
    namespace=NAMESPACE,
    image=IMAGE,
    image_pull_policy='Never',
    cmds=['python3'],
    arguments=['/opt/airflow/scripts/run_data_ingestion.py'],
    volumes=[volume],
    volume_mounts=[volume_mount],
    get_logs=True,
    is_delete_operator_pod=True,
    dag=dag,
)

# Task 3: Data Cleaning
data_cleaning_task = KubernetesPodOperator(
    task_id='data_cleaning',
    name='data-cleaning',
    namespace=NAMESPACE,
    image=IMAGE,
    image_pull_policy='Never',
    cmds=['python3'],
    arguments=['/opt/airflow/scripts/run_data_cleaning.py'],
    volumes=[volume],
    volume_mounts=[volume_mount],
    get_logs=True,
    is_delete_operator_pod=True,
    dag=dag,
)

# Task 4: Feature Engineering
feature_engineering_task = KubernetesPodOperator(
    task_id='feature_engineering',
    name='feature-engineering',
    namespace=NAMESPACE,
    image=IMAGE,
    image_pull_policy='Never',
    cmds=['python3'],
    arguments=['/opt/airflow/scripts/run_feature_engineering.py'],
    volumes=[volume],
    volume_mounts=[volume_mount],
    get_logs=True,
    is_delete_operator_pod=True,
    dag=dag,
)

# Task 5: Model Training
model_training_task = KubernetesPodOperator(
    task_id='model_training',
    name='model-training',
    namespace=NAMESPACE,
    image=IMAGE,
    image_pull_policy='Never',
    cmds=['python3'],
    arguments=['/opt/airflow/scripts/run_model_training.py'],
    volumes=[volume],
    volume_mounts=[volume_mount],
    env_vars={'MLFLOW_TRACKING_URI': 'http://mlflow:5000'},
    get_logs=True,
    is_delete_operator_pod=True,
    dag=dag,
)

# Define task dependencies
setup_task >> data_ingestion_task >> data_cleaning_task >> feature_engineering_task >> model_training_task
