"""Airflow DAG for the recommendation pipeline.

Each task calls one stage from src/orchestration/stages.py, the same list used by
dags/recommendation_pipeline.py, so both runners execute the same stages in the
same order.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.common.config import ensure_dirs, load_config
from src.orchestration.stages import STAGES


def _stage_callable(stage_fn):
    def _run():
        ensure_dirs()
        stage_fn(load_config())
    return _run


default_args = {
    "owner": "data-management",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="recommendation_pipeline",
    description="Metadata -> ingestion -> validation -> preparation -> transformation -> features -> training -> evaluation",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["recommendation", "assignment2"],
) as dag:
    tasks = {}
    for stage in STAGES:
        tasks[stage.name] = PythonOperator(task_id=stage.name, python_callable=_stage_callable(stage.run))

    for stage in STAGES:
        for upstream in stage.deps:
            tasks[upstream] >> tasks[stage.name]
