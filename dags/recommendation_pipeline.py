"""Workflow definition for the end-to-end recommendation pipeline.

Run once:        python -m dags.recommendation_pipeline
Run on a schedule: see the README (cron example).

The stages and their dependencies come from src/orchestration/stages.py, which
is shared with the Airflow DAG in dags/recommendation_dag.py.
"""

from src.common.config import REPORTS_DIR, ensure_dirs, load_config
from src.orchestration.stages import STAGES
from src.orchestration.workflow import Task, run_workflow


def build_dag():
    return {stage.name: Task(stage.name, stage.run, deps=stage.deps, retries=stage.retries)
            for stage in STAGES}


def main():
    cfg = load_config()
    ensure_dirs()
    run_workflow(build_dag(), cfg, log_file=REPORTS_DIR / "run_log.txt")


if __name__ == "__main__":
    main()
