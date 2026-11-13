import os

import mlflow

from src.common.config import MLRUNS_DIR, ROOT


def configure_mlflow(cfg):
    db_path = os.environ.get("MLFLOW_TRACKING_DB", str(ROOT / "mlflow.db"))
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    name = cfg["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=MLRUNS_DIR.as_uri())
    mlflow.set_experiment(name)
    return mlflow
