"""Single definition of the pipeline stages and their dependencies.

Both runners use this list: the plain-Python runner (dags/recommendation_pipeline.py)
and the Airflow DAG (dags/recommendation_dag.py). Changing a stage or dependency
here changes both.
"""

from collections import namedtuple

from src.common import reproducibility
from src.evaluation import evaluate
from src.features import feature_store
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.validation import validate

Stage = namedtuple("Stage", ["name", "run", "deps", "retries"])

STAGES = (
    Stage("record_run_metadata", reproducibility.run, (), 0),
    Stage("ingest_data", ingest.run, ("record_run_metadata",), 1),
    Stage("validate_data", validate.run, ("ingest_data",), 0),
    Stage("prepare_data", prepare.run, ("validate_data",), 0),
    Stage("transform_data", transform.run, ("prepare_data",), 0),
    Stage("build_features", feature_store.run, ("transform_data",), 0),
    Stage("train_models", train.run, ("build_features",), 0),
    Stage("evaluate_models", evaluate.run, ("train_models",), 0),
)
