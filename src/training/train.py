"""Stage 6: model training.

Trains a popularity baseline, a genre-based content recommender (reads the
feature store) and a personalised biased matrix factorization model. Parameters,
per-epoch metrics and the pickled models are logged to MLflow. The run IDs are
stored in models/registry.json so the evaluation stage can attach its metrics
to the same MLflow runs.
"""

import json
import pickle

import pandas as pd

from src.common.config import MODELS_DIR, PROCESSED_DIR, ensure_dirs, get_logger
from src.common.tracking import configure_mlflow
from src.features.feature_store import FeatureStore
from src.training.models import BiasedMatrixFactorization, GenreContentRecommender, PopularityRecommender

log = get_logger(__name__)


def run(cfg):
    ensure_dirs()
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    store = FeatureStore()
    user_features = store.get_user_features()
    item_features = store.get_item_features()
    global_mean = train["rating"].mean()
    tracker = configure_mlflow(cfg)

    popularity = PopularityRecommender().fit(item_features, global_mean)
    content = GenreContentRecommender(fallback=popularity,
                                      popularity_weight=cfg["content"]["popularity_weight"]).fit(
        user_features, item_features, global_mean)
    mf_params = cfg["mf"]
    mf = BiasedMatrixFactorization(
        n_factors=mf_params["n_factors"],
        epochs=mf_params["epochs"],
        learning_rate=mf_params["learning_rate"],
        regularization=mf_params["regularization"],
        seed=cfg["random_seed"],
        fallback=popularity,
    ).fit(train)

    registry = {}
    for model in (popularity, content, mf):
        path = MODELS_DIR / f"{model.name}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(model, fh)

        with tracker.start_run(run_name=model.name) as active_run:
            tracker.log_params(model.params())
            tracker.log_param("train_rows", len(train))
            tracker.log_param("relevance_threshold", cfg["relevance_threshold"])
            for entry in getattr(model, "history", []):
                tracker.log_metric("train_rmse", entry["train_rmse"], step=entry["epoch"])
            tracker.log_artifact(str(path), artifact_path="model")
            registry[model.name] = {"path": str(path), "run_id": active_run.info.run_id}
        log.info("Trained %s and logged MLflow run %s", model.name, active_run.info.run_id)

    (MODELS_DIR / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
