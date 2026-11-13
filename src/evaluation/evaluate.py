"""Stage 7: model evaluation.

Ranking metrics (Precision@K, Recall@K, NDCG@K, coverage, diversity) are computed
per user on the test split. An item is "relevant" if its test rating is at least
the relevance threshold. Items the user already rated in training are excluded
from recommendations. RMSE and MAE are computed on test rows whose user and
item both appear in training.
"""

import json
import pickle

import pandas as pd

from src.common.config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, get_logger
from src.common.tracking import configure_mlflow
from src.evaluation.metrics import (
    catalog_coverage,
    intra_list_diversity,
    mae,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)

log = get_logger(__name__)


def _to_markdown(df):
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(cfg):
    ensure_dirs()
    k = cfg["top_k"]
    threshold = cfg["relevance_threshold"]
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    items = pd.read_parquet(PROCESSED_DIR / "items_clean.parquet")
    registry = json.loads((MODELS_DIR / "registry.json").read_text(encoding="utf-8"))

    item_genres = {int(i): frozenset(g.split("|")) for i, g in zip(items["item_id"], items["genres"])}
    catalog_size = len(items)
    relevant = {int(u): set(int(i) for i in g)
                for u, g in test[test["rating"] >= threshold].groupby("user_id")["item_id"]}
    seen = {int(u): set(int(i) for i in g) for u, g in train.groupby("user_id")["item_id"]}
    train_users = set(seen)
    train_items = set(int(i) for i in train["item_id"])

    eval_users = sorted(relevant)
    cold_users = sum(1 for u in eval_users if u not in train_users)
    rating_rows = test[test["user_id"].isin(train_users) & test["item_id"].isin(train_items)]
    log.info("Evaluating %d users (%d cold-start) and %d rating rows at K=%d",
             len(eval_users), cold_users, len(rating_rows), k)

    rows = []
    for name, meta in registry.items():
        with open(meta["path"], "rb") as fh:
            model = pickle.load(fh)

        recs = {u: model.recommend(u, k, seen.get(u, set())) for u in eval_users}
        precision = [precision_at_k(recs[u], relevant[u], k) for u in eval_users]
        recall = [recall_at_k(recs[u], relevant[u], k) for u in eval_users]
        ndcg = [ndcg_at_k(recs[u], relevant[u], k) for u in eval_users]
        diversity = [intra_list_diversity(recs[u], item_genres) for u in eval_users]
        all_recs = [i for u in eval_users for i in recs[u]]

        predictions = [model.predict(int(u), int(i))
                       for u, i in zip(rating_rows["user_id"], rating_rows["item_id"])]
        actual = rating_rows["rating"].tolist()

        rows.append({
            "model": model.name,
            "run_id": meta["run_id"],
            "metrics": {
                "precision_at_k": sum(precision) / len(precision),
                "recall_at_k": sum(recall) / len(recall),
                "ndcg_at_k": sum(ndcg) / len(ndcg),
                "coverage": catalog_coverage(all_recs, catalog_size),
                "diversity": sum(diversity) / len(diversity),
                "rmse": rmse(actual, predictions),
                "mae": mae(actual, predictions),
            },
        })
        log.info("%s: %s", model.name, {m: round(v, 4) for m, v in rows[-1]["metrics"].items()})

    results = pd.DataFrame([
        {
            "Model": row["model"],
            f"Precision@{k}": row["metrics"]["precision_at_k"],
            f"Recall@{k}": row["metrics"]["recall_at_k"],
            f"NDCG@{k}": row["metrics"]["ndcg_at_k"],
            "Coverage": row["metrics"]["coverage"],
            "Diversity": row["metrics"]["diversity"],
            "RMSE": row["metrics"]["rmse"],
            "MAE": row["metrics"]["mae"],
        }
        for row in rows
    ])
    results_csv = REPORTS_DIR / "model_results.csv"
    results.to_csv(results_csv, index=False)

    summary = {
        "k": k,
        "relevance_threshold": threshold,
        "evaluated_users": len(eval_users),
        "cold_start_users": cold_users,
        "rating_rows_for_rmse_mae": len(rating_rows),
        "models": {row["model"]: row["metrics"] for row in rows},
    }
    (REPORTS_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Model Evaluation Results",
        "",
        f"K = {k}, relevance threshold = rating >= {threshold}. Evaluated users: {len(eval_users)} "
        f"({cold_users} cold-start). RMSE/MAE use {len(rating_rows)} test ratings with known users and items.",
        "",
        _to_markdown(results),
        "",
    ]
    (REPORTS_DIR / "model_results.md").write_text("\n".join(report), encoding="utf-8")

    tracker = configure_mlflow(cfg)
    for row in rows:
        with tracker.start_run(run_id=row["run_id"]):
            tracker.log_metrics(row["metrics"])
            tracker.log_artifact(str(results_csv), artifact_path="evaluation")
    log.info("Logged evaluation metrics to MLflow")
    log.info("Results:\n%s", _to_markdown(results))
