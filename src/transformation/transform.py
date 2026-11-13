"""Stage 4: data transformation.

Builds derived datasets from the training split only:
  - user_activity.parquet     per-user interaction summary
  - item_popularity.parquet   per-item popularity summary
  - user_item_matrix.parquet  user x item rating matrix (NaN = unobserved)
"""

import pandas as pd

from src.common.config import PROCESSED_DIR, ensure_dirs, get_logger

log = get_logger(__name__)


def build_user_activity(train, threshold):
    flagged = train.assign(is_positive=(train["rating"] >= threshold).astype(int))
    return flagged.groupby("user_id").agg(
        n_interactions=("rating", "size"),
        n_positive=("is_positive", "sum"),
        avg_rating=("rating", "mean"),
        first_interaction=("timestamp", "min"),
        last_interaction=("timestamp", "max"),
    )


def build_item_popularity(train, threshold):
    flagged = train.assign(is_positive=(train["rating"] >= threshold).astype(int))
    popularity = flagged.groupby("item_id").agg(
        n_interactions=("rating", "size"),
        n_positive=("is_positive", "sum"),
        avg_rating=("rating", "mean"),
    )
    popularity["popularity_score"] = popularity["n_positive"] / popularity["n_positive"].max()
    return popularity


def build_user_item_matrix(train):
    matrix = train.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean")
    matrix.columns = matrix.columns.astype(str)
    return matrix


def run(cfg):
    ensure_dirs()
    threshold = cfg["relevance_threshold"]
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")

    activity = build_user_activity(train, threshold)
    popularity = build_item_popularity(train, threshold)
    matrix = build_user_item_matrix(train)

    activity.to_parquet(PROCESSED_DIR / "user_activity.parquet")
    popularity.to_parquet(PROCESSED_DIR / "item_popularity.parquet")
    matrix.to_parquet(PROCESSED_DIR / "user_item_matrix.parquet")

    density = matrix.notna().to_numpy().mean()
    log.info("Transformed data: %d users, %d items, matrix density %.2f%%",
             matrix.shape[0], matrix.shape[1], density * 100)
