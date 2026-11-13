"""Stage 3: data preparation.

Cleans the validated tables (imputes missing values, standardises formats,
removes repeated user-item pairs) and creates a time-based train/test split.
The split cutoff is the timestamp at the (1 - test_fraction) position, so
the model never sees interactions that happen after the test period starts.
"""

import json

import pandas as pd

from src.common.config import PROCESSED_DIR, ensure_dirs, get_logger

log = get_logger(__name__)


def _clean_users(users):
    users = users.copy()
    users["age"] = users["age"].fillna(users["age"].median())
    users["gender"] = users["gender"].fillna("unknown").str.strip().str.lower()
    users["occupation"] = users["occupation"].fillna("unknown").str.strip().str.lower()
    users["signup_date"] = pd.to_datetime(users["signup_date"], format="%Y-%m-%d", errors="coerce")
    return users


def _standardise_genres(value):
    if pd.isna(value):
        return "unknown"
    genres = sorted({g.strip().title() for g in str(value).split("|") if g.strip()})
    return "|".join(genres) if genres else "unknown"


def _clean_items(items):
    items = items.copy()
    items["title"] = items["title"].fillna("Unknown").str.strip()
    items["genres"] = items["genres"].map(_standardise_genres)
    items["release_year"] = items["release_year"].fillna(items["release_year"].median()).astype(int)
    return items


def _clean_interactions(inter, cfg):
    n_raw = len(inter)
    inter = inter.sort_values("timestamp", kind="stable")
    inter = inter.drop_duplicates(["user_id", "item_id"], keep="last")
    n_dedup = n_raw - len(inter)
    inter = inter.sort_values("timestamp", kind="stable").reset_index(drop=True)

    test_fraction = cfg["split"]["test_fraction"]
    cutoff_position = int(len(inter) * (1 - test_fraction))
    cutoff = inter["timestamp"].iloc[cutoff_position]
    train = inter[inter["timestamp"] < cutoff].reset_index(drop=True)
    test = inter[inter["timestamp"] >= cutoff].reset_index(drop=True)
    return inter, train, test, cutoff, n_dedup


def run(cfg):
    ensure_dirs()
    users = pd.read_parquet(PROCESSED_DIR / "users_valid.parquet")
    items = pd.read_parquet(PROCESSED_DIR / "items_valid.parquet")
    inter = pd.read_parquet(PROCESSED_DIR / "interactions_valid.parquet")

    users = _clean_users(users)
    items = _clean_items(items)
    clean, train, test, cutoff, n_dedup = _clean_interactions(inter, cfg)

    users.to_parquet(PROCESSED_DIR / "users_clean.parquet", index=False)
    items.to_parquet(PROCESSED_DIR / "items_clean.parquet", index=False)
    clean.to_parquet(PROCESSED_DIR / "interactions_clean.parquet", index=False)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    split_info = {
        "split_type": "global time-based",
        "cutoff": str(cutoff),
        "test_fraction": cfg["split"]["test_fraction"],
        "duplicate_user_item_pairs_removed": int(n_dedup),
        "interactions_clean": len(clean),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_users": int(train["user_id"].nunique()),
        "test_users": int(test["user_id"].nunique()),
        "test_users_unseen_in_train": len(set(test["user_id"]) - set(train["user_id"])),
    }
    (PROCESSED_DIR / "split_info.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    log.info("Prepared data: %d train rows, %d test rows, cutoff %s, %d duplicate pairs removed",
             len(train), len(test), cutoff, n_dedup)
