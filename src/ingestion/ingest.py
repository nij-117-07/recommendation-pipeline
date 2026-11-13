"""Stage 1: data ingestion.

Reads users.csv, items.csv and interactions.csv from data/raw. If the files are
missing, the source named in the config is used: "movielens" downloads MovieLens 100K
and converts it (see movielens.py); otherwise a synthetic MovieLens-style dataset is
generated (seeded, reproducible) with deliberate data-quality problems so the
validation stage has real errors to detect. Real CSV exports can be dropped into
data/raw instead.
"""

import hashlib
import json

import numpy as np
import pandas as pd

from src.common.config import RAW_DIR, ensure_dirs, get_logger
from src.ingestion import movielens

log = get_logger(__name__)

RAW_FILES = ["users.csv", "items.csv", "interactions.csv"]
GENRES = ["Action", "Animation", "Children", "Comedy", "Documentary",
          "Drama", "Horror", "Romance", "Sci-Fi", "Thriller"]
OCCUPATIONS = ["student", "engineer", "teacher", "artist", "doctor", "other"]
START = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp())
END = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())


def _generate_users(rng, n_users):
    signup = pd.to_datetime(rng.integers(START - 5 * 365 * 86400, START, n_users), unit="s")
    return pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "age": rng.integers(13, 75, n_users),
        "gender": rng.choice(["M", "F"], n_users),
        "occupation": rng.choice(OCCUPATIONS, n_users),
        "signup_date": signup.strftime("%Y-%m-%d"),
    })


def _generate_items(rng, n_items):
    n_genres = rng.integers(1, 4, n_items)
    genres = ["|".join(rng.choice(GENRES, size=k, replace=False)) for k in n_genres]
    return pd.DataFrame({
        "item_id": np.arange(1, n_items + 1),
        "title": [f"Item {i:04d}" for i in range(1, n_items + 1)],
        "genres": genres,
        "release_year": rng.integers(1980, 2025, n_items),
    })


def _generate_interactions(rng, n_users, n_items, n_interactions):
    latent_dim = 8
    user_factors = rng.normal(size=(n_users, latent_dim))
    item_factors = rng.normal(size=(n_items, latent_dim))
    popularity = rng.pareto(1.5, n_items) + 1
    popularity /= popularity.sum()

    n_draw = int(n_interactions * 1.6)
    pairs = pd.DataFrame({
        "user_id": rng.integers(1, n_users + 1, n_draw),
        "item_id": rng.choice(n_items, size=n_draw, p=popularity) + 1,
    }).drop_duplicates(["user_id", "item_id"]).head(n_interactions).copy()

    affinity = (user_factors[pairs.user_id.to_numpy() - 1] * item_factors[pairs.item_id.to_numpy() - 1]).sum(axis=1)
    z = (affinity - affinity.mean()) / affinity.std()
    pairs["rating"] = np.clip(np.round(3 + 1.1 * z + rng.normal(0, 0.5, len(pairs))), 1, 5).astype(int)
    pairs["timestamp"] = pd.to_datetime(rng.integers(START, END, len(pairs)), unit="s", utc=True)
    pairs["timestamp"] = pairs["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return pairs.sort_values("timestamp").reset_index(drop=True)


def _inject_user_errors(rng, df, rate):
    df = df.astype(object)
    n = max(1, int(len(df) * rate))
    groups = np.array_split(rng.permutation(len(df))[: n * 2], 2)
    df.loc[groups[0], "age"] = np.nan
    df.loc[groups[1], "age"] = rng.choice([-5, 0, 150], len(groups[1]))
    duplicates = df.iloc[rng.integers(0, len(df), n)]
    return pd.concat([df, duplicates], ignore_index=True)


def _inject_item_errors(rng, df, rate):
    df = df.astype(object)
    n = max(1, int(len(df) * rate))
    groups = np.array_split(rng.permutation(len(df))[: n * 2], 2)
    df.loc[groups[0], "genres"] = np.nan
    df.loc[groups[1], "release_year"] = rng.choice(["TBD", "unknown"], len(groups[1]))
    duplicates = df.iloc[rng.integers(0, len(df), n)]
    return pd.concat([df, duplicates], ignore_index=True)


def _inject_interaction_errors(rng, df, rate, n_users, n_items):
    df = df.astype(object)
    n = max(1, int(len(df) * rate))
    groups = np.array_split(rng.permutation(len(df))[: n * 7], 7)
    df.loc[groups[0], "item_id"] = np.nan
    df.loc[groups[1], "rating"] = np.nan
    df.loc[groups[2], "rating"] = rng.choice([0, 6, -1], len(groups[2]))
    df.loc[groups[3], "rating"] = rng.choice(["four", "five stars", "unknown"], len(groups[3]))
    df.loc[groups[4], "timestamp"] = rng.choice(["2024-13-45", "not-a-date", "31/02/2023"], len(groups[4]))
    df.loc[groups[5], "item_id"] = n_items + 1000 + np.arange(len(groups[5]))
    df.loc[groups[6], "user_id"] = n_users + 5000 + np.arange(len(groups[6]))
    duplicates = df.iloc[rng.integers(0, len(df), n)]
    return pd.concat([df, duplicates], ignore_index=True)


def generate_synthetic_dataset(cfg):
    s = cfg["synthetic"]
    rng = np.random.default_rng(cfg["random_seed"])

    users = _generate_users(rng, s["n_users"])
    items = _generate_items(rng, s["n_items"])
    interactions = _generate_interactions(rng, s["n_users"], s["n_items"], s["n_interactions"])

    users = _inject_user_errors(rng, users, s["error_rate"])
    items = _inject_item_errors(rng, items, s["error_rate"])
    interactions = _inject_interaction_errors(rng, interactions, s["error_rate"], s["n_users"], s["n_items"])

    users.to_csv(RAW_DIR / "users.csv", index=False)
    items.to_csv(RAW_DIR / "items.csv", index=False)
    interactions.to_csv(RAW_DIR / "interactions.csv", index=False)
    log.info("Generated synthetic raw data: %d users, %d items, %d interaction rows",
             len(users), len(items), len(interactions))


def _write_manifest():
    manifest = {}
    for name in RAW_FILES:
        path = RAW_DIR / name
        manifest[name] = {
            "rows": int(len(pd.read_csv(path, dtype=str))),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run(cfg):
    ensure_dirs()
    if all((RAW_DIR / name).exists() for name in RAW_FILES):
        log.info("Using existing raw CSV files in %s", RAW_DIR)
    elif cfg.get("source") == "movielens":
        movielens.prepare_raw_files()
    else:
        generate_synthetic_dataset(cfg)
    manifest = _write_manifest()
    log.info("Raw data manifest: %s", {k: v["rows"] for k, v in manifest.items()})
