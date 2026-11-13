"""Stage 5: feature management.

The same feature functions are used for training (train split) and prediction.
Features are persisted to data/features and documented in
reports/feature_catalog.md. Models read features only through FeatureStore, so
training and prediction always see identical feature definitions.
"""

from pathlib import Path

import pandas as pd

from src.common.config import FEATURES_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs, get_logger

log = get_logger(__name__)

GENRE_FLAG_PREFIX = "item_genre_"
GENRE_AFFINITY_PREFIX = "user_genre_affinity_"

FEATURE_CATALOG = [
    {"name": "user_avg_rating", "entity": "user", "dtype": "float",
     "source": "train interactions",
     "description": "Mean explicit rating given by the user. Global mean if the user has no training history."},
    {"name": "user_activity_count", "entity": "user", "dtype": "int",
     "source": "train interactions",
     "description": "Number of training interactions by the user (activity level)."},
    {"name": "user_preferred_category", "entity": "user", "dtype": "string",
     "source": "train interactions + item genres",
     "description": "Genre that occurs most often among the items the user interacted with (ties broken alphabetically)."},
    {"name": "user_days_since_last_interaction", "entity": "user", "dtype": "float",
     "source": "train interactions",
     "description": "Days between the feature reference time (last training timestamp) and the user's last interaction. -1 if no history."},
    {"name": "item_popularity_count", "entity": "item", "dtype": "int",
     "source": "train interactions",
     "description": "Number of training interactions with the item."},
    {"name": "item_positive_count", "entity": "item", "dtype": "int",
     "source": "train interactions",
     "description": "Number of training interactions where rating >= relevance threshold (positive feedback)."},
    {"name": "item_avg_rating", "entity": "item", "dtype": "float",
     "source": "train interactions",
     "description": "Mean rating of the item in training. Global mean if the item is unseen."},
    {"name": "item_primary_genre", "entity": "item", "dtype": "string",
     "source": "items table",
     "description": "First genre (alphabetical) of the item."},
    {"name": "item_n_genres", "entity": "item", "dtype": "int",
     "source": "items table",
     "description": "Number of genres assigned to the item."},
    {"name": "item_age_years", "entity": "item", "dtype": "int",
     "source": "items table",
     "description": "Reference year minus release year."},
    {"name": "interaction_hour_of_day", "entity": "interaction", "dtype": "int",
     "source": "train interactions",
     "description": "Hour (0-23 UTC) at which the interaction happened."},
    {"name": "interaction_day_of_week", "entity": "interaction", "dtype": "int",
     "source": "train interactions",
     "description": "Day of week of the interaction (0 = Monday)."},
    {"name": "label_is_positive", "entity": "interaction", "dtype": "int",
     "source": "train interactions",
     "description": "Target label (not a model input): 1 if rating >= relevance threshold, else 0."},
]

FILES = {
    "user": "user_features.parquet",
    "item": "item_features.parquet",
    "interaction": "interaction_features.parquet",
}


def genre_vocabulary(items):
    return sorted({g for value in items["genres"] for g in value.split("|")} - {"unknown"})


def _preferred_category(train, items):
    exploded = train[["user_id", "item_id"]].merge(items[["item_id", "genres"]], on="item_id", how="left")
    exploded = exploded.assign(genre=exploded["genres"].str.split("|")).explode("genre")
    counts = exploded.groupby(["user_id", "genre"]).size().reset_index(name="n")
    counts = counts.sort_values(["user_id", "n", "genre"], ascending=[True, False, True])
    return counts.drop_duplicates("user_id").set_index("user_id")["genre"]


def build_user_features(train, items, users, as_of, vocabulary, threshold):
    grouped = train.groupby("user_id")
    features = pd.DataFrame({
        "user_avg_rating": grouped["rating"].mean(),
        "user_activity_count": grouped.size(),
        "user_days_since_last_interaction": (as_of - grouped["timestamp"].max()).dt.total_seconds() / 86400,
        "user_preferred_category": _preferred_category(train, items),
    })
    features = users[["user_id"]].set_index("user_id").join(features)
    features["user_avg_rating"] = features["user_avg_rating"].fillna(train["rating"].mean())
    features["user_activity_count"] = features["user_activity_count"].fillna(0).astype(int)
    features["user_days_since_last_interaction"] = features["user_days_since_last_interaction"].fillna(-1.0)
    features["user_preferred_category"] = features["user_preferred_category"].fillna("unknown")
    return features.join(build_user_genre_affinity(train, items, users, vocabulary, threshold))


def build_user_genre_affinity(train, items, users, vocabulary, threshold):
    positives = train.loc[train["rating"] >= threshold, ["user_id", "item_id"]]
    exploded = positives.merge(items[["item_id", "genres"]], on="item_id", how="left")
    exploded = exploded.assign(genre=exploded["genres"].str.split("|")).explode("genre")
    counts = pd.crosstab(exploded["user_id"], exploded["genre"]).reindex(columns=vocabulary, fill_value=0)
    counts = counts.reindex(users["user_id"].to_numpy(), fill_value=0)
    totals = counts.sum(axis=1)
    share = counts.div(totals.where(totals > 0), axis=0).fillna(0.0)
    share.columns = [f"{GENRE_AFFINITY_PREFIX}{g}" for g in vocabulary]
    share.index.name = "user_id"
    return share


def build_item_features(train, items, as_of, vocabulary, threshold):
    flagged = train.assign(positive=(train["rating"] >= threshold).astype(int))
    grouped = flagged.groupby("item_id")
    aggregates = pd.DataFrame({
        "item_popularity_count": grouped.size(),
        "item_positive_count": grouped["positive"].sum(),
        "item_avg_rating": grouped["rating"].mean(),
    })
    features = items.set_index("item_id")[["genres", "release_year"]].join(aggregates)
    features["item_primary_genre"] = features["genres"].str.split("|").str[0]
    features["item_n_genres"] = features["genres"].str.count(r"\|") + 1
    features["item_age_years"] = as_of.year - features["release_year"].astype(int)
    features["item_popularity_count"] = features["item_popularity_count"].fillna(0).astype(int)
    features["item_positive_count"] = features["item_positive_count"].fillna(0).astype(int)
    features["item_avg_rating"] = features["item_avg_rating"].fillna(train["rating"].mean())
    genre_lists = features["genres"].str.split("|")
    for genre in vocabulary:
        features[f"{GENRE_FLAG_PREFIX}{genre}"] = genre_lists.map(lambda gs, g=genre: int(g in gs))
    return features[["item_popularity_count", "item_positive_count", "item_avg_rating",
                     "item_primary_genre", "item_n_genres", "item_age_years"]
                    + [f"{GENRE_FLAG_PREFIX}{g}" for g in vocabulary]]


def build_interaction_features(train, threshold):
    timestamps = train["timestamp"]
    return pd.DataFrame({
        "user_id": train["user_id"].to_numpy(),
        "item_id": train["item_id"].to_numpy(),
        "interaction_hour_of_day": timestamps.dt.hour.to_numpy(),
        "interaction_day_of_week": timestamps.dt.dayofweek.to_numpy(),
        "label_is_positive": (train["rating"] >= threshold).astype(int).to_numpy(),
    })


class FeatureStore:
    def __init__(self, root=FEATURES_DIR):
        self.root = Path(root)

    def save(self, users, items, interactions):
        self.root.mkdir(parents=True, exist_ok=True)
        users.to_parquet(self.root / FILES["user"])
        items.to_parquet(self.root / FILES["item"])
        interactions.to_parquet(self.root / FILES["interaction"], index=False)

    def get_user_features(self, user_ids=None):
        features = pd.read_parquet(self.root / FILES["user"])
        return features if user_ids is None else features.reindex(user_ids)

    def get_item_features(self, item_ids=None):
        features = pd.read_parquet(self.root / FILES["item"])
        return features if item_ids is None else features.reindex(item_ids)


def _write_catalog(users, items, interactions, as_of, vocabulary):
    lines = [
        "# Feature Catalog",
        "",
        f"Reference time for time-based features: `{as_of}` (last training timestamp).",
        "",
        "Features are computed from the training split only, to avoid leakage into the test period.",
        "",
        "| Feature | Entity | Type | Source | Description |",
        "|---|---|---|---|---|",
    ]
    for f in FEATURE_CATALOG:
        lines.append(f"| `{f['name']}` | {f['entity']} | {f['dtype']} | {f['source']} | {f['description']} |")
    lines.append(
        f"| `{GENRE_FLAG_PREFIX}<genre>` | item | int | items table | Multi-hot genre flag, one column per genre "
        f"({len(vocabulary)} columns: {', '.join(vocabulary)}). |")
    lines.append(
        f"| `{GENRE_AFFINITY_PREFIX}<genre>` | user | float | train interactions + item genres | Share of the user's "
        "positive training interactions (rating >= threshold) that fall in this genre. Shares sum to 1; 0 for users "
        "with no positive interactions. |")
    lines += [
        "",
        "## Stored feature tables",
        "",
        "| File (data/features/) | Rows | Columns | Key |",
        "|---|---:|---:|---|",
        f"| {FILES['user']} | {len(users)} | {len(users.columns)} | user_id |",
        f"| {FILES['item']} | {len(items)} | {len(items.columns)} | item_id |",
        f"| {FILES['interaction']} | {len(interactions)} | {len(interactions.columns)} | (user_id, item_id) |",
        "",
        f"Cold-start users (no training history): {int((users['user_activity_count'] == 0).sum())}",
        f"Cold-start items (no training interactions): {int((items['item_popularity_count'] == 0).sum())}",
        "",
    ]
    (REPORTS_DIR / "feature_catalog.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg):
    ensure_dirs()
    threshold = cfg["relevance_threshold"]
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    items = pd.read_parquet(PROCESSED_DIR / "items_clean.parquet")
    users = pd.read_parquet(PROCESSED_DIR / "users_clean.parquet")
    vocabulary = genre_vocabulary(items)

    as_of = train["timestamp"].max()
    user_features = build_user_features(train, items, users, as_of, vocabulary, threshold)
    item_features = build_item_features(train, items, as_of, vocabulary, threshold)
    interaction_features = build_interaction_features(train, threshold)

    FeatureStore().save(user_features, item_features, interaction_features)
    _write_catalog(user_features, item_features, interaction_features, as_of, vocabulary)
    log.info("Stored %d user, %d item and %d interaction feature rows",
             len(user_features), len(item_features), len(interaction_features))
