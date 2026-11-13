"""MovieLens 100K source adapter.

Downloads ml-100k.zip from GroupLens (once, kept in data/raw) and converts it into the
three raw CSV files the pipeline reads: users.csv, items.csv and interactions.csv.

MovieLens does not record when a user signed up. signup_date is therefore set to the
date of the user's first rating in the dataset.
"""

import io
import urllib.request
import zipfile

import pandas as pd

from src.common.config import RAW_DIR, get_logger

log = get_logger(__name__)

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_NAME = "ml-100k.zip"
ARCHIVE_DIR = "ml-100k"


def _read_member(archive, name):
    return archive.read(f"{ARCHIVE_DIR}/{name}").decode("latin-1")


def _convert(archive):
    genre_names = [line.split("|")[0] for line in _read_member(archive, "u.genre").splitlines() if line]

    ratings = pd.read_csv(io.StringIO(_read_member(archive, "u.data")), sep="\t",
                          names=["user_id", "item_id", "rating", "unix_ts"])
    interactions = pd.DataFrame({
        "user_id": ratings["user_id"],
        "item_id": ratings["item_id"],
        "rating": ratings["rating"],
        "timestamp": pd.to_datetime(ratings["unix_ts"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })

    first_rating = pd.to_datetime(ratings["unix_ts"], unit="s").groupby(ratings["user_id"]).min()

    users_raw = pd.read_csv(io.StringIO(_read_member(archive, "u.user")), sep="|",
                            names=["user_id", "age", "gender", "occupation", "zip_code"])
    users = pd.DataFrame({
        "user_id": users_raw["user_id"],
        "age": users_raw["age"],
        "gender": users_raw["gender"],
        "occupation": users_raw["occupation"],
        "signup_date": users_raw["user_id"].map(first_rating).dt.strftime("%Y-%m-%d"),
    })

    items_raw = pd.read_csv(io.StringIO(_read_member(archive, "u.item")), sep="|", header=None,
                            encoding="latin-1", index_col=False)
    flags = items_raw.iloc[:, 5:5 + len(genre_names)].to_numpy(dtype=int)
    genres = ["|".join(name for name, on in zip(genre_names, row) if on) or "unknown" for row in flags]
    items = pd.DataFrame({
        "item_id": items_raw[0],
        "title": items_raw[1],
        "genres": genres,
        "release_year": pd.to_datetime(items_raw[2], format="%d-%b-%Y", errors="coerce").dt.year.astype("Int64"),
    })

    return users, items, interactions


def prepare_raw_files():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / ZIP_NAME
    if not zip_path.exists():
        log.info("Downloading MovieLens 100K from %s", URL)
        urllib.request.urlretrieve(URL, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        users, items, interactions = _convert(archive)

    users.to_csv(RAW_DIR / "users.csv", index=False)
    items.to_csv(RAW_DIR / "items.csv", index=False)
    interactions.to_csv(RAW_DIR / "interactions.csv", index=False)
    log.info("Converted MovieLens 100K: %d users, %d items, %d interactions",
             len(users), len(items), len(interactions))
