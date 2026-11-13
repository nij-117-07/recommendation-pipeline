"""Stage 2: data validation.

Every rule is either an *error* (the row is moved to data/errors) or a *warning*
(logged only, handled later in preparation). Results are written to
reports/validation_report.md and reports/validation_summary.json.
"""

import json

import pandas as pd

from src.common.config import ERRORS_DIR, PROCESSED_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs, get_logger

log = get_logger(__name__)


class RuleSet:
    def __init__(self, table, df):
        self.table = table
        self.df = df
        self.reasons = pd.Series("", index=df.index, dtype=object)
        self.results = []

    def _as_mask(self, mask):
        return pd.Series(mask, index=self.df.index).fillna(False).astype(bool)

    def rule(self, name, mask):
        mask = self._as_mask(mask)
        self.reasons = self.reasons.mask(mask, self.reasons + name + ";")
        self.results.append({"table": self.table, "check": name, "severity": "error",
                             "failed_rows": int(mask.sum())})

    def warn(self, name, mask):
        mask = self._as_mask(mask)
        self.results.append({"table": self.table, "check": name, "severity": "warning",
                             "failed_rows": int(mask.sum())})

    def split(self):
        bad = self.reasons != ""
        clean = self.df.loc[~bad].copy()
        errors = self.df.loc[bad].copy()
        errors["error_reasons"] = self.reasons[bad].str.rstrip(";")
        return clean, errors, self.results


def validate_users(raw):
    df = raw.copy()
    uid = pd.to_numeric(df["user_id"], errors="coerce")
    age = pd.to_numeric(df["age"], errors="coerce")

    r = RuleSet("users", df)
    r.rule("missing_user_id", df["user_id"].isna())
    r.rule("invalid_user_id_type", df["user_id"].notna() & uid.isna())
    r.rule("duplicate_user_id", uid.notna() & uid.duplicated(keep="first"))
    r.rule("invalid_age_type", df["age"].notna() & age.isna())
    r.rule("invalid_age_range", age.notna() & ~age.between(1, 120))
    r.warn("missing_age", df["age"].isna())

    clean, errors, results = r.split()
    clean = clean.assign(user_id=uid.loc[clean.index].astype(int), age=age.loc[clean.index])
    return clean, errors, results


def validate_items(raw):
    df = raw.copy()
    iid = pd.to_numeric(df["item_id"], errors="coerce")
    year = pd.to_numeric(df["release_year"], errors="coerce")

    r = RuleSet("items", df)
    r.rule("missing_item_id", df["item_id"].isna())
    r.rule("invalid_item_id_type", df["item_id"].notna() & iid.isna())
    r.rule("duplicate_item_id", iid.notna() & iid.duplicated(keep="first"))
    r.rule("invalid_release_year_type", df["release_year"].notna() & year.isna())
    r.rule("invalid_release_year_range", year.notna() & ~year.between(1900, 2030))
    r.warn("missing_genres", df["genres"].isna())

    clean, errors, results = r.split()
    clean = clean.assign(item_id=iid.loc[clean.index].astype(int), release_year=year.loc[clean.index])
    return clean, errors, results


def validate_interactions(raw, users, items, cfg):
    df = raw.copy()
    uid = pd.to_numeric(df["user_id"], errors="coerce")
    iid = pd.to_numeric(df["item_id"], errors="coerce")
    rating = pd.to_numeric(df["rating"], errors="coerce")
    ts = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce", utc=True)
    low, high = cfg["validation"]["min_rating"], cfg["validation"]["max_rating"]
    now = pd.Timestamp(cfg["reference_date"])

    r = RuleSet("interactions", df)
    r.rule("missing_user_id", df["user_id"].isna())
    r.rule("missing_item_id", df["item_id"].isna())
    r.rule("missing_rating", df["rating"].isna())
    r.rule("missing_timestamp", df["timestamp"].isna())
    r.rule("invalid_user_id_type", df["user_id"].notna() & uid.isna())
    r.rule("invalid_item_id_type", df["item_id"].notna() & iid.isna())
    r.rule("invalid_rating_type", df["rating"].notna() & rating.isna())
    r.rule("invalid_rating_value",
           rating.notna() & (~rating.between(low, high) | (rating % 1 != 0)))
    r.rule("invalid_timestamp", df["timestamp"].notna() & ts.isna())
    r.rule("future_timestamp", ts.notna() & (ts > now))
    r.rule("duplicate_record",
           df.duplicated(subset=["user_id", "item_id", "rating", "timestamp"], keep="first"))
    r.rule("orphan_user_id", uid.notna() & ~uid.isin(users["user_id"]))
    r.rule("orphan_item_id", iid.notna() & ~iid.isin(items["item_id"]))
    r.warn("repeated_user_item_pair", df.duplicated(subset=["user_id", "item_id"], keep="first"))

    clean, errors, results = r.split()
    clean = clean.assign(
        user_id=uid.loc[clean.index].astype(int),
        item_id=iid.loc[clean.index].astype(int),
        rating=rating.loc[clean.index].astype(int),
        timestamp=ts.loc[clean.index],
    )
    return clean, errors, results


def _write_report(summary):
    lines = [
        "# Data Validation Report",
        "",
        "## Row counts",
        "",
        "| Table | Raw rows | Valid rows | Rejected rows |",
        "|---|---:|---:|---:|",
    ]
    for table, counts in summary["tables"].items():
        lines.append(f"| {table} | {counts['raw_rows']} | {counts['valid_rows']} | {counts['rejected_rows']} |")
    lines += [
        "",
        "## Rule results",
        "",
        "| Table | Check | Severity | Rows flagged |",
        "|---|---|---|---:|",
    ]
    for check in summary["checks"]:
        lines.append(f"| {check['table']} | {check['check']} | {check['severity']} | {check['failed_rows']} |")
    lines += [
        "",
        "Rejected rows are written to `data/errors/<table>_errors.csv` with an `error_reasons` column.",
        "Warnings do not reject rows; they are handled during preparation (imputation or de-duplication).",
        "",
    ]
    (REPORTS_DIR / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(cfg):
    ensure_dirs()
    users_raw = pd.read_csv(RAW_DIR / "users.csv", dtype=str)
    items_raw = pd.read_csv(RAW_DIR / "items.csv", dtype=str)
    inter_raw = pd.read_csv(RAW_DIR / "interactions.csv", dtype=str)

    users, users_err, users_checks = validate_users(users_raw)
    items, items_err, items_checks = validate_items(items_raw)
    inter, inter_err, inter_checks = validate_interactions(inter_raw, users, items, cfg)

    datasets = {
        "users": (users_raw, users, users_err),
        "items": (items_raw, items, items_err),
        "interactions": (inter_raw, inter, inter_err),
    }
    summary = {"tables": {}, "checks": users_checks + items_checks + inter_checks}
    for name, (raw, valid, errors) in datasets.items():
        valid.to_parquet(PROCESSED_DIR / f"{name}_valid.parquet", index=False)
        errors.to_csv(ERRORS_DIR / f"{name}_errors.csv", index=False)
        summary["tables"][name] = {
            "raw_rows": len(raw),
            "valid_rows": len(valid),
            "rejected_rows": len(errors),
        }
        log.info("%s: %d raw, %d valid, %d rejected", name, len(raw), len(valid), len(errors))

    (REPORTS_DIR / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(summary)
