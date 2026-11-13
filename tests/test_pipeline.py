import hashlib

import numpy as np
import pandas as pd
import pytest

from dags.recommendation_pipeline import build_dag
from src.common.config import load_config
from src.evaluation.metrics import intra_list_diversity, ndcg_at_k, precision_at_k, recall_at_k
from src.features.feature_store import build_user_genre_affinity
from src.ingestion import ingest
from src.orchestration.stages import STAGES
from src.orchestration.workflow import Task, topological_order
from src.training.models import BiasedMatrixFactorization, GenreContentRecommender, PopularityRecommender
from src.validation.validate import RuleSet, validate_interactions


def test_ranking_metrics_on_known_example():
    recs = [1, 2, 3, 4, 5]
    relevant = {2, 5, 9}
    assert precision_at_k(recs, relevant, 5) == pytest.approx(2 / 5)
    assert recall_at_k(recs, relevant, 5) == pytest.approx(2 / 3)
    expected = (1 / np.log2(3) + 1 / np.log2(6)) / (1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4))
    assert ndcg_at_k(recs, relevant, 5) == pytest.approx(expected)


def test_diversity_is_zero_for_identical_genres():
    genres = {1: frozenset({"Drama"}), 2: frozenset({"Drama"})}
    assert intra_list_diversity([1, 2], genres) == 0.0


def test_validation_flags_invalid_rows_and_orphans():
    users = pd.DataFrame({"user_id": [1, 2]})
    items = pd.DataFrame({"item_id": [10]})
    raw = pd.DataFrame({
        "user_id": ["1", "1", "2", "99", "2"],
        "item_id": ["10", "10", "10", "10", "11"],
        "rating": ["4", "9", "3", "5", "2"],
        "timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z",
                      "2024-01-03T00:00:00Z", "2024-01-04T00:00:00Z", "not-a-date"],
    })
    cfg = {"validation": {"min_rating": 1, "max_rating": 5}, "reference_date": "2026-01-01T00:00:00Z"}
    clean, errors, _ = validate_interactions(raw, users, items, cfg)

    assert list(clean.index) == [0, 2]
    reasons = dict(zip(errors.index, errors["error_reasons"]))
    assert "invalid_rating_value" in reasons[1]
    assert "orphan_user_id" in reasons[3]
    assert "orphan_item_id" in reasons[4]
    assert "invalid_timestamp" in reasons[4]


def test_future_timestamps_use_configured_reference_date():
    users = pd.DataFrame({"user_id": [1]})
    items = pd.DataFrame({"item_id": [10]})
    raw = pd.DataFrame({"user_id": ["1"], "item_id": ["10"], "rating": ["4"],
                        "timestamp": ["2026-06-01T00:00:00Z"]})
    cfg = {"validation": {"min_rating": 1, "max_rating": 5}, "reference_date": "2026-01-01T00:00:00Z"}
    clean, errors, _ = validate_interactions(raw, users, items, cfg)
    assert len(clean) == 0
    assert "future_timestamp" in errors.iloc[0]["error_reasons"]


def test_ruleset_warnings_do_not_reject_rows():
    df = pd.DataFrame({"x": [1, None]})
    rules = RuleSet("demo", df)
    rules.warn("missing_x", df["x"].isna())
    clean, errors, results = rules.split()
    assert len(clean) == 2 and len(errors) == 0
    assert results[0]["severity"] == "warning"


def test_popularity_never_recommends_seen_items():
    item_features = pd.DataFrame({
        "item_positive_count": [5, 3, 1],
        "item_avg_rating": [4.0, 4.0, 4.0],
    }, index=[100, 200, 300])
    model = PopularityRecommender().fit(item_features, global_mean=3.5)
    assert model.recommend(1, 2, exclude={100}) == [200, 300]


def test_genre_content_ranks_by_genre_match_and_excludes_seen():
    item_features = pd.DataFrame({
        "item_positive_count": [1, 1, 1],
        "item_avg_rating": [3.0, 3.0, 3.0],
        "item_genre_Action": [1, 0, 1],
        "item_genre_Comedy": [0, 1, 1],
    }, index=[10, 20, 30])
    user_features = pd.DataFrame({
        "user_avg_rating": [4.0, 2.0],
        "user_genre_affinity_Action": [1.0, 0.0],
        "user_genre_affinity_Comedy": [0.0, 0.0],
    }, index=[1, 2])
    popularity = PopularityRecommender().fit(item_features, 3.0)
    model = GenreContentRecommender(fallback=popularity).fit(user_features, item_features, 3.0)

    assert model.recommend(1, 2) == [10, 30]
    assert model.recommend(1, 2, exclude={10}) == [30, 20]
    assert model.recommend(2, 1) == popularity.recommend(2, 1)
    assert model.predict(1, 10) == 4.0


def test_user_genre_affinity_is_share_of_positive_genres():
    train = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [10, 20, 10], "rating": [5, 2, 1]})
    items = pd.DataFrame({"item_id": [10, 20], "genres": ["Action|Comedy", "Drama"]})
    users = pd.DataFrame({"user_id": [1, 2, 3]})
    share = build_user_genre_affinity(train, items, users, ["Action", "Comedy", "Drama"], threshold=4)

    assert share.loc[1].tolist() == [0.5, 0.5, 0.0]
    assert share.loc[2].sum() == 0
    assert share.loc[3].sum() == 0


def test_matrix_factorization_recommends_unseen_items_only():
    train = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3],
        "item_id": [10, 20, 10, 30, 20, 30],
        "rating": [5, 1, 4, 2, 1, 5],
    })
    model = BiasedMatrixFactorization(n_factors=4, epochs=5, seed=0,
                                      fallback=PopularityRecommender().fit(
                                          pd.DataFrame({"item_positive_count": [1, 1, 1],
                                                        "item_avg_rating": [3.0, 3.0, 3.0]},
                                                       index=[10, 20, 30]), 3.0)).fit(train)
    recs = model.recommend(1, 2, exclude={10, 20})
    assert recs == [30]
    assert model.recommend(999, 1) == [model.fallback.ranked_items[0]]


def test_matrix_factorization_is_reproducible_with_same_seed():
    train = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3],
        "item_id": [10, 20, 10, 30, 20, 30],
        "rating": [5, 1, 4, 2, 1, 5],
    })
    first = BiasedMatrixFactorization(n_factors=4, epochs=3, seed=3).fit(train)
    second = BiasedMatrixFactorization(n_factors=4, epochs=3, seed=3).fit(train)
    assert np.array_equal(first.P, second.P)
    assert np.array_equal(first.Q, second.Q)


def test_synthetic_generation_is_reproducible(tmp_path, monkeypatch):
    cfg = {"random_seed": 7,
           "synthetic": {"n_users": 50, "n_items": 40, "n_interactions": 500, "error_rate": 0.02}}
    digests = []
    for name in ("first", "second"):
        target = tmp_path / name
        target.mkdir()
        monkeypatch.setattr(ingest, "RAW_DIR", target)
        ingest.generate_synthetic_dataset(cfg)
        digests.append({f: hashlib.sha256((target / f).read_bytes()).hexdigest() for f in ingest.RAW_FILES})
    assert digests[0] == digests[1]


def test_topological_order_respects_dependencies():
    noop = lambda cfg: None
    tasks = {
        "c": Task("c", noop, deps=("b",)),
        "b": Task("b", noop, deps=("a",)),
        "a": Task("a", noop),
    }
    assert topological_order(tasks) == ["a", "b", "c"]


def test_topological_order_detects_cycles():
    noop = lambda cfg: None
    tasks = {"a": Task("a", noop, deps=("b",)), "b": Task("b", noop, deps=("a",))}
    with pytest.raises(ValueError):
        topological_order(tasks)


def test_stages_reference_only_earlier_stages():
    names = [stage.name for stage in STAGES]
    assert len(names) == len(set(names))
    assert names[0] == "record_run_metadata"
    for position, stage in enumerate(STAGES):
        assert set(stage.deps) <= set(names[:position]), stage.name


def test_project_dag_is_valid():
    dag = build_dag()
    order = topological_order(dag)
    assert order[0] == "record_run_metadata"
    assert order[-1] == "evaluate_models"
    assert {name: set(task.deps) for name, task in dag.items()} == {
        stage.name: set(stage.deps) for stage in STAGES}
    assert load_config()["top_k"] == 10
