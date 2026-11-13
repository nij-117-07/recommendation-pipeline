import math
from itertools import combinations

import numpy as np


def precision_at_k(recs, relevant, k):
    hits = sum(1 for item in recs[:k] if item in relevant)
    return hits / k


def recall_at_k(recs, relevant, k):
    if not relevant:
        return 0.0
    hits = sum(1 for item in recs[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recs, relevant, k):
    dcg = sum(1 / math.log2(rank + 2) for rank, item in enumerate(recs[:k]) if item in relevant)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def catalog_coverage(all_recommended_items, catalog_size):
    return len(set(all_recommended_items)) / catalog_size


def intra_list_diversity(recs, item_genres):
    """Average pairwise genre distance (1 - Jaccard) within one recommendation list."""
    if len(recs) < 2:
        return 0.0
    distances = []
    for a, b in combinations(recs, 2):
        genres_a = item_genres.get(a, frozenset())
        genres_b = item_genres.get(b, frozenset())
        union = genres_a | genres_b
        similarity = len(genres_a & genres_b) / len(union) if union else 0.0
        distances.append(1 - similarity)
    return float(np.mean(distances))


def rmse(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))
