import numpy as np
import pandas as pd

from src.features.feature_store import GENRE_AFFINITY_PREFIX, GENRE_FLAG_PREFIX


class PopularityRecommender:
    """Baseline: recommends the most-liked items in training, ignoring the user."""

    name = "popularity_baseline"

    def __init__(self):
        self.ranked_items = np.array([], dtype=int)
        self.avg_rating = pd.Series(dtype=float)
        self.global_mean = 3.0

    def params(self):
        return {"model": self.name, "ranking": "positive_count with avg_rating tie-break"}

    def fit(self, item_features, global_mean):
        score = item_features["item_positive_count"] + item_features["item_avg_rating"] / 1000
        self.ranked_items = score.sort_values(ascending=False).index.to_numpy()
        self.avg_rating = item_features["item_avg_rating"]
        self.global_mean = float(global_mean)
        return self

    def predict(self, user_id, item_id):
        return float(self.avg_rating.get(item_id, self.global_mean))

    def recommend(self, user_id, k, exclude=()):
        excluded = {int(x) for x in exclude}
        recs = []
        for item in self.ranked_items:
            if int(item) in excluded:
                continue
            recs.append(int(item))
            if len(recs) == k:
                break
        return recs


class GenreContentRecommender:
    """Hybrid content + popularity recommender. The score is the cosine similarity
    between the user's genre affinity and the item's genre flags, plus
    popularity_weight times the item's normalised positive count. Both genre inputs
    come from the feature store. Ratings are predicted as the user's average training rating."""

    name = "genre_content_recommender"

    def __init__(self, fallback, popularity_weight=1.0):
        self.fallback = fallback
        self.popularity_weight = popularity_weight
        self.global_mean = 3.0

    def params(self):
        return {"model": self.name,
                "score": "cosine(user genre affinity, item genre flags) + popularity_weight * normalised item_positive_count",
                "popularity_weight": self.popularity_weight, "fallback": self.fallback.name}

    def fit(self, user_features, item_features, global_mean):
        flag_cols = [c for c in item_features.columns if c.startswith(GENRE_FLAG_PREFIX)]
        affinity_cols = [c for c in user_features.columns if c.startswith(GENRE_AFFINITY_PREFIX)]

        flags = item_features[flag_cols].to_numpy(dtype=float)
        norms = np.sqrt(flags.sum(axis=1))
        norms[norms == 0] = 1.0
        self.item_ids = item_features.index.to_numpy(dtype=int)
        self.item_index = {int(i): n for n, i in enumerate(self.item_ids)}
        self.item_vectors = flags / norms[:, None]

        self.user_index = {int(u): n for n, u in enumerate(user_features.index.to_numpy(dtype=int))}
        self.user_vectors = user_features[affinity_cols].to_numpy(dtype=float)
        self.user_avg_rating = user_features["user_avg_rating"].to_dict()

        positives = item_features["item_positive_count"].to_numpy(dtype=float)
        self.popularity_term = self.popularity_weight * positives / max(positives.max(), 1.0)
        self.global_mean = float(global_mean)
        return self

    def predict(self, user_id, item_id):
        return float(self.user_avg_rating.get(user_id, self.global_mean))

    def recommend(self, user_id, k, exclude=()):
        idx = self.user_index.get(user_id)
        if idx is None or not self.user_vectors[idx].any():
            return self.fallback.recommend(user_id, k, exclude)
        scores = self.item_vectors @ self.user_vectors[idx] + self.popularity_term
        excluded = [self.item_index[int(x)] for x in exclude if int(x) in self.item_index]
        scores[excluded] = -np.inf
        top = np.argsort(-scores, kind="stable")[:k]
        return [int(self.item_ids[j]) for j in top if np.isfinite(scores[j])]


class BiasedMatrixFactorization:
    """Explicit-rating matrix factorization trained with SGD:
    r_hat = mu + b_u + b_i + p_u . q_i
    """

    name = "biased_matrix_factorization"

    def __init__(self, n_factors=32, epochs=20, learning_rate=0.01, regularization=0.05,
                 seed=42, fallback=None):
        self.n_factors = n_factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.seed = seed
        self.fallback = fallback
        self.history = []

    def params(self):
        return {
            "model": self.name,
            "n_factors": self.n_factors,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "seed": self.seed,
        }

    def fit(self, train):
        self.user_ids = np.sort(train["user_id"].unique())
        self.item_ids = np.sort(train["item_id"].unique())
        self.user_index = {int(u): n for n, u in enumerate(self.user_ids)}
        self.item_index = {int(i): n for n, i in enumerate(self.item_ids)}

        u = train["user_id"].map(self.user_index).to_numpy(dtype=int)
        i = train["item_id"].map(self.item_index).to_numpy(dtype=int)
        r = train["rating"].to_numpy(dtype=float)

        rng = np.random.default_rng(self.seed)
        self.mu = float(r.mean())
        self.bu = np.zeros(len(self.user_ids))
        self.bi = np.zeros(len(self.item_ids))
        self.P = rng.normal(0, 0.1, (len(self.user_ids), self.n_factors))
        self.Q = rng.normal(0, 0.1, (len(self.item_ids), self.n_factors))

        lr, reg = self.learning_rate, self.regularization
        ul, il, rl = u.tolist(), i.tolist(), r.tolist()
        for epoch in range(1, self.epochs + 1):
            for n in rng.permutation(len(rl)).tolist():
                uu, ii, rr = ul[n], il[n], rl[n]
                pu = self.P[uu].copy()
                qi = self.Q[ii].copy()
                err = rr - (self.mu + self.bu[uu] + self.bi[ii] + pu @ qi)
                self.bu[uu] += lr * (err - reg * self.bu[uu])
                self.bi[ii] += lr * (err - reg * self.bi[ii])
                self.P[uu] += lr * (err * qi - reg * pu)
                self.Q[ii] += lr * (err * pu - reg * qi)
            rmse = float(np.sqrt(np.mean((r - self._score_arrays(u, i)) ** 2)))
            self.history.append({"epoch": epoch, "train_rmse": rmse})
        return self

    def _score_arrays(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + np.sum(self.P[u] * self.Q[i], axis=1)

    def predict(self, user_id, item_id):
        if user_id not in self.user_index or item_id not in self.item_index:
            if self.fallback is not None:
                return self.fallback.predict(user_id, item_id)
            return self.mu
        uu, ii = self.user_index[user_id], self.item_index[item_id]
        return float(np.clip(self._score_arrays(np.array([uu]), np.array([ii]))[0], 1, 5))

    def recommend(self, user_id, k, exclude=()):
        if user_id not in self.user_index:
            return self.fallback.recommend(user_id, k, exclude)
        uu = self.user_index[user_id]
        scores = self.mu + self.bu[uu] + self.bi + self.Q @ self.P[uu]
        excluded = [self.item_index[int(x)] for x in exclude if int(x) in self.item_index]
        scores[excluded] = -np.inf
        top = np.argsort(-scores)[:k]
        return [int(self.item_ids[j]) for j in top if np.isfinite(scores[j])]
