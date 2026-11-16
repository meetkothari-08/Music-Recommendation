"""Offline evaluation of recommenders.

There is no user-interaction data in this dataset, so we evaluate with an
**artist-continuation** task that mimics playlist continuation:

1. Pick an artist with at least ``min_tracks`` songs in the catalogue.
2. Give the model ``n_seeds`` of their songs as the "playlist so far".
3. Hide the artist's remaining songs and check whether the model retrieves
   them in its top-``k``.

Artist identity is never used as a model feature, so a model can only
succeed by genuinely capturing an artist's sound and style. Retrieving
same-artist songs is a conservative proxy: a recommendation by a different
but similar artist counts as a miss, so absolute numbers understate real
quality. The metrics are best read *relative to the baselines*.

Accuracy alone rewards safe, repetitive lists, so we also report
beyond-accuracy metrics: genre precision, intra-list diversity, catalogue
coverage and popularity bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from music_recommender.features import FeatureBuilder
from music_recommender.recommender import ContentRecommender


class Recommender(Protocol):
    def recommend_indices(self, seed_idx, k: int) -> tuple[np.ndarray, np.ndarray]: ...


# ----------------------------------------------------------------- baselines
class RandomRecommender:
    def __init__(self, tracks: pd.DataFrame, seed: int = 0) -> None:
        self.n = len(tracks)
        self.rng = np.random.default_rng(seed)

    def recommend_indices(self, seed_idx, k):
        pool = np.setdiff1d(
            self.rng.choice(self.n, size=k + len(seed_idx), replace=False), seed_idx
        )
        return pool[:k], np.zeros(k)


class PopularityRecommender:
    """Most popular tracks overall (the classic non-personalised baseline)."""

    def __init__(self, tracks: pd.DataFrame) -> None:
        self.order = np.argsort(-tracks["popularity"].to_numpy(), kind="stable")

    def recommend_indices(self, seed_idx, k):
        head = self.order[: k + len(seed_idx)]
        picked = head[~np.isin(head, seed_idx)][:k]
        return picked, np.zeros(len(picked))


class GenrePopularityRecommender:
    """Most popular tracks within the seeds' genres - a strong simple baseline."""

    def __init__(self, tracks: pd.DataFrame) -> None:
        pop = tracks["popularity"].to_numpy()
        self.pop = pop
        self.genres = tracks["genres"].to_numpy()
        by_genre: dict[str, list[int]] = {}
        for i, gs in enumerate(self.genres):
            for g in gs:
                by_genre.setdefault(g, []).append(i)
        self.by_genre = {
            g: np.array(sorted(ix, key=lambda i: -pop[i])) for g, ix in by_genre.items()
        }

    def recommend_indices(self, seed_idx, k):
        seed_genres = {g for i in seed_idx for g in self.genres[i]}
        cand = np.unique(
            np.concatenate([self.by_genre[g][: k + len(seed_idx)] for g in seed_genres])
        )
        cand = cand[~np.isin(cand, seed_idx)]
        picked = cand[np.argsort(-self.pop[cand], kind="stable")][:k]
        return picked, np.zeros(len(picked))


class ContentModelAdapter:
    """Adapts :class:`ContentRecommender` to the evaluation protocol."""

    def __init__(self, model: ContentRecommender, diversity: float | None = None) -> None:
        self.model = model
        self.diversity = diversity

    def recommend_indices(self, seed_idx, k):
        return self.model.recommend_indices(seed_idx, k, diversity=self.diversity)


# --------------------------------------------------------------------- tasks
@dataclass
class Task:
    seeds: np.ndarray
    targets: np.ndarray


def make_artist_tasks(
    tracks: pd.DataFrame,
    n_tasks: int = 1000,
    n_seeds: int = 3,
    min_tracks: int = 6,
    seed: int = 42,
) -> list[Task]:
    rng = np.random.default_rng(seed)
    groups = tracks.groupby("primary_artist").indices
    eligible = sorted(a for a, ix in groups.items() if len(ix) >= min_tracks)
    chosen = rng.choice(eligible, size=min(n_tasks, len(eligible)), replace=False)
    tasks = []
    for artist in chosen:
        ix = rng.permutation(groups[artist])
        tasks.append(Task(seeds=ix[:n_seeds], targets=ix[n_seeds:]))
    return tasks


# ------------------------------------------------------------------- metrics
def ndcg_at_k(hits: np.ndarray, n_relevant: int) -> float:
    k = len(hits)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((hits * discounts).sum())
    idcg = float(discounts[: min(n_relevant, k)].sum())
    return dcg / idcg if idcg > 0 else 0.0


def intra_list_diversity(vecs: np.ndarray) -> float:
    """Mean pairwise cosine distance between recommended items."""
    n = len(vecs)
    if n < 2:
        return 0.0
    sims = vecs @ vecs.T
    return float(1 - (sims.sum() - np.trace(sims)) / (n * (n - 1)))


class Evaluator:
    def __init__(self, tracks: pd.DataFrame, tasks: list[Task], k: int = 10) -> None:
        self.tracks = tracks.reset_index(drop=True)
        self.tasks = tasks
        self.k = k
        fb = FeatureBuilder().fit(self.tracks)
        self.audio = normalize(fb.audio_matrix(self.tracks))
        self.genres = self.tracks["genres"].map(set).to_numpy()
        self.popularity = self.tracks["popularity"].to_numpy()

    def evaluate(self, model: Recommender) -> dict[str, float]:
        rows, seen = [], set()
        for task in self.tasks:
            recs, _ = model.recommend_indices(task.seeds, self.k)
            hits = np.isin(recs, task.targets).astype(float)
            seed_genres = set().union(*self.genres[task.seeds])
            rows.append(
                {
                    "hit_rate": float(hits.any()),
                    "recall": hits.sum() / min(len(task.targets), self.k),
                    "ndcg": ndcg_at_k(hits, len(task.targets)),
                    "genre_precision": np.mean([bool(self.genres[i] & seed_genres) for i in recs]),
                    "diversity": intra_list_diversity(self.audio[recs]),
                    "avg_popularity": self.popularity[recs].mean(),
                }
            )
            seen.update(recs.tolist())
        df = pd.DataFrame(rows)
        n = len(df)
        out = df.mean().to_dict()
        out["ndcg_ci95"] = 1.96 * df["ndcg"].std(ddof=1) / np.sqrt(n)
        out["hit_rate_ci95"] = 1.96 * df["hit_rate"].std(ddof=1) / np.sqrt(n)
        out["coverage"] = len(seen) / len(self.tracks)
        return out

    def compare(self, models: dict[str, Recommender]) -> pd.DataFrame:
        return pd.DataFrame({name: self.evaluate(m) for name, m in models.items()}).T


def default_model_suite(tracks: pd.DataFrame) -> dict[str, Recommender]:
    """Baselines plus the content model's ablations."""

    def content(**kw) -> ContentRecommender:
        return ContentRecommender(**kw).fit(tracks)

    final = content()  # tuned defaults
    return {
        "Random": RandomRecommender(tracks),
        "Popularity": PopularityRecommender(tracks),
        "Genre popularity": GenrePopularityRecommender(tracks),
        "Audio only (kNN)": ContentModelAdapter(content(genre_weight=0.0), diversity=0.0),
        "Audio + genre embedding": ContentModelAdapter(
            content(genre_mode="embedding"), diversity=0.0
        ),
        "Audio + genre one-hot": ContentModelAdapter(final, diversity=0.0),
        "Audio + genre one-hot + MMR (final)": ContentModelAdapter(final),
    }


def weight_sweep(
    evaluator: Evaluator, tracks: pd.DataFrame, weights=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0)
) -> pd.DataFrame:
    rows = []
    for mode in ("onehot", "embedding"):
        for w in weights:
            model = ContentRecommender(genre_weight=w, genre_mode=mode).fit(tracks)
            metrics = evaluator.evaluate(ContentModelAdapter(model, diversity=0.0))
            rows.append({"genre_mode": mode, "genre_weight": w, **metrics})
    return pd.DataFrame(rows)


def diversity_sweep(
    evaluator: Evaluator, model: ContentRecommender, lambdas=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
) -> pd.DataFrame:
    rows = [
        {"diversity_lambda": lam, **evaluator.evaluate(ContentModelAdapter(model, lam))}
        for lam in lambdas
    ]
    return pd.DataFrame(rows)
