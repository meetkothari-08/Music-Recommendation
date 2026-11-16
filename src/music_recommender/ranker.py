"""Two-stage retrieve-then-rank recommender.

Stage 1 (**retrieval**) is the tuned :class:`ContentRecommender` with MMR
switched off: it returns the ``n_candidates`` nearest tracks to the seeds'
taste profile.

Stage 2 (**ranking**) is a LightGBM ``LGBMRanker`` trained with the
lambdarank objective on pairwise (query, candidate) features. It is trained
on artist-continuation tasks: a candidate is labelled relevant when it is by
the seed artist. The training artists must be disjoint from the validation
and test artists, which :func:`train_ranker` enforces.

Artist and album fields are **never** used as features - they would leak
the label. :data:`FEATURE_NAMES` lists everything the ranker sees, and
:func:`candidate_features` only reads audio features, genres, popularity and
the retrieval output.

Optionally, MMR is applied after the ranker (on per-query min-max scaled
scores) so that diversity can be traded against accuracy as in stage 1.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from music_recommender.config import AUDIO_FEATURES
from music_recommender.recommender import ContentRecommender, mmr_rerank

FEATURE_NAMES: list[str] = [
    "cosine_sim",  # overall similarity to the seed profile (stage-1 score)
    "audio_sim",  # cosine similarity of the audio block only
    "max_seed_sim",  # similarity to the closest individual seed
    "genre_jaccard",  # |genres(c) ∩ genres(seeds)| / |genres(c) ∪ genres(seeds)|
    *[f"absdiff_{f}" for f in AUDIO_FEATURES],  # quantile scale, vs. seed mean
    "tempo_diff_bpm",
    "popularity",
    "popularity_gap",  # candidate popularity - mean seed popularity
    "retrieval_rank",  # 0 = stage-1 nearest neighbour
]

# Column-name fragments that must never appear among the ranker's features.
LEAKY_FIELDS = ("artist", "album", "track_name", "track_id")


def _require_lightgbm():
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The learned ranker needs LightGBM: pip install 'music-recommender[ranker]'"
        ) from exc
    return lightgbm


class LearnedRanker:
    """Retrieve candidates with a :class:`ContentRecommender`, re-rank with LightGBM.

    Parameters
    ----------
    retriever:
        A fitted content model used for stage-1 retrieval (always with
        ``diversity=0``).
    n_candidates:
        Candidates retrieved per query.
    diversity:
        MMR λ applied to the ranker's scores (0 = pure ranker order).
    lgbm_params:
        Overrides for :class:`lightgbm.LGBMRanker`.
    features:
        Subset of :data:`FEATURE_NAMES` to train on (default: all). Used for
        feature-group ablations.
    """

    default_params: dict = {
        "objective": "lambdarank",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 0,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }

    def __init__(
        self,
        retriever: ContentRecommender,
        n_candidates: int = 200,
        diversity: float = 0.0,
        lgbm_params: dict | None = None,
        features: Sequence[str] | None = None,
    ) -> None:
        self.feature_names = list(features or FEATURE_NAMES)
        unknown = set(self.feature_names) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"Unknown ranker features: {sorted(unknown)}")
        self.retriever = retriever
        self.n_candidates = n_candidates
        self.diversity = diversity
        self.lgbm_params = {**self.default_params, **(lgbm_params or {})}
        tracks = retriever.tracks_
        fb = retriever.features
        self._vectors = retriever.vectors_
        self._audio = fb.audio_matrix(tracks)
        self._audio_unit = normalize(self._audio)
        self._genres = fb.mlb_.transform(tracks["genres"]).astype(bool)
        self._tempo = tracks["tempo"].to_numpy(dtype=float)
        self._popularity = tracks["popularity"].to_numpy(dtype=float)

    # ----------------------------------------------------------- features
    def retrieve(self, seed_idx: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Stage 1: top ``n_candidates`` by cosine similarity (no MMR)."""
        return self.retriever.recommend_indices(seed_idx, k=self.n_candidates, diversity=0.0)

    def candidate_features(
        self, seed_idx: Sequence[int], cand: np.ndarray, sims: np.ndarray
    ) -> pd.DataFrame:
        """Feature matrix for ``cand`` (ordered by stage-1 rank) given the seeds."""
        seed_idx = np.asarray(seed_idx)
        audio_profile = self._audio[seed_idx].mean(axis=0)
        audio_profile_unit = audio_profile / (np.linalg.norm(audio_profile) + 1e-12)
        seed_genres = self._genres[seed_idx].any(axis=0)
        cand_genres = self._genres[cand]
        inter = (cand_genres & seed_genres).sum(axis=1)
        union = (cand_genres | seed_genres).sum(axis=1)

        cols = {
            "cosine_sim": sims,
            "audio_sim": self._audio_unit[cand] @ audio_profile_unit,
            "max_seed_sim": (self._vectors[cand] @ self._vectors[seed_idx].T).max(axis=1),
            "genre_jaccard": inter / np.maximum(union, 1),
        }
        diffs = np.abs(self._audio[cand] - audio_profile)
        for j, f in enumerate(AUDIO_FEATURES):
            cols[f"absdiff_{f}"] = diffs[:, j]
        cols["tempo_diff_bpm"] = np.abs(self._tempo[cand] - self._tempo[seed_idx].mean())
        cols["popularity"] = self._popularity[cand]
        cols["popularity_gap"] = self._popularity[cand] - self._popularity[seed_idx].mean()
        cols["retrieval_rank"] = np.arange(len(cand), dtype=float)
        return pd.DataFrame(cols, columns=FEATURE_NAMES)[self.feature_names].astype(np.float32)

    def build_training_set(
        self, tasks: Iterable
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
        """Features, labels and group sizes for a list of tasks.

        Queries whose candidates contain no positive are dropped (they carry
        no gradient under lambdarank). Also returns stage-1 recall@n, i.e.
        the share of queries with at least one positive among the candidates.
        """
        frames, labels, groups = [], [], []
        n_tasks = n_with_pos = 0
        for task in tasks:
            n_tasks += 1
            cand, sims = self.retrieve(task.seeds)
            y = np.isin(cand, task.targets).astype(int)
            if not y.any():
                continue
            n_with_pos += 1
            frames.append(self.candidate_features(task.seeds, cand, sims))
            labels.append(y)
            groups.append(len(cand))
        if not frames:
            raise ValueError("No training query has a positive candidate")
        X = pd.concat(frames, ignore_index=True)
        return X, np.concatenate(labels), np.array(groups), n_with_pos / max(n_tasks, 1)

    # ---------------------------------------------------------------- fit
    def fit(self, tasks: Sequence) -> LearnedRanker:
        lgb = _require_lightgbm()
        X, y, groups, self.train_candidate_recall_ = self.build_training_set(tasks)
        self.n_train_queries_ = len(groups)
        self.model_ = lgb.LGBMRanker(**self.lgbm_params)
        self.model_.fit(X, y, group=groups)
        return self

    def feature_importance(self, kind: str = "gain") -> pd.Series:
        """Feature importance (``"gain"`` or ``"split"``), largest first."""
        imp = self.model_.booster_.feature_importance(importance_type=kind)
        return pd.Series(imp, index=self.feature_names, name=kind).sort_values(ascending=False)

    # ---------------------------------------------------------- inference
    def score(self, seed_idx: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Stage 1 + stage 2: candidates and their ranker scores (unsorted)."""
        cand, sims = self.retrieve(seed_idx)
        if len(cand) == 0:
            return cand, np.array([])
        X = self.candidate_features(seed_idx, cand, sims)
        return cand, self.model_.predict(X)

    def rerank(
        self, cand: np.ndarray, scores: np.ndarray, k: int, diversity: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Order scored candidates, optionally with MMR on min-max scaled scores."""
        span = scores.max() - scores.min() if len(scores) else 0.0
        relevance = (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)
        order = mmr_rerank(self._vectors[cand], relevance, k, diversity)
        return cand[order], scores[order]

    def recommend_indices(
        self, seed_idx: Sequence[int], k: int = 10, diversity: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        diversity = self.diversity if diversity is None else diversity
        cand, scores = self.score(seed_idx)
        return self.rerank(cand, scores, k, diversity)


def train_ranker(
    tracks: pd.DataFrame,
    exclude_artists: Iterable[str],
    retriever: ContentRecommender | None = None,
    n_tasks: int | None = None,
    repeats: int = 2,
    seed: int = 123,
    **kwargs,
) -> LearnedRanker:
    """Train a :class:`LearnedRanker` on artists disjoint from ``exclude_artists``.

    ``repeats`` independent seed/target splits are drawn per training artist
    to get more queries out of the same artists.
    """
    from music_recommender.evaluation import make_artist_tasks

    retriever = retriever or ContentRecommender(diversity=0.0).fit(tracks)
    exclude = set(exclude_artists)
    tasks = []
    for r in range(repeats):
        tasks += make_artist_tasks(
            tracks, n_tasks=n_tasks or len(tracks), seed=seed + r, exclude_artists=exclude
        )
    ranker = LearnedRanker(retriever, **kwargs).fit(tasks)
    ranker.train_artists_ = set(tracks["primary_artist"].iloc[[t.seeds[0] for t in tasks]])
    assert not ranker.train_artists_ & exclude, "training artists overlap evaluation artists"
    return ranker
