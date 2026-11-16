"""Content-based recommender with diversity-aware re-ranking."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from music_recommender.config import PROFILE_FEATURES
from music_recommender.features import FeatureBuilder, GenreMode


def mmr_rerank(
    candidate_vecs: np.ndarray,
    relevance: np.ndarray,
    k: int,
    diversity: float,
) -> np.ndarray:
    """Maximal Marginal Relevance re-ranking.

    Greedily picks items maximising
    ``(1 - diversity) * relevance - diversity * max_sim_to_already_picked``.
    ``diversity=0`` reproduces the plain relevance ranking.
    Returns positions into the candidate arrays.
    """
    k = min(k, len(relevance))
    if diversity <= 0 or k == 0:
        return np.argsort(-relevance)[:k]
    selected: list[int] = []
    max_sim = np.full(len(relevance), -np.inf)
    available = np.ones(len(relevance), dtype=bool)
    for _ in range(k):
        redundancy = np.where(np.isfinite(max_sim), max_sim, 0.0)
        score = (1 - diversity) * relevance - diversity * redundancy
        score[~available] = -np.inf
        best = int(np.argmax(score))
        selected.append(best)
        available[best] = False
        max_sim = np.maximum(max_sim, candidate_vecs @ candidate_vecs[best])
    return np.array(selected)


class ContentRecommender:
    """Recommend tracks similar to one or more seed tracks.

    Multiple seeds are combined into a single "taste profile" (the mean of
    their vectors), so the model doubles as a playlist-continuation engine.

    Defaults (one-hot genres, ``genre_weight=1.5``, ``diversity=0.3``) were
    chosen on a validation task set; see ``reports/evaluation.md``.

    Example
    -------
    >>> rec = ContentRecommender().fit(tracks)
    >>> rec.recommend(["4uLU6hMCjMI75M1A2tKUQC"], k=10, diversity=0.3)
    """

    def __init__(
        self,
        audio_weight: float = 1.0,
        genre_weight: float = 1.5,
        genre_mode: GenreMode = "onehot",
        diversity: float = 0.3,
        n_candidates: int = 200,
    ) -> None:
        self.features = FeatureBuilder(
            audio_weight=audio_weight, genre_weight=genre_weight, genre_mode=genre_mode
        )
        self.diversity = diversity
        self.n_candidates = n_candidates

    # ------------------------------------------------------------------ fit
    def fit(self, tracks: pd.DataFrame) -> ContentRecommender:
        self.tracks_ = tracks.reset_index(drop=True)
        self.vectors_ = self.features.fit_transform(self.tracks_)
        profile = self.tracks_[PROFILE_FEATURES].astype(float)
        self._feature_mean = profile.mean()
        self._feature_std = profile.std().replace(0, 1.0)
        self.index_ = pd.Series(np.arange(len(self.tracks_)), index=self.tracks_["track_id"])
        self._search_text = (
            self.tracks_["track_name"].str.lower() + " " + self.tracks_["artists"].str.lower()
        )
        return self

    # -------------------------------------------------------------- helpers
    def to_indices(self, track_ids: Iterable[str]) -> np.ndarray:
        ids = list(track_ids)
        unknown = [t for t in ids if t not in self.index_.index]
        if unknown:
            raise KeyError(f"Unknown track ids: {unknown}")
        return self.index_.loc[ids].to_numpy()

    def profile(self, seed_idx: Sequence[int]) -> np.ndarray:
        vec = self.vectors_[np.asarray(seed_idx)].mean(axis=0)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def search(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Case-insensitive search over title and artists, most popular first."""
        terms = query.lower().split()
        if not terms:
            return self.tracks_.head(0)
        mask = np.ones(len(self.tracks_), dtype=bool)
        for term in terms:
            mask &= self._search_text.str.contains(term, regex=False).to_numpy()
        return self.tracks_[mask].head(limit)

    # ------------------------------------------------------------ inference
    def recommend_indices(
        self,
        seed_idx: Sequence[int],
        k: int = 10,
        diversity: float | None = None,
        candidate_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Core ranking routine. Returns (track indices, similarity scores)."""
        diversity = self.diversity if diversity is None else diversity
        sims = self.vectors_ @ self.profile(seed_idx)
        sims[np.asarray(seed_idx)] = -np.inf
        if candidate_mask is not None:
            sims[~candidate_mask] = -np.inf
        n_valid = int(np.isfinite(sims).sum())
        n_cand = min(max(self.n_candidates, k), n_valid)
        if n_cand == 0:
            return np.array([], dtype=int), np.array([])
        cand = np.argpartition(-sims, n_cand - 1)[:n_cand]
        order = mmr_rerank(self.vectors_[cand], sims[cand], k, diversity)
        picked = cand[order]
        return picked, sims[picked]

    def recommend(
        self,
        seed_ids: Sequence[str],
        k: int = 10,
        diversity: float | None = None,
        exclude_seed_artists: bool = False,
        allow_explicit: bool = True,
        min_popularity: int = 0,
        genres: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Recommend ``k`` tracks for the given seed track ids, with optional filters."""
        seed_idx = self.to_indices(seed_ids)
        t = self.tracks_
        mask = (t["popularity"] >= min_popularity).to_numpy().copy()
        if not allow_explicit:
            mask &= ~t["explicit"].to_numpy()
        if exclude_seed_artists:
            seed_artists = set(t.loc[seed_idx, "primary_artist"])
            mask &= ~t["primary_artist"].isin(seed_artists).to_numpy()
        if genres:
            wanted = set(genres)
            mask &= t["genres"].map(lambda gs: bool(wanted.intersection(gs))).to_numpy()

        idx, scores = self.recommend_indices(seed_idx, k, diversity, mask)
        out = t.loc[idx].copy()
        out.insert(0, "similarity", np.round(scores, 4))
        return out.reset_index(drop=True)

    def explain(self, seed_ids: Sequence[str], track_id: str, max_traits: int = 3) -> dict:
        """Human-readable reasons a track matches the seeds.

        A trait is reported when the seeds are *distinctive* on a feature
        (far from the catalogue average, in standard deviations) and the
        recommended track shares that trait. This avoids trivial reasons
        such as "both have low instrumentalness" for two pop songs.
        """
        seed_idx = self.to_indices(seed_ids)
        (rec_idx,) = self.to_indices([track_id])
        seeds = self.tracks_.loc[seed_idx]
        rec = self.tracks_.loc[rec_idx]

        seed_profile = seeds[PROFILE_FEATURES].astype(float).mean()
        track_profile = rec[PROFILE_FEATURES].astype(float)
        seed_z = (seed_profile - self._feature_mean) / self._feature_std
        rec_z = (track_profile - self._feature_mean) / self._feature_std
        same_side = np.sign(seed_z) == np.sign(rec_z)
        strength = np.minimum(seed_z.abs(), rec_z.abs()).where(same_side, 0.0)
        traits = [
            f"{'high' if seed_z[f] > 0 else 'low'} {f}"
            for f in strength.sort_values(ascending=False).index[:max_traits]
            if strength[f] >= 0.5
        ]
        seed_genres = {g for gs in seeds["genres"] for g in gs}
        return {
            "similarity": float(self.vectors_[rec_idx] @ self.profile(seed_idx)),
            "shared_genres": sorted(seed_genres.intersection(rec["genres"])),
            "shared_traits": traits,
            "seed_profile": seed_profile.to_dict(),
            "track_profile": track_profile.to_dict(),
            "tempo_diff_bpm": float(abs(rec["tempo"] - seeds["tempo"].mean())),
        }

    def explain_text(self, seed_ids: Sequence[str], track_id: str) -> str:
        ex = self.explain(seed_ids, track_id)
        parts = []
        if ex["shared_genres"]:
            parts.append(", ".join(ex["shared_genres"]))
        if ex["shared_traits"]:
            parts.append(", ".join(ex["shared_traits"]))
        if ex["tempo_diff_bpm"] < 5:
            parts.append("same tempo")
        return " · ".join(parts) or "overall audio similarity"
