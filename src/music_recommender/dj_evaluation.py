"""Evaluate DJ-set sequencing against simpler ways of building a playlist.

For many random set requests (start mood, end mood, arc, optional genre),
each method produces an ordered list of tracks. All lists are scored by the
same functions in :mod:`music_recommender.dj`, so the comparison is fair:

* **Random (mood-matched)** - a random track among the 50 closest to each
  step's target mood.
* **Mood only (greedy)** - the closest track to each step's target mood.
* **Similar songs** - the content recommender's top matches for the
  opening track, in similarity order (a typical "radio" playlist).
* **DJ greedy (beam = 1)** - the DJ objective, one step at a time.
* **DJ beam search** - the full method (beam width 20).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from music_recommender.dj import ARCS, MOODS, DJSetBuilder, SetRequest, target_curve


def random_requests(
    builder: DJSetBuilder, n: int = 200, n_tracks: int = 10, seed: int = 0
) -> list[SetRequest]:
    rng = np.random.default_rng(seed)
    t = builder.tracks
    counts = pd.Series([g for gs in t.loc[builder.base_mask, "genres"] for g in gs]).value_counts()
    common = sorted(counts[counts >= 150].index)
    moods, arcs = sorted(MOODS), sorted(ARCS)
    reqs = []
    for _ in range(n):
        genres = [str(rng.choice(common))] if common and rng.random() < 0.5 else []
        reqs.append(
            SetRequest(
                str(rng.choice(moods)),
                str(rng.choice(moods)),
                n_tracks,
                str(rng.choice(arcs)),
                genres=genres,
            )
        )
    return reqs


def _eligible(builder: DJSetBuilder, req: SetRequest) -> np.ndarray:
    return np.flatnonzero(builder._mask(req))


def _mood_dist(builder: DJSetBuilder, idx: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.hypot(builder.valence[idx] - target[0], builder.energy[idx] - target[1])


def _pick_by_mood(builder, req, targets, choose) -> list[int]:
    elig = _eligible(builder, req)
    path: list[int] = []
    for target in targets:
        free = elig[~np.isin(builder.song_group[elig], builder.song_group[path])]
        path.append(int(choose(free, _mood_dist(builder, free, target))))
    return path


def random_mood_path(builder, req, targets, rng) -> list[int]:
    def choose(free, dist):
        k = min(50, len(free))
        return rng.choice(free[np.argpartition(dist, k - 1)[:k]])

    return _pick_by_mood(builder, req, targets, choose)


def mood_greedy_path(builder, req, targets, rng) -> list[int]:
    return _pick_by_mood(builder, req, targets, lambda free, dist: free[np.argmin(dist)])


def similar_songs_path(builder, req, targets, rng) -> list[int]:
    elig = _eligible(builder, req)
    first = int(elig[np.argmin(_mood_dist(builder, elig, targets[0]))])
    mask = np.zeros(len(builder.tracks), dtype=bool)
    mask[elig] = True
    recs, _ = builder.rec.recommend_indices(
        [first], req.n_tracks - 1, diversity=0.0, candidate_mask=mask
    )
    return [first, *map(int, recs)]


def dj_path(beam_width: int) -> Callable:
    def build(builder, req, targets, rng) -> list[int]:
        dj_set = builder.build(req, beam_width=beam_width)
        return builder.rec.to_indices(dj_set.tracks["track_id"]).tolist()

    return build


METHODS: dict[str, Callable] = {
    "Random (mood-matched)": random_mood_path,
    "Mood only (greedy)": mood_greedy_path,
    "Similar songs": similar_songs_path,
    "DJ greedy (beam = 1)": dj_path(1),
    "DJ beam search (final)": dj_path(20),
}


def evaluate_sets(
    builder: DJSetBuilder,
    requests: list[SetRequest],
    methods: dict[str, Callable] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean metrics per method (rows) across all requests."""
    methods = methods or METHODS
    rows = []
    for name, method in methods.items():
        rng = np.random.default_rng(seed)
        for i, req in enumerate(requests):
            targets = target_curve(req.start_mood, req.end_mood, req.n_tracks, req.arc)
            start = time.perf_counter()
            path = method(builder, req, targets, rng)
            ms = (time.perf_counter() - start) * 1000
            dj_set = builder.assemble(path, targets, req)
            genres = dj_set.tracks["genres"].map(set).tolist()
            coherence = np.mean([bool(a & b) for a, b in zip(genres, genres[1:], strict=False)])
            rows.append(
                {
                    "method": name,
                    "request": i,
                    **dj_set.summary,
                    "genre_coherence": coherence,
                    "ms": ms,
                }
            )
    df = pd.DataFrame(rows)
    agg = df.groupby("method", sort=False).mean(numeric_only=True).drop(columns="request")
    n = df.groupby("method", sort=False)["smooth_share"].count()
    agg["smooth_ci95"] = 1.96 * df.groupby("method", sort=False)["smooth_share"].std() / np.sqrt(n)
    return agg
