"""Synthetic fixtures so the test suite runs without downloading data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from music_recommender.data import clean

GENRE_CENTRES = {
    # genre: (danceability, energy, acousticness, instrumentalness, valence, tempo)
    "classical": (0.25, 0.10, 0.95, 0.90, 0.15, 80),
    "metal": (0.35, 0.95, 0.02, 0.30, 0.30, 150),
    "edm": (0.75, 0.85, 0.05, 0.60, 0.50, 128),
    "pop": (0.70, 0.70, 0.15, 0.01, 0.65, 115),
}


def make_raw(
    n_artists_per_genre: int = 6, songs_per_artist: int = 8, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for genre, (dance, energy, acoustic, instr, valence, tempo) in GENRE_CENTRES.items():
        for a in range(n_artists_per_genre):
            artist = f"{genre}_artist_{a}"
            # each artist has their own small offset -> artist "style"
            offset = rng.normal(0, 0.04, size=5)
            for s in range(songs_per_artist):
                noise = rng.normal(0, 0.02, size=5)
                d, e, ac, ins, v = np.clip(
                    np.array([dance, energy, acoustic, instr, valence]) + offset + noise, 0, 1
                )
                rows.append(
                    {
                        "track_id": f"{artist}_t{s}",
                        "artists": artist,
                        "album_name": f"{artist} album",
                        "track_name": f"{artist} song {s}",
                        "popularity": int(rng.integers(0, 100)),
                        "duration_ms": int(rng.integers(150_000, 300_000)),
                        "explicit": bool(rng.random() < 0.2),
                        "danceability": d,
                        "energy": e,
                        "loudness": -5 - 10 * (1 - e),
                        "speechiness": 0.05,
                        "acousticness": ac,
                        "instrumentalness": ins,
                        "liveness": 0.1 + rng.random() * 0.1,
                        "valence": v,
                        "tempo": tempo + rng.normal(0, 3),
                        "key": int(rng.integers(0, 12)),
                        "mode": int(rng.integers(0, 2)),
                        "time_signature": 4,
                        "track_genre": genre,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def tracks(raw) -> pd.DataFrame:
    cleaned, _ = clean(raw)
    return cleaned
