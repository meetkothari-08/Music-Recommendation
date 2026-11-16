"""Project-wide paths and constants."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MUSIC_REC_DATA_DIR", PROJECT_ROOT / "data"))
RAW_PATH = DATA_DIR / "raw" / "spotify_tracks.csv"
PROCESSED_PATH = DATA_DIR / "processed" / "tracks.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Public dataset: ~114k Spotify tracks across 114 genres with audio features.
# https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
DATASET_URL = (
    "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
)

# Continuous audio descriptors used to represent how a track *sounds*.
AUDIO_FEATURES: list[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# Features bounded in [0, 1]; used for human-readable explanations and charts.
PROFILE_FEATURES: list[str] = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]

MIN_DURATION_MS = 30_000  # shorter tracks are intros/skits/sound effects
MAX_DURATION_MS = 15 * 60_000  # longer tracks are mostly DJ mixes / audiobooks
