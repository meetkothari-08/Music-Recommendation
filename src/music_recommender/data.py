"""Download, clean and persist the track catalogue.

The raw dataset has several quality issues that matter for a recommender:

* the same ``track_id`` appears once per genre it was sampled under, so a
  naive model would recommend the seed track back to the user;
* the same song (title + artists) is often released several times
  (single, album, deluxe, compilation) under different ``track_id`` values;
* some rows have failed audio analysis (``tempo == 0``) or are not songs at
  all (sub-30-second clips, multi-hour mixes).

:func:`clean` resolves these and returns an audit trail of every step.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from music_recommender.config import (
    AUDIO_FEATURES,
    DATASET_URL,
    MAX_DURATION_MS,
    MIN_DURATION_MS,
    PROCESSED_PATH,
    RAW_PATH,
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "popularity",
    "explicit",
    "track_genre",
    *AUDIO_FEATURES,
    "key",
    "mode",
    "time_signature",
]


@dataclass
class CleaningReport:
    """Row counts after each cleaning step, for transparency."""

    steps: list[tuple[str, int]] = field(default_factory=list)

    def log(self, step: str, df: pd.DataFrame) -> None:
        self.steps.append((step, len(df)))
        logger.info("%-55s %8d rows", step, len(df))

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame(self.steps, columns=["step", "rows"])
        out["rows_removed"] = (-out["rows"].diff()).fillna(0).astype(int)
        return out


def download(url: str = DATASET_URL, dest: Path = RAW_PATH, force: bool = False) -> Path:
    """Download the raw CSV (skipped if it already exists)."""
    if dest.exists() and not force:
        logger.info("Raw data already present at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.replace(dest)
    return dest


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {sorted(missing)}")
    return df[REQUIRED_COLUMNS]


def _normalise_text(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


# Suffixes that label the *same recording* (e.g. "Song - Remastered 2011",
# "Song (Mono)", "Song - Single Version"). Live, acoustic and remix versions
# are deliberately NOT stripped: they sound different and are kept as songs.
_SAME_RECORDING_SUFFIX = (
    r"\s*(?:-\s+|\(|\[)\s*"
    r"(?:\d{4}\s+)?(?:digital(?:ly)?\s+)?"
    r"(?:remaster(?:ed)?(?:\s+\d{4})?(?:\s+version)?|mono|stereo|single version|album version)"
    r"\s*[)\]]?\s*$"
)

# Any trailing version label: " - Live", "(Acoustic)", "[Radio Edit]", "(feat. X)".
_ANY_VERSION_SUFFIX = r"\s*(?:\s-\s.*|\([^()]*\)|\[[^\[\]]*\])\s*$"


def recording_title(titles: pd.Series) -> pd.Series:
    """Normalised title with same-recording labels (remaster, mono...) removed."""
    out = _normalise_text(titles)
    for _ in range(2):  # handle stacked suffixes, e.g. "(mono) - remastered"
        out = out.str.replace(_SAME_RECORDING_SUFFIX, "", regex=True).str.strip()
    return out


def base_title(titles: pd.Series) -> pd.Series:
    """Normalised title with every trailing version label removed."""
    out = _normalise_text(titles)
    for _ in range(3):
        stripped = out.str.replace(_ANY_VERSION_SUFFIX, "", regex=True).str.strip()
        out = stripped.where(stripped != "", out)  # never strip a title to nothing
    return out


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Turn the raw genre-sampled rows into one row per unique song."""
    report = CleaningReport()
    df = raw.copy()
    report.log("raw rows", df)

    df = df.dropna(subset=["track_id", "track_name", "artists"])
    report.log("drop rows with missing title/artist", df)

    valid = (df["tempo"] > 0) & (df["duration_ms"].between(MIN_DURATION_MS, MAX_DURATION_MS))
    df = df[valid]
    report.log("drop failed audio analysis / non-song durations", df)

    # One row per track_id, collecting every genre it was listed under.
    genres = df.groupby("track_id")["track_genre"].agg(lambda g: sorted(set(g)))
    df = df.drop_duplicates("track_id").drop(columns="track_genre")
    df = df.merge(genres.rename("genres"), left_on="track_id", right_index=True)
    report.log("merge duplicate track_ids (multi-genre listings)", df)

    # Same recording released multiple times (single, album, remaster...):
    # keep the most popular release, but union the genres from all releases.
    df["_song_key"] = recording_title(df["track_name"]) + "||" + _normalise_text(df["artists"])
    song_genres = df.groupby("_song_key")["genres"].agg(
        lambda gs: sorted({g for lst in gs for g in lst})
    )
    df = df.sort_values("popularity", ascending=False).drop_duplicates("_song_key")
    df["genres"] = df["_song_key"].map(song_genres)
    df = df.drop(columns="_song_key")
    report.log("merge re-releases of the same song", df)

    df["artist_list"] = df["artists"].str.split(";")
    df["primary_artist"] = df["artist_list"].str[0].str.strip()
    df["artists"] = df["artist_list"].str.join(", ")
    df = df.drop(columns="artist_list")
    # Groups all versions of a song (live, acoustic, remix...) so the
    # recommender never suggests another version of a seed.
    df["song_group"] = base_title(df["track_name"]) + "||" + df["primary_artist"].str.lower()
    df["explicit"] = df["explicit"].astype(bool)
    df["popularity"] = df["popularity"].astype(int)

    df = df.sort_values("popularity", ascending=False).reset_index(drop=True)
    return df, report


def _fallback_path(path: Path) -> Path:
    return path.with_suffix(".pkl")


def save_tracks(tracks: pd.DataFrame, path: Path = PROCESSED_PATH) -> Path:
    """Write tracks as parquet, or as a pickle if no parquet engine can load.

    Parquet support lives in a compiled extension of pyarrow, which some
    locked-down machines (e.g. Windows Smart App Control) refuse to load.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tracks.to_parquet(path, index=False)
        _fallback_path(path).unlink(missing_ok=True)
        return path
    except ImportError as exc:
        fallback = _fallback_path(path)
        logger.warning("Parquet unavailable (%s); saving %s instead", exc, fallback.name)
        tracks.to_pickle(fallback)
        return fallback


def build(raw_path: Path = RAW_PATH, out_path: Path = PROCESSED_PATH) -> CleaningReport:
    """Run the full raw -> processed pipeline and save the result."""
    tracks, report = clean(load_raw(raw_path))
    written = save_tracks(tracks, out_path)
    logger.info("Wrote %d tracks to %s", len(tracks), written)
    return report


def processed_exists(path: Path = PROCESSED_PATH) -> bool:
    return path.exists() or _fallback_path(path).exists()


def load_tracks(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    fallback = _fallback_path(path)
    try:
        df = pd.read_parquet(path)
    except (ImportError, FileNotFoundError):
        if not fallback.exists():
            raise
        df = pd.read_pickle(fallback)
    df["genres"] = df["genres"].map(list)
    return df


def ensure_tracks(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load processed tracks, downloading and building them if needed."""
    if not processed_exists(path):
        download()
        build(out_path=path)
    return load_tracks(path)
