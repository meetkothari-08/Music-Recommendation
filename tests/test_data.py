import pandas as pd
import pytest

from music_recommender.data import (
    base_title,
    build,
    clean,
    download,
    load_raw,
    load_tracks,
    recording_title,
)


def test_clean_merges_multi_genre_duplicates(raw):
    dup = raw.iloc[[0]].copy()
    dup["track_genre"] = "pop"
    cleaned, _ = clean(pd.concat([raw, dup], ignore_index=True))
    row = cleaned[cleaned.track_id == raw.track_id.iloc[0]]
    assert len(row) == 1
    assert set(row.genres.iloc[0]) == {raw.track_genre.iloc[0], "pop"}


def test_clean_keeps_most_popular_rerelease_and_unions_genres(raw):
    base = raw.iloc[[0]].copy()
    rerelease = base.copy()
    rerelease["track_id"] = "rerelease"
    rerelease["track_name"] = base.track_name.iloc[0].upper() + "  "  # case/space variant
    rerelease["popularity"] = 100
    rerelease["track_genre"] = "edm"
    cleaned, _ = clean(pd.concat([raw, rerelease], ignore_index=True))
    assert "rerelease" in set(cleaned.track_id)
    assert base.track_id.iloc[0] not in set(cleaned.track_id)
    merged = cleaned[cleaned.track_id == "rerelease"].genres.iloc[0]
    assert {"edm", base.track_genre.iloc[0]} <= set(merged)


def test_clean_drops_invalid_rows(raw):
    bad = raw.iloc[:3].copy()
    bad["track_id"] = ["a", "b", "c"]
    bad["track_name"] = ["x", "y", None]
    bad.loc[bad.index[0], "tempo"] = 0
    bad.loc[bad.index[1], "duration_ms"] = 10_000
    cleaned, report = clean(pd.concat([raw, bad], ignore_index=True))
    assert not {"a", "b", "c"} & set(cleaned.track_id)
    assert report.to_frame().rows_removed.sum() == len(raw) + 3 - len(cleaned)


def test_clean_splits_artists(raw):
    feat = raw.iloc[[0]].copy()
    feat["track_id"] = "feat"
    feat["track_name"] = "collab"
    feat["artists"] = "Main;Guest"
    cleaned, _ = clean(pd.concat([raw, feat], ignore_index=True))
    row = cleaned.set_index("track_id").loc["feat"]
    assert row.primary_artist == "Main"
    assert row.artists == "Main, Guest"


def _variant(raw, track_id, title, popularity=50):
    row = raw.iloc[[0]].copy()
    row["track_id"] = track_id
    row["track_name"] = title
    row["popularity"] = popularity
    return row


def test_remasters_merge_but_live_versions_stay(raw):
    variants = pd.concat(
        [
            _variant(raw, "orig", "Anthem", 40),
            _variant(raw, "remaster", "Anthem - Remastered 2011", 90),
            _variant(raw, "mono", "Anthem (Mono)", 10),
            _variant(raw, "live", "Anthem - Live at Wembley", 30),
        ]
    )
    cleaned, _ = clean(pd.concat([raw, variants], ignore_index=True))
    ids = set(cleaned.track_id)
    assert "remaster" in ids  # most popular release of the recording wins
    assert not {"orig", "mono"} & ids
    assert "live" in ids  # a live version is a different recording
    groups = cleaned.set_index("track_id").song_group
    assert groups["live"] == groups["remaster"]  # ...but the same song


def test_title_normalisation():
    titles = pd.Series(
        [
            "Hey Jude - 2015 Remaster",
            "Help! (Remastered 2009)",
            "Creep - Acoustic",
            "(I Can't Get No) Satisfaction",
            "Song (feat. X) [Radio Edit]",
            "(Intro)",
        ]
    )
    assert recording_title(titles).tolist() == [
        "hey jude",
        "help!",
        "creep - acoustic",
        "(i can't get no) satisfaction",
        "song (feat. x) [radio edit]",
        "(intro)",
    ]
    assert base_title(titles).tolist() == [
        "hey jude",
        "help!",
        "creep",
        "(i can't get no) satisfaction",
        "song",
        "(intro)",
    ]


def test_download_and_build_roundtrip(raw, tmp_path):
    src = tmp_path / "source.csv"
    raw.to_csv(src)  # the real file also has a leading unnamed index column
    raw_path = download(url=src.as_uri(), dest=tmp_path / "raw" / "tracks.csv")
    assert raw_path.exists()
    out = tmp_path / "processed" / "tracks.parquet"
    report = build(raw_path=raw_path, out_path=out)
    loaded = load_tracks(out)
    assert len(loaded) == report.steps[-1][1]
    assert isinstance(loaded.genres.iloc[0], list)


def test_load_raw_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"track_id": ["a"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_raw(bad)


def test_falls_back_to_pickle_when_parquet_is_blocked(raw, tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise ImportError("DLL load failed: blocked by Application Control")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", blocked)
    monkeypatch.setattr(pd, "read_parquet", blocked)
    raw_csv = tmp_path / "raw.csv"
    raw.to_csv(raw_csv, index=False)
    out = tmp_path / "processed" / "tracks.parquet"
    build(raw_path=raw_csv, out_path=out)
    assert not out.exists() and out.with_suffix(".pkl").exists()
    loaded = load_tracks(out)
    assert len(loaded) > 0 and isinstance(loaded.genres.iloc[0], list)
