import pandas as pd

from music_recommender.data import clean


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
