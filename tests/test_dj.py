import numpy as np
import pandas as pd
import pytest

from music_recommender import ContentRecommender
from music_recommender.data import clean
from music_recommender.dj import (
    DJSetBuilder,
    SetRequest,
    camelot,
    key_compatibility,
    key_relation,
    plan_transition,
    target_curve,
    tempo_match,
)
from music_recommender.dj_evaluation import evaluate_sets, random_requests


@pytest.fixture(scope="module")
def builder(tracks):
    return DJSetBuilder(ContentRecommender().fit(tracks), beam_width=8, pool_size=200)


def test_camelot_wheel():
    assert camelot(0, 1) == "8B"  # C major
    assert camelot(9, 0) == "8A"  # A minor (relative of C major)
    assert camelot(7, 1) == "9B"  # G major
    assert camelot(4, 0) == "9A"  # E minor
    assert camelot(-1, 1) == "?"


def test_key_compatibility_rules():
    assert key_compatibility(8, False, 8, False) == 1.0
    assert key_compatibility(8, False, 9, False) == 0.9
    assert key_compatibility(12, False, 1, False) == 0.9  # wraps around the wheel
    assert key_compatibility(8, False, 8, True) == 0.85
    assert key_compatibility(8, False, 10, False) == 0.6  # +2 boost
    assert key_compatibility(8, False, 6, False) == 0.0  # -2 is not a boost
    assert key_compatibility(8, False, 2, False) == 0.0
    assert key_relation(8, False, 9, False) == "adjacent key (±1)"
    assert key_relation(8, False, 3, True) == "key clash"


def test_tempo_match_handles_half_and_double_time():
    change, factor, score = tempo_match(120.0, 123.0)
    assert change == pytest.approx(120 / 123 - 1) and factor == 1.0 and score > 0.6
    change, factor, _ = tempo_match(140.0, 70.0)
    assert factor == 2.0 and change == pytest.approx(0.0)
    _, _, score = tempo_match(100.0, 130.0)
    assert score == 0.0


def test_target_curve():
    curve = target_curve("sad", "happy", 5)
    assert curve.shape == (5, 2)
    np.testing.assert_allclose(curve[0], [0.2, 0.3])
    np.testing.assert_allclose(curve[-1], [0.8, 0.65])
    peak = target_curve("chill", "chill", 11, arc="peak")
    assert 5 <= int(np.argmax(peak[:, 1])) <= 8  # energy peaks ~70% of the way
    with pytest.raises(ValueError):
        target_curve("sad", "elated", 5)


def test_build_respects_dj_constraints(builder):
    dj_set = builder.build(SetRequest("calm", "energetic", 8))
    t = dj_set.tracks
    assert len(t) == 8 and len(dj_set.transitions) == 7
    ids = builder.rec.to_indices(t.track_id)
    assert builder.artist[ids].tolist() == list(dict.fromkeys(builder.artist[ids]))  # no repeats
    assert len(set(builder.song_group[ids])) == 8
    assert dj_set.summary["mood_error"] < 0.4
    assert {"smooth_share", "mood_error", "key_clash_share"} <= set(dj_set.summary)


def test_build_from_seed_and_genre(builder):
    seed = builder.tracks[builder.tracks.genres.map(lambda g: "edm" in g)].track_id.iloc[0]
    dj_set = builder.build(SetRequest(None, "energetic", 4, seed_track_id=seed, genres=["edm"]))
    assert dj_set.tracks.track_id.iloc[0] == seed
    assert all("edm" in g for g in dj_set.tracks.genres)


def test_beam_search_beats_mood_only_ordering(builder):
    req = SetRequest("chill", "happy", 6, genres=["pop"])
    beam = builder.build(req)
    shuffled = beam.tracks.track_id.sample(frac=1, random_state=1)
    worse = builder.assemble(builder.rec.to_indices(shuffled), beam.targets, req)
    assert beam.summary["mean_transition_score"] >= worse.summary["mean_transition_score"]


def test_non_dj_genres_excluded_unless_requested(raw):
    kids = raw[raw.track_genre == "pop"].copy()
    kids["track_id"] = "kid-" + kids["track_id"]
    kids["artists"] = "kid-" + kids["artists"]
    kids["track_genre"] = "kids"
    tracks, _ = clean(pd.concat([raw, kids], ignore_index=True))
    b = DJSetBuilder(ContentRecommender().fit(tracks), beam_width=4, pool_size=100)
    default = b.build(SetRequest("happy", "happy", 5))
    assert not any("kids" in g for g in default.tracks.genres)
    asked = b.build(SetRequest("happy", "happy", 3, genres=["kids"]))
    assert all("kids" in g for g in asked.tracks.genres)


def test_invalid_requests(builder):
    for bad in (
        SetRequest(n_tracks=1),
        SetRequest("sad", "elated"),
        SetRequest(arc="zigzag"),
        SetRequest(None, "happy"),
    ):
        with pytest.raises(ValueError):
            builder.build(bad)
    with pytest.raises(ValueError, match="Not enough tracks"):
        builder.build(SetRequest("sad", "happy", 5, min_popularity=101))


def _track(**kw):
    base = {"tempo": 120.0, "energy": 0.6, "danceability": 0.7, "duration_ms": 200_000}
    return pd.Series({**base, **kw})


def test_transition_techniques():
    smooth = plan_transition(_track(), _track(tempo=118.0), 120 / 118 - 1, 1.0, 1.0, "same key")
    assert smooth.technique == "long blend" and smooth.mix_beats == 32
    assert "Speed up" in smooth.tempo_instruction and not smooth.warnings
    assert smooth.start_at == "3:00"  # 200s - (32 + 8 beats at 120 BPM = 20s)
    clash = plan_transition(_track(), _track(), 0.0, 1.0, 0.0, "key clash")
    assert clash.technique == "short blend" and clash.warnings
    far = plan_transition(_track(), _track(tempo=95.0), 120 / 95 - 1, 1.0, 1.0, "same key")
    assert far.technique == "echo out"
    lift = plan_transition(_track(energy=0.3), _track(energy=0.8), 0.0, 1.0, 1.0, "same key")
    assert any("energy lift" in s for s in lift.steps)


def test_dj_evaluation_runs(builder):
    reqs = random_requests(builder, n=4, n_tracks=4, seed=1)
    results = evaluate_sets(builder, reqs)
    assert len(results) == 5
    beam = results.loc["DJ beam search (final)"]
    assert beam.mixable_share >= results.loc["Random (mood-matched)"].mixable_share
    assert 0 <= beam.smooth_share <= 1
