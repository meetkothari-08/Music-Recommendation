import numpy as np
import pytest

from music_recommender import ContentRecommender
from music_recommender.recommender import mmr_rerank


@pytest.fixture(scope="module")
def rec(tracks):
    return ContentRecommender().fit(tracks)


def seed_of(rec, artist):
    return rec.tracks_[rec.tracks_.primary_artist == artist].track_id.iloc[0]


def test_recommend_never_returns_seed(rec):
    seed = seed_of(rec, "metal_artist_0")
    out = rec.recommend([seed], k=10)
    assert len(out) == 10
    assert seed not in set(out.track_id)


def test_recommendations_stay_in_genre(rec):
    out = rec.recommend([seed_of(rec, "classical_artist_0")], k=10)
    assert all("classical" in g for g in out.genres)


def test_audio_only_still_finds_similar_sounding_tracks(tracks):
    rec = ContentRecommender(genre_weight=0.0, diversity=0.0).fit(tracks)
    out = rec.recommend([seed_of(rec, "classical_artist_0")], k=10)
    assert np.mean(["classical" in g for g in out.genres]) >= 0.9


def test_filters(rec):
    seed = seed_of(rec, "pop_artist_0")
    out = rec.recommend(
        [seed],
        k=10,
        exclude_seed_artists=True,
        allow_explicit=False,
        min_popularity=20,
        genres=["pop", "edm"],
    )
    assert "pop_artist_0" not in set(out.primary_artist)
    assert not out.explicit.any()
    assert (out.popularity >= 20).all()
    assert all({"pop", "edm"} & set(g) for g in out.genres)


def test_impossible_filter_returns_empty(rec):
    out = rec.recommend([seed_of(rec, "pop_artist_0")], min_popularity=101)
    assert out.empty


def test_multi_seed_profile(rec):
    seeds = [seed_of(rec, "classical_artist_0"), seed_of(rec, "classical_artist_1")]
    out = rec.recommend(seeds, k=5)
    assert not set(seeds) & set(out.track_id)


def test_unknown_seed_raises(rec):
    with pytest.raises(KeyError):
        rec.recommend(["does-not-exist"])


def test_search_is_case_insensitive_and_multi_term(rec):
    hits = rec.search("METAL_ARTIST_1 song")
    assert len(hits) > 0
    assert (hits.primary_artist == "metal_artist_1").all()
    assert rec.search("").empty


def test_explain(rec):
    seed = seed_of(rec, "classical_artist_0")
    top = rec.recommend([seed], k=1).track_id[0]
    ex = rec.explain([seed], top)
    assert ex["shared_genres"] == ["classical"]
    assert 0 < ex["similarity"] <= 1.0 + 1e-6
    assert any("acousticness" in t or "energy" in t for t in ex["shared_traits"])
    assert "classical" in rec.explain_text([seed], top)


def test_mmr_without_diversity_is_plain_ranking():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(20, 4))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    rel = rng.random(20)
    np.testing.assert_array_equal(mmr_rerank(vecs, rel, 5, 0.0), np.argsort(-rel)[:5])


def test_mmr_avoids_near_duplicates():
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    rel = np.array([1.0, 0.99, 0.5])
    assert list(mmr_rerank(vecs, rel, 2, 0.0)) == [0, 1]
    assert list(mmr_rerank(vecs, rel, 2, 0.6)) == [0, 2]
