import numpy as np
import pytest

pytest.importorskip("lightgbm")

from music_recommender.evaluation import (  # noqa: E402
    RANKER_NAME,
    Evaluator,
    RandomRecommender,
    default_model_suite,
    make_artist_tasks,
    paired_difference,
    ranker_diversity_sweep,
    task_artists,
)
from music_recommender.ranker import (  # noqa: E402
    FEATURE_NAMES,
    LEAKY_FIELDS,
    LearnedRanker,
    train_ranker,
)
from music_recommender.recommender import ContentRecommender  # noqa: E402

SMALL = {"n_estimators": 20, "min_child_samples": 5}


@pytest.fixture(scope="module")
def split(tracks):
    """Validation / test tasks, and a ranker trained on the remaining artists."""
    val = make_artist_tasks(tracks, n_tasks=4, seed=7)
    test = make_artist_tasks(tracks, n_tasks=4, seed=42)
    excluded = task_artists(tracks, val) | task_artists(tracks, test)
    ranker = train_ranker(tracks, excluded, n_candidates=50, lgbm_params=SMALL)
    return val, test, excluded, ranker


def test_make_artist_tasks_excludes_artists(tracks):
    test = make_artist_tasks(tracks, n_tasks=5, seed=42)
    excluded = task_artists(tracks, test)
    train = make_artist_tasks(tracks, n_tasks=len(tracks), seed=1, exclude_artists=excluded)
    assert train and not task_artists(tracks, train) & excluded
    # default draws are unchanged by the new argument
    again = make_artist_tasks(tracks, n_tasks=5, seed=42, exclude_artists=None)
    assert all(np.array_equal(a.seeds, b.seeds) for a, b in zip(test, again, strict=True))


def test_train_and_eval_artists_disjoint(split, tracks):
    val, test, excluded, ranker = split
    assert ranker.train_artists_
    assert not ranker.train_artists_ & task_artists(tracks, val)
    assert not ranker.train_artists_ & task_artists(tracks, test)
    assert ranker.train_artists_ | excluded <= set(tracks["primary_artist"])


def test_no_leaky_features(split, tracks):
    *_, ranker = split
    for name in FEATURE_NAMES:
        assert not any(field in name for field in LEAKY_FIELDS), name
    assert list(ranker.model_.feature_name_) == FEATURE_NAMES


def test_features_ignore_artist_and_album(tracks):
    """Scrambling artist/album/title columns must not change any feature."""
    scrambled = tracks.copy()
    rng = np.random.default_rng(0)
    for col in ("artists", "primary_artist", "album_name", "track_name"):
        scrambled[col] = rng.permutation(scrambled[col].to_numpy())
    a = LearnedRanker(ContentRecommender(diversity=0.0).fit(tracks), n_candidates=30)
    b = LearnedRanker(ContentRecommender(diversity=0.0).fit(scrambled), n_candidates=30)
    seeds = np.array([0, 1, 2])
    fa = a.candidate_features(seeds, *a.retrieve(seeds))
    fb = b.candidate_features(seeds, *b.retrieve(seeds))
    np.testing.assert_allclose(fa.to_numpy(), fb.to_numpy())


def test_candidate_features_shape_and_values(split):
    *_, ranker = split
    seeds = np.array([0, 1, 2])
    cand, sims = ranker.retrieve(seeds)
    X = ranker.candidate_features(seeds, cand, sims)
    assert list(X.columns) == FEATURE_NAMES and len(X) == len(cand) == 50
    assert np.isfinite(X.to_numpy()).all()
    assert X["genre_jaccard"].between(0, 1).all()
    assert (X[[c for c in FEATURE_NAMES if c.startswith("absdiff_")]] >= 0).all().all()
    np.testing.assert_array_equal(X["retrieval_rank"], np.arange(50))


def test_ranker_trains_and_recommends(split, tracks):
    _, test, _, ranker = split
    assert ranker.n_train_queries_ > 0
    assert 0 < ranker.train_candidate_recall_ <= 1
    imp = ranker.feature_importance()
    assert set(imp.index) == set(FEATURE_NAMES) and imp.sum() > 0
    for task in test:
        recs, scores = ranker.recommend_indices(task.seeds, k=10)
        assert len(recs) == len(set(recs)) == 10
        assert not set(recs) & set(task.seeds)
        assert np.all(np.diff(scores) <= 1e-12)  # sorted when diversity=0
    ev = Evaluator(tracks, test, k=5)
    assert ev.evaluate(ranker)["hit_rate"] >= ev.evaluate(RandomRecommender(tracks))["hit_rate"]


def test_mmr_after_ranker_and_sweep(split, tracks):
    val, *_, ranker = split
    recs, _ = ranker.recommend_indices(val[0].seeds, k=10, diversity=0.5)
    assert len(set(recs)) == 10
    sweep = ranker_diversity_sweep(Evaluator(tracks, val, k=5), ranker, lambdas=(0.0, 0.5))
    assert list(sweep["diversity_lambda"]) == [0.0, 0.5]


def test_feature_subset(tracks, split):
    *_, excluded, _ = split
    feats = ["cosine_sim", "retrieval_rank"]
    ranker = train_ranker(tracks, excluded, n_candidates=30, features=feats, lgbm_params=SMALL)
    assert list(ranker.feature_importance().index.sort_values()) == sorted(feats)
    with pytest.raises(ValueError):
        LearnedRanker(ContentRecommender().fit(tracks), features=["artists"])


def test_suite_includes_ranker_and_paired_difference(split, tracks):
    _, test, _, ranker = split
    suite = default_model_suite(tracks, ranker=ranker)
    assert len(suite) == 8 and suite[RANKER_NAME] is ranker
    ev = Evaluator(tracks, test, k=5)
    a = ev.per_task(ranker)
    diff = paired_difference(a, a)
    assert diff["mean"] == 0 and diff["wins"] == diff["losses"] == 0
