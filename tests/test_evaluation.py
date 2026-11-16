import numpy as np
import pytest

from music_recommender.evaluation import (
    ContentModelAdapter,
    Evaluator,
    GenrePopularityRecommender,
    PopularityRecommender,
    RandomRecommender,
    default_model_suite,
    intra_list_diversity,
    make_artist_tasks,
    ndcg_at_k,
)
from music_recommender.recommender import ContentRecommender


def test_ndcg():
    assert ndcg_at_k(np.array([1.0, 0, 0]), 1) == pytest.approx(1.0)
    assert ndcg_at_k(np.array([0, 0, 0.0]), 3) == 0.0
    assert ndcg_at_k(np.array([0, 1.0, 0]), 1) == pytest.approx(1 / np.log2(3))


def test_intra_list_diversity():
    same = np.array([[1.0, 0], [1.0, 0]])
    orth = np.array([[1.0, 0], [0, 1.0]])
    assert intra_list_diversity(same) == pytest.approx(0.0)
    assert intra_list_diversity(orth) == pytest.approx(1.0)


def test_tasks_split_seeds_and_targets(tracks):
    tasks = make_artist_tasks(tracks, n_tasks=5, n_seeds=3, min_tracks=6)
    assert len(tasks) == 5
    for t in tasks:
        assert len(t.seeds) == 3
        assert not set(t.seeds) & set(t.targets)
        artists = tracks.loc[np.r_[t.seeds, t.targets], "primary_artist"]
        assert artists.nunique() == 1


class OracleRecommender:
    """Cheats by returning the hidden targets - checks the metric plumbing."""

    def __init__(self, tasks):
        self.lookup = {tuple(t.seeds): t.targets for t in tasks}

    def recommend_indices(self, seed_idx, k):
        return self.lookup[tuple(seed_idx)][:k], np.ones(k)


def test_oracle_scores_perfectly(tracks):
    tasks = make_artist_tasks(tracks, n_tasks=10)
    res = Evaluator(tracks, tasks, k=5).evaluate(OracleRecommender(tasks))
    assert res["hit_rate"] == 1.0
    assert res["recall"] == pytest.approx(1.0)
    assert res["ndcg"] == pytest.approx(1.0)


def test_baselines_and_content_model_run(tracks):
    tasks = make_artist_tasks(tracks, n_tasks=10)
    ev = Evaluator(tracks, tasks, k=5)
    for model in (
        RandomRecommender(tracks),
        PopularityRecommender(tracks),
        GenrePopularityRecommender(tracks),
    ):
        recs, _ = model.recommend_indices(tasks[0].seeds, 5)
        assert len(recs) == 5 and not set(recs) & set(tasks[0].seeds)
    content = ev.evaluate(ContentModelAdapter(ContentRecommender().fit(tracks)))
    rand = ev.evaluate(RandomRecommender(tracks))
    assert content["hit_rate"] >= rand["hit_rate"]
    assert content["genre_precision"] == pytest.approx(1.0)


def test_default_suite_builds(tracks):
    assert len(default_model_suite(tracks)) == 7
