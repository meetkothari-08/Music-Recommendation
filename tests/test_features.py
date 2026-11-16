import numpy as np
import pytest

from music_recommender.config import AUDIO_FEATURES
from music_recommender.features import FeatureBuilder, genre_embeddings


@pytest.mark.parametrize("mode", ["onehot", "embedding"])
def test_vectors_are_unit_norm(tracks, mode):
    X = FeatureBuilder(genre_mode=mode, n_genre_components=2).fit_transform(tracks)
    assert X.shape[0] == len(tracks)
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, rtol=1e-5)


def test_zero_genre_weight_uses_audio_only(tracks):
    X = FeatureBuilder(genre_weight=0.0).fit_transform(tracks)
    assert X.shape[1] == len(AUDIO_FEATURES)


def test_cooccurring_genres_get_similar_embeddings():
    # genres 0/1 often appear together, as do 2/3; the two pairs never mix
    rows = [[1, 1, 0, 0]] * 5 + [[1, 0, 0, 0]] * 3 + [[0, 0, 1, 1]] * 5 + [[0, 0, 0, 1]] * 3
    emb = genre_embeddings(np.array(rows, dtype=float), n_components=2)
    assert emb.shape == (4, 2)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-6)
    assert emb[0] @ emb[1] > emb[0] @ emb[2]
    assert emb[2] @ emb[3] > emb[1] @ emb[3]
