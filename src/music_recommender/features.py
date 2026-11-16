"""Turn tracks into dense vectors whose cosine similarity means "sounds alike".

Each track vector is the concatenation of two blocks:

* **Audio block** - the continuous Spotify audio features, passed through a
  quantile transform. Several features (speechiness, instrumentalness,
  liveness, duration) are extremely skewed, and plain standardisation would
  let a handful of outliers dominate distances. Mapping every feature to a
  standard normal by rank puts them on an equal footing.
* **Genre block** - either a one-hot encoding of the track's genres, or a
  learned genre *embedding*. The embedding is derived from how often genres
  co-occur on the same song (positive PMI + truncated SVD), so related
  genres such as ``rock`` / ``alt-rock`` / ``grunge`` end up close together
  instead of being treated as unrelated categories.

Both blocks are scaled to unit norm and then weighted, so ``genre_weight``
directly controls how much genre matters relative to sound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer, QuantileTransformer, normalize

from music_recommender.config import AUDIO_FEATURES

GenreMode = Literal["onehot", "embedding"]


def genre_embeddings(multi_hot: np.ndarray, n_components: int = 24, seed: int = 0) -> np.ndarray:
    """Learn genre vectors from genre co-occurrence (PPMI + SVD).

    Returns an array of shape ``(n_genres, n_components)`` with unit-norm rows.
    """
    counts = multi_hot.T @ multi_hot  # (G, G) co-occurrence, diagonal = genre size
    total = counts.sum()
    row = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((counts * total) / (row @ row.T))
    ppmi = np.nan_to_num(np.maximum(pmi, 0.0), nan=0.0, posinf=0.0)
    n_components = min(n_components, ppmi.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    emb = svd.fit_transform(ppmi)
    return normalize(emb)


@dataclass
class FeatureBuilder:
    audio_weight: float = 1.0
    genre_weight: float = 1.0
    genre_mode: GenreMode = "embedding"
    n_genre_components: int = 24
    seed: int = 0

    def fit(self, tracks: pd.DataFrame) -> FeatureBuilder:
        n_quantiles = min(1000, len(tracks))
        self.quantiles_ = QuantileTransformer(
            n_quantiles=n_quantiles, output_distribution="normal", random_state=self.seed
        ).fit(tracks[AUDIO_FEATURES].to_numpy(dtype=float))
        self.mlb_ = MultiLabelBinarizer().fit(tracks["genres"])
        if self.genre_mode == "embedding":
            multi_hot = self.mlb_.transform(tracks["genres"]).astype(float)
            self.genre_vectors_ = genre_embeddings(multi_hot, self.n_genre_components, self.seed)
        return self

    @property
    def genres(self) -> list[str]:
        return list(self.mlb_.classes_)

    def audio_matrix(self, tracks: pd.DataFrame) -> np.ndarray:
        """Quantile-normalised audio features (one column per feature)."""
        return self.quantiles_.transform(tracks[AUDIO_FEATURES].to_numpy(dtype=float))

    def genre_matrix(self, tracks: pd.DataFrame) -> np.ndarray:
        multi_hot = self.mlb_.transform(tracks["genres"]).astype(float)
        if self.genre_mode == "embedding":
            return normalize(multi_hot @ self.genre_vectors_)
        return normalize(multi_hot)

    def transform(self, tracks: pd.DataFrame) -> np.ndarray:
        audio = self.audio_matrix(tracks) / np.sqrt(len(AUDIO_FEATURES))
        blocks = [self.audio_weight * audio]
        if self.genre_weight > 0:
            blocks.append(self.genre_weight * self.genre_matrix(tracks))
        return normalize(np.hstack(blocks)).astype(np.float32)

    def fit_transform(self, tracks: pd.DataFrame) -> np.ndarray:
        return self.fit(tracks).transform(tracks)
