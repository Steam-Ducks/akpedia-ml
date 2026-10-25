"""Suggest a category by comparing the document with one centroid per category.

A centroid is the normalized mean of the vectors of the documents known to belong to
a category (see ``scripts/build_centroids.py``). A new document is embedded the same
way, compared by cosine similarity with every centroid, and the similarities turned
into probabilities with a temperature-scaled softmax.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import CENTROIDS_PATH

_LABELS_KEY = "labels"
_VECTORS_KEY = "vectors"


@dataclass(frozen=True)
class CategoryScores:
    """Winning category, its probability and the probability of every category."""

    category: str
    confidence: float
    scores: dict[str, float]


@dataclass(frozen=True)
class CategoryCentroids:
    """One unit-length vector per category, in the order of ``labels``."""

    labels: tuple[str, ...]
    vectors: np.ndarray

    @property
    def dimensions(self) -> int:
        return int(self.vectors.shape[1])

    def score(self, document: np.ndarray, *, temperature: float) -> CategoryScores:
        """Rank the categories for a unit-length ``document`` vector."""
        if document.shape != (self.dimensions,):
            raise ValueError(
                f"Document vector has shape {document.shape}, but the centroids have {self.dimensions} "
                "dimensions. Rebuild them with the embedding model the service is running."
            )
        probabilities = _softmax((self.vectors @ document) / temperature)
        best = int(np.argmax(probabilities))
        return CategoryScores(
            category=self.labels[best],
            confidence=float(probabilities[best]),
            scores={label: float(p) for label, p in zip(self.labels, probabilities, strict=True)},
        )


def sample_chunks(chunks: Sequence[str], limit: int) -> list[str]:
    """Keep at most ``limit`` chunks, spread evenly across the document.

    Taking the first ones would classify a long document by its cover, table of
    contents and introduction only.
    """
    if len(chunks) <= limit:
        return list(chunks)
    return [chunks[index * len(chunks) // limit] for index in range(limit)]


def document_vector(chunk_vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Average the chunk vectors into one unit-length vector for the document.

    The mean of unit vectors is shorter than 1, so it is normalized again: otherwise
    the dot product with a centroid would no longer be the cosine similarity.
    """
    matrix = np.asarray(chunk_vectors, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("At least one chunk vector is needed.")
    return _unit(matrix.mean(axis=0))


def build_centroids(labels: Sequence[str], document_vectors: Sequence[np.ndarray]) -> CategoryCentroids:
    """Group document vectors by label and reduce each group to its normalized mean."""
    if len(labels) != len(document_vectors):
        raise ValueError("Every document vector needs exactly one label.")
    grouped: dict[str, list[np.ndarray]] = {}
    for label, vector in zip(labels, document_vectors, strict=True):
        grouped.setdefault(label, []).append(vector)
    if not grouped:
        raise ValueError("No documents to build centroids from.")

    ordered = sorted(grouped)
    vectors = np.stack([_unit(np.mean(grouped[label], axis=0)) for label in ordered])
    return CategoryCentroids(labels=tuple(ordered), vectors=vectors)


def save_centroids(centroids: CategoryCentroids, path: Path) -> None:
    """Write the centroids in the format :func:`load_centroids` reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez(
            file,
            **{_LABELS_KEY: np.array(centroids.labels, dtype=str), _VECTORS_KEY: centroids.vectors.astype(np.float32)},
        )


def load_centroids(path: Path) -> CategoryCentroids | None:
    """Read the centroids file, or return ``None`` when there is none.

    A missing file is an expected state (no labelled base yet) and only disables the
    category suggestion. A file that exists but is malformed is a deployment mistake,
    so it raises instead of being silently ignored.
    """
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            labels = tuple(str(label) for label in data[_LABELS_KEY])
            vectors = np.asarray(data[_VECTORS_KEY], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Could not read the category centroids at '{path}': {exc}") from exc

    if vectors.ndim != 2 or not labels or vectors.shape[0] != len(labels):
        raise ValueError(
            f"Malformed category centroids at '{path}': {len(labels)} labels for vectors of shape {vectors.shape}."
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"Malformed category centroids at '{path}': repeated labels.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(np.isfinite(vectors)) or np.any(norms == 0):
        raise ValueError(f"Malformed category centroids at '{path}': empty or non-finite vectors.")
    return CategoryCentroids(labels=labels, vectors=vectors / norms)


@lru_cache(maxsize=1)
def get_centroids() -> CategoryCentroids | None:
    """Return the centroids at :data:`~app.config.CENTROIDS_PATH`, read once per process."""
    return load_centroids(CENTROIDS_PATH)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not norm or not np.isfinite(norm):
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return vector / norm


def _softmax(values: np.ndarray) -> np.ndarray:
    # Shifting by the maximum leaves the result unchanged and keeps exp() from overflowing.
    exponentials = np.exp(values - values.max())
    return exponentials / exponentials.sum()
