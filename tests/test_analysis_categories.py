from pathlib import Path

import numpy as np
import pytest

from app.analysis import (
    CategoryCentroids,
    build_centroids,
    document_vector,
    load_centroids,
    sample_chunks,
    save_centroids,
)


def centroids(**vectors: list[float]) -> CategoryCentroids:
    return build_centroids(list(vectors), [np.asarray(vector, dtype=float) for vector in vectors.values()])


def test_document_vector_is_the_normalized_mean():
    vector = document_vector([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    np.testing.assert_allclose(vector, [2**-0.5, 2**-0.5, 0.0])
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_document_vector_needs_at_least_one_chunk():
    with pytest.raises(ValueError):
        document_vector([])


def test_closest_centroid_wins_and_scores_sum_to_one():
    known = centroids(Tecnico=[1, 0, 0], Juridico=[0, 1, 0], Qualidade=[0, 0, 1])

    result = known.score(document_vector([[0.9, 0.3, 0.1]]), temperature=0.05)

    assert result.category == "Tecnico"
    assert result.confidence == result.scores["Tecnico"] == max(result.scores.values())
    assert sum(result.scores.values()) == pytest.approx(1.0)
    assert set(result.scores) == {"Tecnico", "Juridico", "Qualidade"}


def test_lower_temperature_sharpens_the_confidence():
    known = centroids(Tecnico=[1, 0, 0], Juridico=[0.9, 0.44, 0])
    document = document_vector([[1, 0.1, 0]])

    plain = known.score(document, temperature=1.0)
    sharp = known.score(document, temperature=0.05)

    assert plain.category == sharp.category == "Tecnico"
    assert plain.confidence < 0.55 < sharp.confidence


def test_softmax_does_not_overflow_on_tiny_temperatures():
    known = centroids(Tecnico=[1, 0, 0], Juridico=[0, 1, 0])

    result = known.score(document_vector([[1, 0, 0]]), temperature=1e-6)

    assert result.confidence == pytest.approx(1.0)
    assert all(np.isfinite(list(result.scores.values())))


def test_vector_with_another_size_is_refused():
    known = centroids(Tecnico=[1, 0, 0])

    with pytest.raises(ValueError, match="dimensions"):
        known.score(np.ones(4) / 2, temperature=0.05)


def test_documents_of_the_same_label_are_averaged():
    known = build_centroids(["A", "B", "A"], [np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, 1.0, 0])])

    assert known.labels == ("A", "B")
    np.testing.assert_allclose(known.vectors[0], [2**-0.5, 2**-0.5, 0])


def test_sample_keeps_short_documents_whole():
    assert sample_chunks(["a", "b"], 20) == ["a", "b"]


def test_sample_spreads_across_long_documents():
    chunks = [str(index) for index in range(100)]

    sampled = sample_chunks(chunks, 20)

    assert len(sampled) == 20 == len(set(sampled))
    assert sampled[0] == "0"
    assert int(sampled[-1]) >= 90


def test_saved_centroids_load_back(tmp_path: Path):
    path = tmp_path / "nested" / "centroids.npz"
    save_centroids(centroids(Tecnico=[3, 4, 0], Juridico=[0, 0, 2]), path)

    loaded = load_centroids(path)

    assert loaded is not None
    assert loaded.labels == ("Juridico", "Tecnico")
    np.testing.assert_allclose(loaded.vectors, [[0, 0, 1], [0.6, 0.8, 0]], atol=1e-6)


def test_missing_file_means_no_centroids(tmp_path: Path):
    assert load_centroids(tmp_path / "nao-existe.npz") is None


def test_file_that_is_not_npz_is_refused(tmp_path: Path):
    path = tmp_path / "centroids.npz"
    path.write_bytes(b"lixo")

    with pytest.raises(ValueError, match="centroids"):
        load_centroids(path)


@pytest.mark.parametrize(
    "arrays",
    [
        {"labels": np.array(["A"])},
        {"labels": np.array(["A", "B"]), "vectors": np.ones((3, 3))},
        {"labels": np.array(["A", "A"]), "vectors": np.ones((2, 3))},
        {"labels": np.array(["A"]), "vectors": np.zeros((1, 3))},
        {"labels": np.array(["A"]), "vectors": np.ones(3)},
    ],
    ids=["no-vectors", "count-mismatch", "repeated-labels", "zero-vector", "not-a-matrix"],
)
def test_malformed_file_is_refused(tmp_path: Path, arrays: dict[str, np.ndarray]):
    path = tmp_path / "centroids.npz"
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="centroids"):
        load_centroids(path)
