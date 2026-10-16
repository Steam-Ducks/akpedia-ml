"""Settings read from environment variables.

Everything tunable lives here so the defaults are visible in one place and each
module gets a plain constant instead of scattering ``os.getenv`` calls around. The
values are read once, at import time: changing them means restarting the process.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repository root (``src/app/config.py`` -> two levels up from ``src``). Relative
#: paths below are anchored here so they do not depend on the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Fallback used when ``EMBEDDING_MODEL`` is unset or empty.
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

#: Fallback used when ``MAX_UPLOAD_BYTES`` is unset: 25 MiB.
DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024

#: Fallback used when ``CENTROIDS_PATH`` is unset, relative to :data:`PROJECT_ROOT`.
DEFAULT_CENTROIDS_PATH = "data/categories/centroids.npz"

#: Fallback used when ``ANALYSIS_MAX_CHUNKS`` is unset.
DEFAULT_ANALYSIS_MAX_CHUNKS = 20

#: Fallback used when ``CATEGORY_TEMPERATURE`` is unset.
DEFAULT_CATEGORY_TEMPERATURE = 0.05


def _positive_int(name: str, default: int) -> int:
    """Read a positive integer from the environment, failing loudly on a bad value.

    A typo in a limit is worth a crash at startup: the alternative is silently
    falling back to the default and only finding out in production.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got '{raw}'.") from None
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}.")
    return value


def _positive_float(name: str, default: float) -> float:
    """Read a positive number from the environment; same contract as :func:`_positive_int`."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got '{raw}'.") from None
    if not value > 0 or value == float("inf"):
        raise ValueError(f"{name} must be a finite number greater than zero, got {raw}.")
    return value


def _project_path(name: str, default: str) -> Path:
    """Read a path from the environment, resolving relative ones against :data:`PROJECT_ROOT`."""
    path = Path(os.getenv(name, "").strip() or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


#: Sentence-transformers model loaded by the real embedder.
#:
#: ``chunking.py``'s default sizes assume the 512-token window of the e5 models; a
#: model with a smaller window also needs a smaller ``chunk_size``.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL

#: Largest upload the processing endpoint accepts, in bytes. Anything past it is
#: refused with ``413`` instead of being read into memory.
MAX_UPLOAD_BYTES = _positive_int("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)

#: Category centroids built by ``scripts/build_centroids.py``. When the file does not
#: exist the analysis endpoint still answers, with ``category: null``.
CENTROIDS_PATH = _project_path("CENTROIDS_PATH", DEFAULT_CENTROIDS_PATH)

#: Most chunks embedded to classify one document. Caps the cost of a long document;
#: the chunks are sampled evenly across it, not taken from the start.
ANALYSIS_MAX_CHUNKS = _positive_int("ANALYSIS_MAX_CHUNKS", DEFAULT_ANALYSIS_MAX_CHUNKS)

#: Softmax temperature applied to the cosine similarities against each centroid.
#:
#: Similarities between a document and every centroid sit close together (a clear
#: winner may lead by 0.05), so a plain softmax (``1.0``) spreads the probability
#: almost evenly. Dividing by a small temperature turns that lead into a confidence
#: that reads like one: with ``0.05`` a 0.05 lead is worth a factor of e (~2.7).
CATEGORY_TEMPERATURE = _positive_float("CATEGORY_TEMPERATURE", DEFAULT_CATEGORY_TEMPERATURE)
