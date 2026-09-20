"""Settings read from environment variables.

Everything tunable lives here so the defaults are visible in one place and each
module gets a plain constant instead of scattering ``os.getenv`` calls around. The
values are read once, at import time: changing them means restarting the process.
"""

from __future__ import annotations

import os

#: Fallback used when ``EMBEDDING_MODEL`` is unset or empty.
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

#: Fallback used when ``MAX_UPLOAD_BYTES`` is unset: 25 MiB.
DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024

#: Fallback used when ``MAX_QUERY_CHARS`` is unset.
DEFAULT_MAX_QUERY_CHARS = 1000


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


#: Sentence-transformers model loaded by the real embedder.
#:
#: ``chunking.py``'s default sizes assume the 512-token window of the e5 models; a
#: model with a smaller window also needs a smaller ``chunk_size``.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL

#: Largest upload the processing endpoint accepts, in bytes. Anything past it is
#: refused with ``413`` instead of being read into memory.
MAX_UPLOAD_BYTES = _positive_int("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)


#: Longest search text ``POST /api/v1/embeddings/query`` accepts, in characters.
#:
#: Matches ``chunking.py``'s default ``chunk_size`` on purpose: a query is compared
#: against chunks, so there is no point accepting one that could not have been a chunk.
#: The limit also keeps the input inside the model's window -- sentence-transformers
#: would otherwise truncate it silently and answer with a vector of a text nobody sent.
MAX_QUERY_CHARS = _positive_int("MAX_QUERY_CHARS", DEFAULT_MAX_QUERY_CHARS)
