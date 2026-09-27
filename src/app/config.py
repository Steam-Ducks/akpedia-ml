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
