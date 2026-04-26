"""SHA-256 helpers used to fingerprint artifacts in the manifest.

Existing nodes computed sha256 inline in many places (xtb, marc, orca,
prism). Centralizing here so artifact records always come from the same
implementation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE: int = 1024 * 1024  # 1 MiB — matches existing node code


def sha256_file(path: str | Path, *, chunk_size: int = _CHUNK_SIZE) -> str:
    """Return the hex SHA-256 digest of ``path`` (file contents).

    Streams in 1 MiB chunks to avoid loading large ORCA outputs / xyz
    ensembles fully into memory.
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    """Convenience: SHA-256 of an in-memory string (UTF-8 encoded)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
