"""Small hashing helpers shared by benchmark and analysis code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA256 digest of an in-memory payload."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a file SHA256 without loading the entire file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Hash a stable UTF-8 JSON representation of ``value``."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)
