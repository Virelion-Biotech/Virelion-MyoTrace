"""Atomic, checksum-verified acquisition of pinned public validation inputs."""

from __future__ import annotations

import hashlib
from http.client import IncompleteRead
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def verified_source(source: dict, destination: Path, *, attempts: int = 3) -> dict:
    """Reuse valid bytes, or replace only after a fresh download verifies.

    A failed transfer never installs partial bytes or changes source pins.
    Integrity mismatches fail closed and retain expected/observed fingerprints.
    """
    if not 1 <= attempts <= 5:
        raise ValueError("attempts must be between 1 and 5")
    expected_hash = source["sha256"]
    expected_size = source["size"]
    destination = Path(destination)
    if destination.is_file():
        size, digest = _fingerprint(destination)
        if (size, digest) == (expected_size, expected_hash):
            return {"sha256": digest, "size": size, "reused": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False
            ) as handle:
                temporary = Path(handle.name)
                request = urllib.request.Request(source["url"], headers={"User-Agent": "MyoTrace-public-input-audit"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(chunk)
            size, digest = _fingerprint(temporary)
            if (size, digest) != (expected_size, expected_hash):
                raise ValueError(
                    f"Source integrity mismatch for {destination.name}: expected "
                    f"{expected_size} bytes / sha256 {expected_hash}; observed {size} bytes / sha256 {digest}"
                )
            os.replace(temporary, destination)
            return {"sha256": digest, "size": size, "reused": False}
        except (urllib.error.URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                raise
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError("Public-source acquisition exhausted without a result")  # pragma: no cover
