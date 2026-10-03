from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Provenance:
    tool: str
    version: str
    python: str
    platform: str
    source_file: str
    source_sha256: str
    parameters: dict[str, Any]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_provenance(source_file: str | Path, *, version: str, parameters: dict[str, Any] | None = None) -> Provenance:
    return Provenance(
        tool="Virelion-MyoTrace",
        version=version,
        python=sys.version.split()[0],
        platform=platform.platform(),
        source_file=str(source_file),
        source_sha256=sha256_file(source_file),
        parameters=parameters or {},
    )


def write_json(provenance: Provenance, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(provenance), indent=2, sort_keys=True), encoding="utf-8")
