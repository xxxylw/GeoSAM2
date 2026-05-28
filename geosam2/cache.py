"""Render cache keying for GLB segmentation runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_render_cache_key(
    *,
    input_path: str | Path,
    render_settings: dict[str, Any] | None = None,
    render_script_path: str | Path = "geosam2_render.py",
) -> str:
    input_path = Path(input_path)
    render_script_path = Path(render_script_path)
    settings = render_settings or {}

    digest = hashlib.sha256()
    digest.update(b"geosam2-render-cache-v1\0")
    digest.update(_file_sha256(input_path).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    if render_script_path.exists():
        digest.update(_file_sha256(render_script_path).encode("ascii"))
    else:
        digest.update(str(render_script_path).encode("utf-8"))
    return digest.hexdigest()[:32]


def ensure_render_cache_dir(cache_dir: str | Path, cache_key: str) -> Path:
    path = Path(cache_dir) / cache_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
