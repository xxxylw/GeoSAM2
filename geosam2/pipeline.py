"""High-level GLB segmentation pipeline.

This module owns the public API used by both Python callers and the CLI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import trimesh

from .cache import compute_render_cache_key, ensure_render_cache_dir
from .exporter import export_part_object_glb, load_source_meshes
from .pipeline_errors import SegmentExportError


SegmentationMode = Literal["automatic", "prompted"]


class SegmentError(RuntimeError):
    """Raised when a segmentation run cannot produce a valid result."""


@dataclass(frozen=True)
class SegmentResult:
    status: str
    mode: SegmentationMode
    output_glb: str
    face_labels_path: str
    manifest_path: str
    warnings_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelRequest:
    input_path: str
    mode: SegmentationMode
    render_cache_dir: str
    gpu: int
    mask_path: str | None = None
    mask_view: int | None = None
    points_path: str | None = None
    prompt_view: int | None = None


class FaceLabelRunner(Protocol):
    def infer_face_labels(self, request: LabelRequest) -> np.ndarray:
        """Return one integer label per input face."""


def segment_glb(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    mode: SegmentationMode = "automatic",
    cache_dir: str | Path = ".cache/geosam2",
    gpu: int | None = None,
    strict_pbr: bool = True,
    write_preview: bool = False,
    mask_path: str | Path | None = None,
    mask_view: int | None = None,
    points_path: str | Path | None = None,
    prompt_view: int | None = None,
    label_runner: FaceLabelRunner | None = None,
) -> SegmentResult:
    """Segment a GLB into a Part-object GLB.

    The first implementation establishes the public contract and produces a
    Strict PBR-preserving Unsegmented Part by copying the source GLB.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)

    if mode not in {"automatic", "prompted"}:
        raise SegmentError(f"Unsupported segmentation mode: {mode}")
    if not input_path.exists():
        raise SegmentError(f"Input GLB does not exist: {input_path}")
    if input_path.suffix.lower() != ".glb":
        raise SegmentError(f"Input must be a .glb file: {input_path}")
    if mode == "prompted" and not (mask_path or points_path):
        raise SegmentError("Prompt-driven Part Segmentation requires a Mask Prompt or Point Prompt")
    if mask_path is not None and mask_view is None:
        raise SegmentError("mask_view is required when mask_path is provided")
    if points_path is not None and prompt_view is None:
        raise SegmentError("prompt_view is required when points_path is provided")

    labels_dir = output_dir / "labels"
    preview_dir = output_dir / "preview"
    debug_dir = output_dir / "debug"
    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    render_settings = {"num_views": 12, "resolution": 1024}
    render_cache_key = compute_render_cache_key(
        input_path=input_path,
        render_settings=render_settings,
    )
    render_cache_dir = ensure_render_cache_dir(cache_dir, render_cache_key)

    face_count = _count_faces(input_path)
    if gpu is None:
        selected_gpu = 0
    else:
        selected_gpu = int(gpu)

    request = LabelRequest(
        input_path=str(input_path),
        mode=mode,
        render_cache_dir=str(render_cache_dir),
        gpu=selected_gpu,
        mask_path=str(mask_path) if mask_path is not None else None,
        mask_view=mask_view,
        points_path=str(points_path) if points_path is not None else None,
        prompt_view=prompt_view,
    )
    if label_runner is None:
        label_runner = create_default_label_runner()

    try:
        face_labels = np.asarray(label_runner.infer_face_labels(request), dtype=np.int32).reshape(-1)
    except Exception as exc:
        if isinstance(exc, SegmentError):
            raise
        raise SegmentError(str(exc)) from exc
    if len(face_labels) != face_count:
        raise SegmentError(
            f"Face label count mismatch: expected {face_count}, got {len(face_labels)}"
        )

    face_labels, status, filter_summary = _apply_reliable_coarse_filter(face_labels)
    face_labels_path = labels_dir / "face_labels.npy"
    np.save(face_labels_path, face_labels)

    output_glb = output_dir / "segmented_parts.glb"
    try:
        if status == "segmented":
            export_part_object_glb(
                input_path=input_path,
                face_labels=face_labels,
                output_path=output_glb,
                strict_pbr=strict_pbr,
                source_name=input_path.stem,
            )
        else:
            shutil.copyfile(input_path, output_glb)
    except SegmentExportError as exc:
        if output_glb.exists():
            output_glb.unlink()
        raise SegmentError(str(exc)) from exc

    warnings: list[dict[str, str]] = []
    preview_manifest: dict[str, str | None] = {"segmentation_colored_glb": None}
    if write_preview:
        preview_glb = preview_dir / "segmentation_colored.glb"
        shutil.copyfile(output_glb, preview_glb)
        preview_manifest["segmentation_colored_glb"] = "preview/segmentation_colored.glb"
        warnings.append(
            {
                "code": "preview_not_successful_output",
                "message": "Visualization Colours preview is an optional artifact and is not Successful Output.",
            }
        )
    source_ranges = _source_object_ranges(input_path)
    parts = _build_parts(labels=face_labels, status=status, source_ranges=source_ranges)

    manifest = {
        "status": status,
        "mode": mode,
        "input": {
            "path": str(input_path),
            "face_count": face_count,
        },
        "run": {
            "cache_dir": str(cache_dir),
            "render_cache_dir": str(render_cache_dir),
            "render_cache_key": render_cache_key,
            "gpu": selected_gpu,
            "strict_pbr": strict_pbr,
            "write_preview": write_preview,
        },
        "successful_output": "segmented_parts.glb",
        "preview": preview_manifest,
        "filtering": filter_summary,
        "parts": parts,
    }

    manifest_path = labels_dir / "part_manifest.json"
    warnings_path = debug_dir / "warnings.json"
    render_cache_key_path = debug_dir / "render_cache_key.txt"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    warnings_path.write_text(json.dumps(warnings, indent=2) + "\n")
    render_cache_key_path.write_text(render_cache_key + "\n")

    return SegmentResult(
        status=status,
        mode=mode,
        output_glb=str(output_glb),
        face_labels_path=str(face_labels_path),
        manifest_path=str(manifest_path),
        warnings_path=str(warnings_path),
    )


def create_default_label_runner() -> FaceLabelRunner:
    """Create the production label runner.

    Tests and explicitly offline smoke runs can set
    ``GEOSAM2_DEFAULT_RUNNER=unsegmented`` to avoid invoking Blender/GPU.
    """
    from .geosam2_runner import GeoSAM2FaceLabelRunner, UnsegmentedFaceLabelRunner

    if os.environ.get("GEOSAM2_DEFAULT_RUNNER") == "unsegmented":
        return UnsegmentedFaceLabelRunner()
    return GeoSAM2FaceLabelRunner()


def _count_faces(input_path: Path) -> int:
    loaded = trimesh.load(input_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        return int(sum(len(geometry.faces) for geometry in loaded.geometry.values()))
    return int(len(loaded.faces))


def _source_object_ranges(input_path: Path) -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    offset = 0
    for source_name, mesh in load_source_meshes(input_path, fallback_name=input_path.stem):
        next_offset = offset + len(mesh.faces)
        ranges.append((source_name, offset, next_offset))
        offset = next_offset
    return ranges


def _build_parts(
    *,
    labels: np.ndarray,
    status: str,
    source_ranges: list[tuple[str, int, int]],
) -> list[dict[str, Any]]:
    face_count = int(len(labels))
    parts: list[dict[str, Any]] = []
    for source_name, start, end in source_ranges:
        source_labels = labels[start:end]
        source_face_count = int(len(source_labels))
        for label in sorted(np.unique(source_labels).tolist()):
            label_int = int(label)
            if label_int <= 0:
                continue
            count = int(np.sum(source_labels == label_int))
            part_status = "unsegmented" if status == "unsegmented" else "segmented"
            suffix = "unsegmented" if part_status == "unsegmented" else f"part_{label_int:03d}"
            parts.append(
                {
                    "name": f"{source_name}_{suffix}",
                    "source_object": source_name,
                    "label": label_int,
                    "face_count": count,
                    "area_ratio": count / source_face_count if source_face_count else 0.0,
                    "status": part_status,
                }
            )
    if not parts:
        fallback_name = source_ranges[0][0] if source_ranges else "asset"
        parts.append(
            {
                "name": f"{fallback_name}_unsegmented",
                "source_object": fallback_name,
                "label": 1,
                "face_count": face_count,
                "area_ratio": 1.0,
                "status": "unsegmented",
            }
        )
    return parts


def _apply_reliable_coarse_filter(
    labels: np.ndarray,
    *,
    min_area_ratio: float = 0.01,
    min_coverage: float = 0.95,
    min_fillable_coverage: float = 0.25,
    max_parts: int = 12,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    total = int(len(labels))
    if total == 0:
        return labels, "unsegmented", {
            "coverage": 0.0,
            "reliable_labels": [],
            "filtered_labels": [],
            "reason": "empty_mesh",
        }

    counts = {
        int(label): int(np.sum(labels == label))
        for label in np.unique(labels)
        if int(label) > 0
    }
    sorted_labels = sorted(counts, key=lambda label: counts[label], reverse=True)
    reliable = [
        label
        for label in sorted_labels
        if counts[label] / total >= min_area_ratio
    ][:max_parts]
    reliable_count = int(sum(counts[label] for label in reliable))
    coverage = reliable_count / total

    summary = {
        "coverage": coverage,
        "reliable_labels": reliable,
        "filtered_labels": [label for label in sorted_labels if label not in reliable],
        "min_area_ratio": min_area_ratio,
        "min_coverage": min_coverage,
        "min_fillable_coverage": min_fillable_coverage,
        "max_parts": max_parts,
        "reason": None,
    }

    if len(reliable) < 2:
        summary["reason"] = "no_reliable_coarse_parts"
        return np.ones((total,), dtype=np.int32), "unsegmented", summary
    if coverage < min_coverage and coverage < min_fillable_coverage:
        summary["reason"] = "insufficient_label_coverage"
        return np.ones((total,), dtype=np.int32), "unsegmented", summary

    filtered = np.zeros((total,), dtype=np.int32)
    for new_label, old_label in enumerate(sorted(reliable), start=1):
        filtered[labels == old_label] = new_label
    if coverage < min_coverage:
        dominant_old_label = max(reliable, key=lambda label: counts[label])
        dominant_new_label = sorted(reliable).index(dominant_old_label) + 1
        filtered[filtered == 0] = dominant_new_label
        summary["reason"] = "filled_unreliable_labels"
    summary["reliable_labels"] = sorted(reliable)
    return filtered, "segmented", summary
