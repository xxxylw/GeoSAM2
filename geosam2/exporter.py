"""Part-object GLB export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .pipeline_errors import SegmentExportError


def export_part_object_glb(
    *,
    input_path: str | Path,
    face_labels: np.ndarray,
    output_path: str | Path,
    strict_pbr: bool = True,
    source_name: str | None = None,
) -> None:
    """Export a single-source-object Part-object GLB from per-face labels."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    source_name = source_name or input_path.stem
    sources = load_source_meshes(input_path, fallback_name=source_name)
    labels = np.asarray(face_labels, dtype=np.int32).reshape(-1)
    total_faces = sum(len(mesh.faces) for _, mesh in sources)
    if len(labels) != total_faces:
        raise SegmentExportError(
            f"Face label count mismatch during export: expected {total_faces}, got {len(labels)}"
        )

    unique_labels = [int(label) for label in sorted(np.unique(labels).tolist()) if int(label) > 0]
    if len(unique_labels) <= 1:
        loaded = trimesh.load(input_path, force="scene", process=False)
        loaded.export(output_path)
        return

    scene = trimesh.Scene()
    offset = 0
    expected_parts = 0
    for source_object, source in sources:
        source_labels = labels[offset: offset + len(source.faces)]
        offset += len(source.faces)
        for label in [int(item) for item in sorted(np.unique(source_labels).tolist()) if int(item) > 0]:
            face_indices = np.flatnonzero(source_labels == label)
            if len(face_indices) == 0:
                continue
            expected_parts += 1
            part_mesh = source.submesh([face_indices], append=True, repair=False)
            part_name = f"{source_object}_part_{label:03d}"
            scene.add_geometry(part_mesh, geom_name=part_name, node_name=part_name)

    if strict_pbr and len(scene.geometry) != expected_parts:
        raise SegmentExportError("Strict PBR export failed to create all part objects")
    scene.export(output_path)


def load_source_meshes(input_path: str | Path, fallback_name: str | None = None) -> list[tuple[str, trimesh.Trimesh]]:
    input_path = Path(input_path)
    loaded = trimesh.load(input_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 1:
            return [(fallback_name or input_path.stem, next(iter(loaded.geometry.values())))]
        return [(str(name), mesh) for name, mesh in loaded.geometry.items()]
    return [(fallback_name or input_path.stem, loaded)]
