"""Default GeoSAM2 face-label runner used by the public GLB API."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .pipeline import LabelRequest


class GeoSAM2RunnerError(RuntimeError):
    """Raised when the external GeoSAM2 render/inference runner fails."""


class UnsegmentedFaceLabelRunner:
    """Offline runner for tests and explicit fallback runs."""

    def infer_face_labels(self, request: "LabelRequest") -> np.ndarray:
        import trimesh

        loaded = trimesh.load(request.input_path, force="scene", process=False)
        if isinstance(loaded, trimesh.Scene):
            face_count = sum(len(geometry.faces) for geometry in loaded.geometry.values())
        else:
            face_count = len(loaded.faces)
        return np.ones((int(face_count),), dtype=np.int32)


class GeoSAM2FaceLabelRunner:
    """Run Blender rendering plus GeoSAM2 inference and return per-face labels."""

    def __init__(
        self,
        *,
        blender_executable: str | os.PathLike[str] | None = None,
        python_executable: str | os.PathLike[str] | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        model_cfg: str | os.PathLike[str] | None = None,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.blender_executable = str(blender_executable or self._find_blender())
        self.python_executable = str(python_executable or sys.executable)
        self.checkpoint_path = Path(checkpoint_path or self.repo_root / "ckpt" / "geosam2.pt")
        self.model_cfg = str(model_cfg or "configs/geosam2.yaml")

    def infer_face_labels(self, request: "LabelRequest") -> np.ndarray:
        if request.points_path is not None:
            raise GeoSAM2RunnerError("Point prompts are not supported by the GeoSAM2 runner yet; use mask prompts")

        render_cache_dir = Path(request.render_cache_dir)
        render_dir = render_cache_dir / "renders"
        inference_dir = render_cache_dir / self._inference_cache_name(request)
        log_dir = render_cache_dir / "logs"
        render_dir.mkdir(parents=True, exist_ok=True)
        inference_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        if not self._render_cache_complete(render_dir):
            self._run_render(request, render_dir, log_dir)
        self._validate_render_cache(render_dir)

        label_path = self._latest_label_path(inference_dir)
        if label_path is None:
            self._run_inference(request, render_dir, inference_dir, log_dir)
            label_path = self._latest_label_path(inference_dir)
        if label_path is None:
            raise GeoSAM2RunnerError(f"GeoSAM2 inference did not write face labels in {inference_dir}")

        return np.load(label_path).astype(np.int32).reshape(-1)

    def _run_render(self, request: "LabelRequest", render_dir: Path, log_dir: Path) -> None:
        command = [
            self.blender_executable,
            "-b",
            "-P",
            str(self.repo_root / "geosam2_render.py"),
            request.input_path,
            "glb",
            str(render_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self._write_process_log(log_dir / "render.log", command, completed)
        if completed.returncode != 0 and not self._render_cache_complete(render_dir):
            raise GeoSAM2RunnerError(
                f"Blender render failed with exit code {completed.returncode}; see {log_dir / 'render.log'}"
            )

    def _run_inference(
        self,
        request: "LabelRequest",
        render_dir: Path,
        inference_dir: Path,
        log_dir: Path,
    ) -> None:
        command = [
            self.python_executable,
            "inference.py",
            "--data-root",
            str(render_dir),
            "--output-dir",
            str(inference_dir),
            "--sam2-checkpoint",
            str(self.checkpoint_path),
            "--model-cfg",
            self.model_cfg,
        ]
        if request.mask_path is not None:
            command.extend(["--mask-path", request.mask_path, "--mask-view", str(request.mask_view)])

        completed = self._run_inference_command(
            command,
            log_dir / "inference.log",
            gpu=request.gpu,
        )
        if completed.returncode == 0:
            return

        retry_command = command + ["--no-enable-postprocess"]
        retry_completed = self._run_inference_command(
            retry_command,
            log_dir / "inference_no_postprocess.log",
            gpu=request.gpu,
        )
        if retry_completed.returncode != 0:
            raise GeoSAM2RunnerError(
                f"GeoSAM2 inference failed with exit code {retry_completed.returncode}; "
                f"see {log_dir / 'inference_no_postprocess.log'}"
            )

    def _run_inference_command(
        self,
        command: list[str],
        log_path: Path,
        gpu: int,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self._write_process_log(log_path, command, completed)
        return completed

    def _inference_cache_name(self, request: "LabelRequest") -> str:
        if request.mask_path is None:
            return "inference"
        mask_stem = Path(request.mask_path).stem
        return f"inference_mask_view_{int(request.mask_view)}_{mask_stem}"

    def _render_cache_complete(self, render_dir: Path) -> bool:
        return all(path.exists() for path in self._required_render_files(render_dir))

    def _validate_render_cache(self, render_dir: Path) -> None:
        missing = [str(path) for path in self._required_render_files(render_dir) if not path.exists()]
        if missing:
            raise GeoSAM2RunnerError("Render cache is incomplete: " + ", ".join(missing[:5]))

    def _required_render_files(self, render_dir: Path) -> list[Path]:
        files = [render_dir / "mesh.glb", render_dir / "meta.json"]
        for idx in range(12):
            stem = f"{idx:04d}"
            files.extend(
                [
                    render_dir / f"color_{stem}.webp",
                    render_dir / f"depth_{stem}.exr",
                    render_dir / f"normal_{stem}.webp",
                ]
            )
        return files

    def _latest_label_path(self, inference_dir: Path) -> Path | None:
        candidates = list(inference_dir.glob("*.npy"))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _write_process_log(
        self,
        path: Path,
        command: list[str],
        completed: subprocess.CompletedProcess[str],
    ) -> None:
        path.write_text(
            "COMMAND: " + " ".join(command) + "\n"
            + f"EXIT: {completed.returncode}\n\n"
            + "STDOUT:\n"
            + (completed.stdout or "")
            + "\nSTDERR:\n"
            + (completed.stderr or "")
        )

    def _find_blender(self) -> str:
        env_path = os.environ.get("GEOSAM2_BLENDER")
        if env_path:
            return env_path

        path_blender = shutil.which("blender")
        if path_blender:
            return path_blender

        known_paths = [
            self.repo_root.parent / "SAMPart3D" / "blender-4.0.0-linux-x64" / "blender",
            self.repo_root.parent
            / "PartPipeline"
            / "third_party"
            / "SAMPart3D"
            / "blender-4.0.0-linux-x64"
            / "blender",
        ]
        for candidate in known_paths:
            if candidate.exists():
                return str(candidate)

        raise GeoSAM2RunnerError(
            "Blender executable not found. Set GEOSAM2_BLENDER or install blender on PATH."
        )
