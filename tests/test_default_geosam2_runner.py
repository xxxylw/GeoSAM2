import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from geosam2.geosam2_runner import GeoSAM2FaceLabelRunner
from geosam2.pipeline import LabelRequest


def touch_required_render_files(render_dir: Path) -> None:
    render_dir.mkdir(parents=True, exist_ok=True)
    (render_dir / "mesh.glb").write_bytes(b"glb")
    (render_dir / "meta.json").write_text("{}")
    for idx in range(12):
        stem = f"{idx:04d}"
        (render_dir / f"color_{stem}.webp").write_bytes(b"color")
        (render_dir / f"depth_{stem}.exr").write_bytes(b"depth")
        (render_dir / f"normal_{stem}.webp").write_bytes(b"normal")


class GeoSAM2FaceLabelRunnerTests(unittest.TestCase):
    def test_runner_invokes_render_and_inference_then_loads_face_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            input_path.write_bytes(b"glb")
            render_cache_dir = root / "cache"
            labels = np.array([1, 1, 2, 2], dtype=np.int32)

            def fake_run(command, **kwargs):
                if command[0] == "/opt/blender":
                    touch_required_render_files(render_cache_dir / "renders")
                else:
                    np.save(render_cache_dir / "inference" / "segmentation_result.npy", labels)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            runner = GeoSAM2FaceLabelRunner(
                blender_executable="/opt/blender",
                python_executable=sys.executable,
                checkpoint_path=root / "geosam2.pt",
            )

            request = LabelRequest(
                input_path=str(input_path),
                mode="automatic",
                render_cache_dir=str(render_cache_dir),
                gpu=2,
            )

            with patch("subprocess.run", side_effect=fake_run) as run_mock:
                actual = runner.infer_face_labels(request)

            self.assertEqual(actual.tolist(), labels.tolist())
            render_command = run_mock.call_args_list[0].args[0]
            inference_command = run_mock.call_args_list[1].args[0]
            inference_env = run_mock.call_args_list[1].kwargs["env"]

            self.assertEqual(render_command[0], "/opt/blender")
            self.assertIn("geosam2_render.py", render_command[3])
            self.assertEqual(inference_command[:2], [sys.executable, "inference.py"])
            self.assertEqual(inference_env["CUDA_VISIBLE_DEVICES"], "2")

    def test_runner_reuses_complete_render_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            input_path.write_bytes(b"glb")
            render_cache_dir = root / "cache"
            touch_required_render_files(render_cache_dir / "renders")
            labels = np.array([1, 2, 2, 1], dtype=np.int32)

            def fake_run(command, **kwargs):
                self.assertNotEqual(command[0], "/opt/blender")
                np.save(render_cache_dir / "inference" / "segmentation_result.npy", labels)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            runner = GeoSAM2FaceLabelRunner(
                blender_executable="/opt/blender",
                python_executable=sys.executable,
                checkpoint_path=root / "geosam2.pt",
            )

            request = LabelRequest(
                input_path=str(input_path),
                mode="automatic",
                render_cache_dir=str(render_cache_dir),
                gpu=0,
            )

            with patch("subprocess.run", side_effect=fake_run):
                actual = runner.infer_face_labels(request)

            self.assertEqual(actual.tolist(), labels.tolist())

    def test_runner_retries_without_postprocess_when_inference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            input_path.write_bytes(b"glb")
            render_cache_dir = root / "cache"
            touch_required_render_files(render_cache_dir / "renders")
            labels = np.array([1, 2, 3, 3], dtype=np.int32)
            inference_attempts = []

            def fake_run(command, **kwargs):
                inference_attempts.append(command)
                if "--no-enable-postprocess" not in command:
                    return subprocess.CompletedProcess(command, 1, stdout="", stderr="oom")
                np.save(render_cache_dir / "inference" / "segmentation_result.npy", labels)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            runner = GeoSAM2FaceLabelRunner(
                blender_executable="/opt/blender",
                python_executable=sys.executable,
                checkpoint_path=root / "geosam2.pt",
            )
            request = LabelRequest(
                input_path=str(input_path),
                mode="automatic",
                render_cache_dir=str(render_cache_dir),
                gpu=0,
            )

            with patch("subprocess.run", side_effect=fake_run):
                actual = runner.infer_face_labels(request)

            self.assertEqual(actual.tolist(), labels.tolist())
            self.assertEqual(len(inference_attempts), 2)
            self.assertIn("--no-enable-postprocess", inference_attempts[1])


if __name__ == "__main__":
    unittest.main()
