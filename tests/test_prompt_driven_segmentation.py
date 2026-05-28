import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from geosam2.pipeline import segment_glb


class PromptRunner:
    def __init__(self):
        self.request = None

    def infer_face_labels(self, request):
        self.request = request
        return np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.int32)


def make_box_glb(path: Path) -> None:
    trimesh.creation.box().export(path)


class PromptDrivenSegmentationBehaviorTests(unittest.TestCase):
    def test_mask_prompt_is_passed_to_label_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            output_dir = root / "out"
            mask_path = root / "mask.png"
            make_box_glb(input_path)
            mask_path.write_bytes(b"placeholder")
            runner = PromptRunner()

            result = segment_glb(
                input_path=input_path,
                output_dir=output_dir,
                mode="prompted",
                mask_path=mask_path,
                mask_view=0,
                label_runner=runner,
            )

            self.assertEqual(result.status, "segmented")
            self.assertEqual(runner.request.mode, "prompted")
            self.assertEqual(runner.request.mask_path, str(mask_path))
            self.assertEqual(runner.request.mask_view, 0)

    def test_cli_accepts_points_prompt_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            output_dir = root / "out"
            points_path = root / "points.json"
            make_box_glb(input_path)
            points_path.write_text("[]")

            completed = subprocess.run(
                [
                    sys.executable,
                    "segment_glb.py",
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--mode",
                    "prompted",
                    "--points",
                    str(points_path),
                    "--prompt-view",
                    "0",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["mode"], "prompted")


if __name__ == "__main__":
    unittest.main()
