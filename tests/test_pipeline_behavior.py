import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import trimesh

from geosam2.pipeline import SegmentError, segment_glb


def make_box_glb(path: Path) -> None:
    mesh = trimesh.creation.box()
    mesh.export(path)


class SegmentGlbBehaviorTests(unittest.TestCase):
    def test_automatic_run_writes_standard_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            output_dir = root / "out"
            make_box_glb(input_path)

            result = segment_glb(input_path=input_path, output_dir=output_dir)

            self.assertEqual(result.status, "unsegmented")
            self.assertEqual(result.mode, "automatic")
            self.assertTrue((output_dir / "segmented_parts.glb").exists())
            self.assertTrue((output_dir / "labels" / "face_labels.npy").exists())
            self.assertTrue((output_dir / "labels" / "part_manifest.json").exists())
            self.assertTrue((output_dir / "debug" / "warnings.json").exists())

            manifest = json.loads((output_dir / "labels" / "part_manifest.json").read_text())
            self.assertEqual(manifest["mode"], "automatic")
            self.assertEqual(manifest["status"], "unsegmented")
            self.assertEqual(manifest["parts"][0]["name"], "asset_unsegmented")

    def test_invalid_input_fails_before_writing_successful_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"

            with self.assertRaises(SegmentError):
                segment_glb(input_path=root / "missing.glb", output_dir=output_dir)

            self.assertFalse((output_dir / "segmented_parts.glb").exists())

    def test_cli_uses_same_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            output_dir = root / "out"
            make_box_glb(input_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    "segment_glb.py",
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["status"], "unsegmented")
            self.assertTrue((output_dir / "labels" / "part_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
