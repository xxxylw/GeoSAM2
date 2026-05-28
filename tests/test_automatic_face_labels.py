import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from geosam2.pipeline import segment_glb


class StaticAutomaticRunner:
    def infer_face_labels(self, request):
        return np.array([1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2], dtype=np.int32)


def make_box_glb(path: Path) -> None:
    trimesh.creation.box().export(path)


class AutomaticFaceLabelBehaviorTests(unittest.TestCase):
    def test_automatic_mode_saves_runner_face_labels_and_manifest_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            output_dir = root / "out"
            make_box_glb(input_path)

            result = segment_glb(
                input_path=input_path,
                output_dir=output_dir,
                label_runner=StaticAutomaticRunner(),
            )

            self.assertEqual(result.status, "segmented")
            labels = np.load(output_dir / "labels" / "face_labels.npy")
            self.assertEqual(labels.tolist(), [1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2])

            manifest = json.loads((output_dir / "labels" / "part_manifest.json").read_text())
            self.assertEqual(manifest["status"], "segmented")
            self.assertEqual(len(manifest["parts"]), 2)
            self.assertEqual(manifest["parts"][0]["label"], 1)
            self.assertEqual(manifest["parts"][0]["face_count"], 6)
            self.assertEqual(manifest["parts"][1]["label"], 2)
            self.assertEqual(manifest["parts"][1]["face_count"], 6)


if __name__ == "__main__":
    unittest.main()
