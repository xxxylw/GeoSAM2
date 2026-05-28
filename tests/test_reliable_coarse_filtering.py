import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from geosam2.pipeline import segment_glb


class SequenceRunner:
    def __init__(self, labels):
        self.labels = np.array(labels, dtype=np.int32)

    def infer_face_labels(self, request):
        return self.labels


def make_many_face_glb(path: Path) -> int:
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.export(path)
    return len(mesh.faces)


class ReliableCoarseFilteringBehaviorTests(unittest.TestCase):
    def test_small_candidate_part_is_not_emitted_as_independent_part(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            output_dir = root / "out"
            face_count = make_many_face_glb(input_path)
            small_count = 2
            first_count = (face_count - small_count) // 2
            second_count = face_count - small_count - first_count
            labels = [1] * first_count + [2] * second_count + [3] * small_count

            segment_glb(
                input_path=input_path,
                output_dir=output_dir,
                label_runner=SequenceRunner(labels),
            )

            manifest = json.loads((output_dir / "labels" / "part_manifest.json").read_text())
            self.assertEqual(manifest["status"], "segmented")
            self.assertEqual([part["label"] for part in manifest["parts"]], [1, 2])

            saved_labels = np.load(output_dir / "labels" / "face_labels.npy")
            self.assertNotIn(3, saved_labels.tolist())

    def test_insufficient_label_coverage_falls_back_to_unsegmented_part(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            output_dir = root / "out"
            face_count = make_many_face_glb(input_path)
            labels = [1] * 30 + [2] * 30 + [0] * (face_count - 60)

            result = segment_glb(
                input_path=input_path,
                output_dir=output_dir,
                label_runner=SequenceRunner(labels),
            )

            self.assertEqual(result.status, "unsegmented")
            manifest = json.loads((output_dir / "labels" / "part_manifest.json").read_text())
            self.assertEqual(manifest["status"], "unsegmented")
            self.assertEqual(len(manifest["parts"]), 1)
            self.assertEqual(manifest["parts"][0]["status"], "unsegmented")


if __name__ == "__main__":
    unittest.main()
