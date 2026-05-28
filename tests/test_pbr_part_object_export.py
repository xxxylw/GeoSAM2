import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from geosam2.pipeline import segment_glb


class TwoPartRunner:
    def infer_face_labels(self, request):
        return np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], dtype=np.int32)


def make_box_glb(path: Path) -> None:
    mesh = trimesh.creation.box()
    mesh.visual.vertex_colors = np.tile(np.array([[200, 100, 50, 255]], dtype=np.uint8), (len(mesh.vertices), 1))
    mesh.export(path)


class PbrPartObjectExportBehaviorTests(unittest.TestCase):
    def test_segmented_single_object_exports_multiple_part_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            output_dir = root / "out"
            make_box_glb(input_path)

            segment_glb(
                input_path=input_path,
                output_dir=output_dir,
                label_runner=TwoPartRunner(),
            )

            exported = trimesh.load(output_dir / "segmented_parts.glb", force="scene", process=False)
            self.assertIsInstance(exported, trimesh.Scene)
            self.assertEqual(len(exported.geometry), 2)
            self.assertEqual(sorted(exported.geometry.keys()), ["asset_part_001", "asset_part_002"])


if __name__ == "__main__":
    unittest.main()
