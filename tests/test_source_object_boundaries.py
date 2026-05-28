import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from geosam2.pipeline import segment_glb


class RepeatedLabelsRunner:
    def infer_face_labels(self, request):
        return np.array(
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
            + [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            dtype=np.int32,
        )


def make_two_object_glb(path: Path) -> None:
    scene = trimesh.Scene()
    body = trimesh.creation.box()
    wheel = trimesh.creation.box()
    scene.add_geometry(body, geom_name="body", node_name="body")
    scene.add_geometry(wheel, geom_name="wheel", node_name="wheel")
    scene.export(path)


class SourceObjectBoundaryBehaviorTests(unittest.TestCase):
    def test_multi_object_export_splits_within_each_source_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            output_dir = root / "out"
            make_two_object_glb(input_path)

            segment_glb(
                input_path=input_path,
                output_dir=output_dir,
                label_runner=RepeatedLabelsRunner(),
            )

            exported = trimesh.load(output_dir / "segmented_parts.glb", force="scene", process=False)
            self.assertEqual(
                sorted(exported.geometry.keys()),
                ["body_part_001", "body_part_002", "wheel_part_001", "wheel_part_002"],
            )

            manifest = json.loads((output_dir / "labels" / "part_manifest.json").read_text())
            self.assertEqual(
                sorted((part["source_object"], part["label"]) for part in manifest["parts"]),
                [("body", 1), ("body", 2), ("wheel", 1), ("wheel", 2)],
            )


if __name__ == "__main__":
    unittest.main()
