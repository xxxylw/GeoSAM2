import json
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
    trimesh.creation.box().export(path)


class VisualizationDebugArtifactBehaviorTests(unittest.TestCase):
    def test_preview_artifact_is_opt_in_and_separate_from_successful_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            output_dir = root / "out"
            make_box_glb(input_path)

            result = segment_glb(
                input_path=input_path,
                output_dir=output_dir,
                write_preview=True,
                label_runner=TwoPartRunner(),
            )

            self.assertEqual(result.status, "segmented")
            self.assertTrue((output_dir / "segmented_parts.glb").exists())
            self.assertTrue((output_dir / "preview" / "segmentation_colored.glb").exists())
            self.assertNotEqual(
                str(output_dir / "preview" / "segmentation_colored.glb"),
                result.output_glb,
            )

            manifest = json.loads((output_dir / "labels" / "part_manifest.json").read_text())
            self.assertEqual(manifest["preview"]["segmentation_colored_glb"], "preview/segmentation_colored.glb")
            self.assertEqual(manifest["successful_output"], "segmented_parts.glb")

    def test_preview_is_not_written_by_default(self) -> None:
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

            self.assertFalse((output_dir / "preview" / "segmentation_colored.glb").exists())


if __name__ == "__main__":
    unittest.main()
