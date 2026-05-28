import json
import tempfile
import unittest
from pathlib import Path

import trimesh

from geosam2.pipeline import segment_glb


def make_box_glb(path: Path, scale: float = 1.0) -> None:
    mesh = trimesh.creation.box(extents=(scale, 1.0, 1.0))
    mesh.export(path)


class RenderCacheAndGpuBehaviorTests(unittest.TestCase):
    def test_same_input_and_settings_reuses_render_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            cache_dir = root / "cache"
            make_box_glb(input_path)

            first = root / "first"
            second = root / "second"
            segment_glb(input_path=input_path, output_dir=first, cache_dir=cache_dir)
            segment_glb(input_path=input_path, output_dir=second, cache_dir=cache_dir)

            key1 = (first / "debug" / "render_cache_key.txt").read_text()
            key2 = (second / "debug" / "render_cache_key.txt").read_text()
            self.assertEqual(key1, key2)
            self.assertTrue((cache_dir / key1.strip()).is_dir())

    def test_changed_glb_content_changes_render_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_input = root / "asset_a.glb"
            second_input = root / "asset_b.glb"
            cache_dir = root / "cache"
            make_box_glb(first_input, scale=1.0)
            make_box_glb(second_input, scale=2.0)

            first = root / "first"
            second = root / "second"
            segment_glb(input_path=first_input, output_dir=first, cache_dir=cache_dir)
            segment_glb(input_path=second_input, output_dir=second, cache_dir=cache_dir)

            key1 = (first / "debug" / "render_cache_key.txt").read_text()
            key2 = (second / "debug" / "render_cache_key.txt").read_text()
            self.assertNotEqual(key1, key2)

    def test_gpu_selection_is_recorded_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "asset.glb"
            output_dir = root / "out"
            make_box_glb(input_path)

            segment_glb(input_path=input_path, output_dir=output_dir, gpu=2)

            manifest = json.loads((output_dir / "labels" / "part_manifest.json").read_text())
            self.assertEqual(manifest["run"]["gpu"], 2)


if __name__ == "__main__":
    unittest.main()
