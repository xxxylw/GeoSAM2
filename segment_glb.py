#!/usr/bin/env python
"""CLI wrapper for GLB part-object segmentation."""

from __future__ import annotations

import argparse
import json
import sys

from geosam2.pipeline import SegmentError, segment_glb


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment a GLB into part objects",
        allow_abbrev=False,
    )
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["automatic", "prompted"], default="automatic")
    parser.add_argument("--cache-dir", default=".cache/geosam2")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--strict-pbr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-preview", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mask-path", default=None)
    parser.add_argument("--mask-view", type=int, default=None)
    parser.add_argument("--points", "--points-path", dest="points_path", default=None)
    parser.add_argument("--prompt-view", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        result = segment_glb(
            input_path=args.input_path,
            output_dir=args.output_dir,
            mode=args.mode,
            cache_dir=args.cache_dir,
            gpu=args.gpu,
            strict_pbr=args.strict_pbr,
            write_preview=args.write_preview,
            mask_path=args.mask_path,
            mask_view=args.mask_view,
            points_path=args.points_path,
            prompt_view=args.prompt_view,
        )
    except SegmentError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
