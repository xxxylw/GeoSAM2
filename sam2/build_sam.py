# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
#
# Modifications copyright 2025 VAST-AI-Research and the GeoSAM2 authors.

import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2

# Detect the common foot-gun where Python is launched from the parent of the
# repository so that `sam2` resolves to the repo directory rather than the
# package inside it.
if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
    raise RuntimeError(
        "You're likely running Python from the parent directory of the GeoSAM2 "
        "repository, which shadows the `sam2` Python package. Please run from "
        "inside the repo (or any unrelated directory) after installing the package."
    )


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
    **kwargs,
):
    """Build the GeoSAM2 image-level model.

    Args:
        config_file: Hydra config name (e.g. ``"configs/geosam2.yaml"``).
        ckpt_path: Optional path to a checkpoint to load.
        device: Torch device for the resulting model.
        mode: ``"eval"`` to switch the module to eval mode after loading.
        hydra_overrides_extra: Extra Hydra overrides to merge into the config.
        apply_postprocessing: If True, enable dynamic multimask stability for
            postprocessing.
    """
    hydra_overrides_extra = list(hydra_overrides_extra or [])
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra + [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor_geosam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
    **kwargs,
):
    """Build the GeoSAM2 video predictor used for multi-view propagation."""
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor_geosam2.SAM2VideoPredictor",
    ]

    hydra_overrides_extra = list(hydra_overrides_extra or [])
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra + [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    if missing_keys:
        logging.warning("Missing keys when loading checkpoint: %s", missing_keys)
    if unexpected_keys:
        logging.warning("Unexpected keys when loading checkpoint: %s", unexpected_keys)
    logging.info("Loaded checkpoint from %s", ckpt_path)
