#!/usr/bin/env bash
# End-to-end example: run the single-view point-prompt segmenter to obtain a 2D
# mask, then propagate it across all rendered views to produce a 3D segmentation.
#
# Override DATA_ROOT/PROMPT_FILE/VIEW_IDX with environment variables to run on
# your own assets, e.g. `DATA_ROOT=/abs/path/to/render bash scripts/run_example.sh`.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-example/sample_00}"
PROMPT_FILE="${PROMPT_FILE:-${DATA_ROOT}/point_prompts_scale1.json}"
VIEW_IDX="${VIEW_IDX:-0}"

OBJ_NAME="$(basename "${DATA_ROOT}")"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${OBJ_NAME}}"
SINGLE_VIEW_OUTPUT_DIR="${BASE_OUTPUT_DIR}/2d_seg"
FINAL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/3d_seg"

# Use --no-opposite-auto-segmentation / --no-enable-postprocess to disable.
AUTO_SEG_FLAG="${AUTO_SEG_FLAG:---opposite-auto-segmentation}"
POSTPROCESS_FLAG="${POSTPROCESS_FLAG:---enable-postprocess}"

mkdir -p "${SINGLE_VIEW_OUTPUT_DIR}" "${FINAL_OUTPUT_DIR}"

python single_view_point_prompt_infer.py \
  --data-root "${DATA_ROOT}" \
  --view-idx "${VIEW_IDX}" \
  --point-prompt-file "${PROMPT_FILE}" \
  --output-dir "${SINGLE_VIEW_OUTPUT_DIR}"

MASK_PATH="${SINGLE_VIEW_OUTPUT_DIR}/mask_view$(printf '%04d' "${VIEW_IDX}").npy"

python inference.py \
  --data-root "${DATA_ROOT}" \
  --mask-path "${MASK_PATH}" \
  --mask-view "${VIEW_IDX}" \
  "${AUTO_SEG_FLAG}" \
  "${POSTPROCESS_FLAG}" \
  --output-dir "${FINAL_OUTPUT_DIR}"
