#!/bin/bash
set -euo pipefail

# 2x80G A100 launcher for the MQ-only overfit10 training script.
#
# Topology:
#   - single process
#   - 2 GPUs per process
#   - auto_device_map => dit_device=0, encoder_device=1
#
# Expected effect:
#   - GPU0 mainly hosts Wan DiT / VAE-side work
#   - GPU1 mainly hosts Qwen3-VL / MQ encoder / connector
# This is the right fit for a "single-card peak ~92G" job that needs to be
# split across exactly two 80G cards.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/train_stage1_openvid_local_metaquery_overfit10.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "[ERROR] base script not found: ${BASE_SCRIPT}"
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1"
fi

IFS=',' read -r -a _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#_GPU_ARR[@]} < 2 )); then
  echo "[ERROR] This launcher requires 2 visible GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  exit 2
fi

# Force the intended 2x80G topology.
export NPROC_PER_NODE="1"
export GPUS_PER_PROCESS="2"

# Keep the process group pinned to the encoder card, which matches the DDP-wrapped
# MQ encoder in train_metaquery_*_new.py.
export WAN_DIST_PROCESS_DEVICE="${WAN_DIST_PROCESS_DEVICE:-encoder}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# Give the dual-GPU run its own output directory by default to avoid overwriting
# single-GPU experiments.
export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit20_2x80g}"
export WANDB_TAGS="${WANDB_TAGS:-overfit,debug,overfit10,2x80g}"

echo "[2x80G] base_script=${BASE_SCRIPT}"
echo "[2x80G] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[2x80G] NPROC_PER_NODE=${NPROC_PER_NODE} GPUS_PER_PROCESS=${GPUS_PER_PROCESS}"
echo "[2x80G] topology=single-process dual-gpu split (dit->gpu0, encoder->gpu1)"
echo "[2x80G] output_root=${OUTPUT_ROOT}"

exec bash "${BASE_SCRIPT}" "$@"
