#!/bin/bash
set -euo pipefail

# 2x80G A100 launcher for the overfit20_i2v_frame training script.
#
# Topology:
#   - single process
#   - 2 GPUs per process
#   - auto_device_map => dit_device=0, encoder_device=1
#
# This matches the current code path:
#   - WAN_TRAIN_MODE != frozen is recommended to stay single-process
#   - model split across two GPUs is the safest way to keep each card under 80G

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/train_stage1_openvid_local_metaquery_overfit20_ti2v_frame.sh"

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

export NPROC_PER_NODE="1"
export GPUS_PER_PROCESS="2"
export WAN_DIST_PROCESS_DEVICE="${WAN_DIST_PROCESS_DEVICE:-encoder}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit20_ti2v_firstframe_wan_cond_2x80g}"
export WANDB_TAGS="${WANDB_TAGS:-overfit,firstframe,ti2v,mq,wan-condition,2x80g}"

echo "[2x80G] base_script=${BASE_SCRIPT}"
echo "[2x80G] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[2x80G] NPROC_PER_NODE=${NPROC_PER_NODE} GPUS_PER_PROCESS=${GPUS_PER_PROCESS}"
echo "[2x80G] topology=single-process dual-gpu split (dit->gpu0, encoder->gpu1)"
echo "[2x80G] output_root=${OUTPUT_ROOT}"

exec bash "${BASE_SCRIPT}" "$@"
