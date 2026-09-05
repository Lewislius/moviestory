#!/bin/bash
# =============================================================================
# MetaQuery + Qwen3-VL Training Launch Script
# =============================================================================
# Usage:
#   Stage 1 (Text-to-Image, on cc12m):
#     bash scripts/train_qwen3vl.sh qwen3vl2b_t2i qwen3vl2b_sana.yaml /path/to/base_dir
#
#   Stage 2 (Instruction Tuning, on MetaQuery-Instruct-2.4M):
#     bash scripts/train_qwen3vl.sh qwen3vl2b_inst qwen3vl2b_sana_inst.yaml /path/to/base_dir
#
#   Stage 2 Alt (Image Editing, on OmniEdit):
#     bash scripts/train_qwen3vl.sh qwen3vl2b_edit qwen3vl2b_sana_edit.yaml /path/to/base_dir
# =============================================================================

set -e

run_name=${1:-"qwen3vl2b_t2i"}
config_file=${2:-"qwen3vl2b_sana.yaml"}
base_dir=${3:-"/path/to/base_dir"}
num_gpus=${4:-8}

echo "============================================="
echo "MetaQuery + Qwen3-VL Training"
echo "============================================="
echo "Run Name:    ${run_name}"
echo "Config:      ${config_file}"
echo "Base Dir:    ${base_dir}"
echo "Num GPUs:    ${num_gpus}"
echo "============================================="

export OMP_NUM_THREADS=12

# Single-node training
torchrun \
    --nproc-per-node=${num_gpus} \
    train.py \
    --run_name ${run_name} \
    --config_file ${config_file} \
    --base_dir ${base_dir}
