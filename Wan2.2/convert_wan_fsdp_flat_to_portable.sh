#!/bin/bash
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

export http_proxy="${http_proxy:-10.130.130.6:56830}"
export https_proxy="${https_proxy:-10.130.130.6:56830}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29549}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-/home/liuzhirui/model/Wan2.2/convert_wan_fsdp_flat_to_portable.py}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_full_4gpu/ti2v_overfit30_steps800_nummq256_nullimg0.1_nullcap0.1/checkpoint-600}"
WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"

echo "[CONVERT] nproc_per_node=${NPROC_PER_NODE} master_port=${MASTER_PORT}"
echo "[CONVERT] checkpoint_dir=${CHECKPOINT_DIR}"
echo "[CONVERT] wan_checkpoint_dir=${WAN_CHECKPOINT_DIR}"
# FSDP flat checkpoint 需要尽量和训练时保持一致的 world_size/拓扑，否则容易出现
# missing/unexpected 或 flat 参数无法注入的问题。

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" "${CONVERT_SCRIPT}" \
  --distributed \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --wan_checkpoint_dir "${WAN_CHECKPOINT_DIR}" \
  --replace
