#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${TRAIN_ROOT}/.." && pwd)"
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

# Usage: ./train_openvid4000_3router_4x48g.sh 0|1
CONDITIONING_MODE="${1:-0}"
case "${CONDITIONING_MODE}" in
  0) MODE_NAME="mq-replaces-t5" ;;
  1) MODE_NAME="mapped-mq-plus-t5" ;;
  *) echo "Usage: $0 0|1" >&2; exit 2 ;;
esac

export http_proxy="${http_proxy:-10.130.130.6:56830}"
export https_proxy="${https_proxy:-10.130.130.6:56830}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_API_KEY="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"
export WANDB_PROJECT="${WANDB_PROJECT:-moviestory-3router-4x48g}"
# The v2 preparation step has already applied the same filters in parallel.
# Avoid decoding all 4000 videos a second time inside every torchrun rank.
export WAN_DATA_PRECLEAN="0"

OPENVID_ROOT="/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M"
OPENVID_LIMIT=4000
FRAME_NUM=81
# Keep preparation caches separate when the training frame requirement changes.
SUBSET_ROOT="${CODE_ROOT}/tmp/openvid_trainable_first${OPENVID_LIMIT}_v2_f${FRAME_NUM}"
CAPTION_TOKENIZER="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B/google/umt5-xxl"
LEGACY_SUBSET_MANIFEST="${CODE_ROOT}/tmp/openvid_first${OPENVID_LIMIT}/manifest.json"
CANDIDATE_MANIFEST_ARGS=()
if [[ -f "${LEGACY_SUBSET_MANIFEST}" ]]; then
  CANDIDATE_MANIFEST_ARGS+=(--candidate_manifest "${LEGACY_SUBSET_MANIFEST}")
fi
NUM_TRAIN_STEPS=520
GLOBAL_EFFECTIVE_BATCH=8
OUTPUT_DIR="${CODE_ROOT}/checkpoint/three_router_${MODE_NAME}_strongbind_openvid${OPENVID_LIMIT}_4x48g_steps${NUM_TRAIN_STEPS}"

source "${CODE_ROOT}/scripts/wandb_env.sh"
configure_wandb_args \
  "three-router-${MODE_NAME}-strongbind-openvid${OPENVID_LIMIT}-4x48g" \
  "moviestory,three-router,${MODE_NAME},first-frame,strong-bind,4x48g,fsdp"

python "${CODE_ROOT}/scripts/prepare_openvid_subset.py" \
  --video_root "${OPENVID_ROOT}/video" \
  --csv_path "${OPENVID_ROOT}/data/train/OpenVid-1M.csv" \
  --output_root "${SUBSET_ROOT}" \
  --limit "${OPENVID_LIMIT}" \
  --probe_training_video \
  --frame_num "${FRAME_NUM}" \
  --min_duration_sec 0.5 \
  --max_duration_sec 20.0 \
  --max_caption_tokens 512 \
  --probe_workers 8 \
  --caption_tokenizer_path "${CAPTION_TOKENIZER}" \
  "${CANDIDATE_MANIFEST_ARGS[@]}"

torchrun --standalone --nproc_per_node=4 \
  "${TRAIN_ROOT}/train_3router_wan_4x48g.py" \
  --conditioning_mode "${CONDITIONING_MODE}" \
  --expected_world_size 4 \
  --minimum_gpu_memory_gib 44 \
  --minimum_free_gpu_memory_gib 42 \
  --global_effective_batch "${GLOBAL_EFFECTIVE_BATCH}" \
  --expected_train_samples "${OPENVID_LIMIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --local_openvid_video_root "${SUBSET_ROOT}/videos" \
  --local_openvid_csv_path "${SUBSET_ROOT}/openvid_first${OPENVID_LIMIT}.csv" \
  --local_openvid_limit "${OPENVID_LIMIT}" \
  --caption_tokenizer_path "${CAPTION_TOKENIZER}" \
  --num_metaqueries 256 \
  --connector_num_hidden_layers 24 \
  --num_train_steps "${NUM_TRAIN_STEPS}" \
  --batch_size 1 \
  --gradient_accumulation_steps 2 \
  --save_steps 250 \
  --learning_rate 1e-5 \
  --warmup_steps 100 \
  --router_log_steps 1 \
  --router_stale_update_patience 5 \
  --frame_num "${FRAME_NUM}" \
  --max_area 262144 \
  --wan_train_mode cond_only \
  --dit_fsdp \
  --t5_cpu \
  --wan_first_frame_strong_bind \
  --enable_ti2v_first_frame_condition \
  --train_video_conditioning_mode wan_animate_slot \
  --train_ref_anchor_mode none \
  --joint_null_prob 0.1 \
  --freeze_mq_input_embeddings \
  --mq_gradient_checkpointing \
  --disable_t5_alignment \
  --disable_wan_func_distill \
  --mq_norm_probe_with_t5 \
  --mq_norm_match_t5 \
  --null_image_prob 0.1 \
  --null_caption_prob 0.1 \
  --aggressive_empty_cache \
  --log_cuda_memory \
  "${WANDB_ARGS[@]}"
