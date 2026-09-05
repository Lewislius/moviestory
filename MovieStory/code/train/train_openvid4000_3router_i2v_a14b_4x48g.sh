#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${TRAIN_ROOT}/.." && pwd)"
CONDA_SH="${CONDA_SH:-/home/liuzhirui/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/home/liuzhirui/miniconda3/envs/moviestory}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# 0: MQ replaces T5; 1: mapped MQ followed by frozen native T5 context.
CONDITIONING_MODE="${1:-${WAN_I2V_CONDITIONING_MODE:-0}}"
case "${CONDITIONING_MODE}" in
  0) MODE_NAME="mq-only" ;;
  1) MODE_NAME="mapped-mq-plus-t5" ;;
  *) echo "Usage: $0 0|1" >&2; exit 2 ;;
esac

export http_proxy="${http_proxy:-10.130.130.6:56830}"
export https_proxy="${https_proxy:-10.130.130.6:56830}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export WAN_I2V_DIST_TIMEOUT_SEC="${WAN_I2V_DIST_TIMEOUT_SEC:-300}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="false"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_PROJECT="${WANDB_PROJECT:-wan-i2v-a14b-native-3router}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WAN_DATA_PRECLEAN="${WAN_DATA_PRECLEAN:-1}"
export WAN_DATA_PRECLEAN_POOL_FACTOR="${WAN_DATA_PRECLEAN_POOL_FACTOR:-1.25}"
export WAN_DATA_PRECLEAN_LOG_INTERVAL="${WAN_DATA_PRECLEAN_LOG_INTERVAL:-25}"
export WAN_DATA_PROBE_TIMEOUT_SEC="${WAN_DATA_PROBE_TIMEOUT_SEC:-8}"
export WAN_DATA_DECODE_TIMEOUT_SEC="${WAN_DATA_DECODE_TIMEOUT_SEC:-30}"
export WAN_DATA_MAX_TRIALS="${WAN_DATA_MAX_TRIALS:-4}"
export WAN_DATA_TRIAL_LOG_INTERVAL="${WAN_DATA_TRIAL_LOG_INTERVAL:-1}"
export WAN_DATA_CACHE_WAIT_TIMEOUT_SEC="${WAN_DATA_CACHE_WAIT_TIMEOUT_SEC:-1800}"
export WAN_KEEP_BEFORE_TRAINING_CHECKPOINT="${WAN_KEEP_BEFORE_TRAINING_CHECKPOINT:-0}"

WAN_CHECKPOINT="${WAN_CHECKPOINT:-/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B}"
QWEN_CHECKPOINT="${QWEN_CHECKPOINT:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
OPENVID_ROOT="${OPENVID_ROOT:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M}"
OPENVID_VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-${OPENVID_ROOT}/video}"
OPENVID_CSV_PATH="${OPENVID_CSV_PATH:-${OPENVID_ROOT}/data/train/OpenVid-1M.csv}"
CAPTION_TOKENIZER="${CAPTION_TOKENIZER:-${WAN_CHECKPOINT}/google/umt5-xxl}"

OPENVID_LIMIT="${OPENVID_LIMIT:-4000}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-200}"
GLOBAL_EFFECTIVE_BATCH="${GLOBAL_EFFECTIVE_BATCH:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WAN_TRAIN_MODE="${WAN_TRAIN_MODE:-cond_only}"
WAN_ACTIVATION_CHECKPOINTING="${WAN_ACTIVATION_CHECKPOINTING:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_ROOT}/checkpoint/native_i2v_a14b_3router_${MODE_NAME}_openvid${OPENVID_LIMIT}_${NPROC_PER_NODE}x48g_steps${NUM_TRAIN_STEPS}}"

if (( NPROC_PER_NODE <= 0 )); then
  echo "NPROC_PER_NODE must be positive, got ${NPROC_PER_NODE}" >&2
  exit 2
fi
if (( GLOBAL_EFFECTIVE_BATCH % NPROC_PER_NODE != 0 )); then
  echo "GLOBAL_EFFECTIVE_BATCH=${GLOBAL_EFFECTIVE_BATCH} must be divisible by NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
  exit 2
fi

WAN_CHECKPOINT_ARGS=()
case "${WAN_ACTIVATION_CHECKPOINTING,,}" in
  1|true|on|yes)
    WAN_CHECKPOINT_ARGS+=(--enable_wan_activation_checkpointing)
    ;;
  0|false|off|no)
    WAN_CHECKPOINT_ARGS+=(--disable_wan_activation_checkpointing)
    ;;
  *)
    echo "WAN_ACTIVATION_CHECKPOINTING must be 0/1 or false/true" >&2
    exit 2
    ;;
esac

WANDB_ARGS=()
if [[ "${WANDB_ENABLED:-0}" == "1" ]]; then
  WANDB_ARGS+=(
    --wandb_enabled
    --wandb_project "${WANDB_PROJECT:-wan-i2v-a14b-3router}"
    --wandb_run_name "${WANDB_RUN_NAME:-native-i2v-a14b-3router-${MODE_NAME}-openvid${OPENVID_LIMIT}}"
    --wandb_mode "${WANDB_MODE:-online}"
  )
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${TRAIN_ROOT}/train_metaquery_i2v_3router_4x48g.py" \
  --conditioning_mode "${CONDITIONING_MODE}" \
  --wan_checkpoint_dir "${WAN_CHECKPOINT}" \
  --qwen3vl_model_id "${QWEN_CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --local_openvid_video_root "${OPENVID_VIDEO_ROOT}" \
  --local_openvid_csv_path "${OPENVID_CSV_PATH}" \
  --local_openvid_limit "${OPENVID_LIMIT}" \
  --caption_tokenizer_path "${CAPTION_TOKENIZER}" \
  --expected_world_size "${NPROC_PER_NODE}" \
  --expected_train_samples "${OPENVID_LIMIT}" \
  --global_effective_batch "${GLOBAL_EFFECTIVE_BATCH}" \
  --num_train_steps "${NUM_TRAIN_STEPS}" \
  --num_metaqueries 256 \
  --router_role_tokens 96 \
  --router_action_tokens 96 \
  --router_global_tokens 64 \
  --connector_num_hidden_layers 24 \
  --frame_num 37 \
  --max_area 147456 \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --warmup_steps "${WARMUP_STEPS:-25}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --wan_train_mode "${WAN_TRAIN_MODE}" \
  "${WAN_CHECKPOINT_ARGS[@]}" \
  --null_caption_prob "${NULL_CAPTION_PROB:-0.1}" \
  --null_image_prob "${NULL_IMAGE_PROB:-0.1}" \
  --strict_dataset_size \
  "${WANDB_ARGS[@]}"
