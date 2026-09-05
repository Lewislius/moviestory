#!/bin/bash
# set -euo pipefail
# eval "$(conda shell.bash hook)"
# conda env list
# conda activate /home/liuzhirui/miniconda3/envs/moviestory
# export http_proxy=10.130.130.6:56830
# export https_proxy=10.130.130.6:56830
# export HF_ENDPOINT=https://hf-mirror.com
# Set HF_TOKEN in the environment if authentication is required.
# export WANDB_MODE="${WANDB_MODE:-online}"
# export WANDB_API_KEY="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"
# export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"
# export TORCH_NCCL_BLOCKING_WAIT=1
# # 用法:
# #   bash train_stage1_openvid_local_metaquery_with_wan.sh
# export OPENVID_VIDEO_ROOT=/home/liuzhirui/dataset/OpenVid-1M/video
# export OPENVID_CSV_PATH=/home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVid-1M.csv
# export OPENVID_HD_VIDEO_ROOT=/home/liuzhirui/dataset/OpenVid-1M/video_HD
# export OPENVID_HD_CSV_PATH=/home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVidHD.csv

# # 任务类型: ti2v | i2v | animate (直接改这里)
# TASK="animate"

# VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-/home/liuzhirui/dataset/OpenVid-1M/video}"
# CSV_PATH="${OPENVID_CSV_PATH:-/home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVid-1M.csv}"
# LOCAL_LIMIT="${OPENVID_LOCAL_LIMIT:-}"
# VIDEO_HD_ROOT="${OPENVID_HD_VIDEO_ROOT:-/home/liuzhirui/dataset/OpenVid-1M/video_HD}"
# CSV_HD_PATH="${OPENVID_HD_CSV_PATH:-/home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVidHD.csv}"
# LOCAL_HD_LIMIT="${OPENVID_HD_LOCAL_LIMIT:-}"
# LOCAL_VIDEO_CACHE_DIR="${OPENVID_LOCAL_CACHE_DIR:-/home/liuzhirui/dataset/OpenVid-1M/.cache_video}"

# QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
# WAN_TI2V_CKPT="${WAN_TI2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
# WAN_I2V_CKPT="${WAN_I2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B}"
# WAN_ANIMATE_CKPT="${WAN_ANIMATE_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-Animate-14B}"

# OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_openvid_stage1_full_training}"
# NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-5000}"
# SAVE_STEPS="${SAVE_STEPS:-500}"
# LOG_STEPS="${LOG_STEPS:-10}"
# NUM_METAQUERIES="${NUM_METAQUERIES:-128}"

# # Determined 场景下不显式绑定卡号；由平台分配可见GPU
# # 默认按4卡启动，可通过环境变量 NPROC_PER_NODE 覆盖
# NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# COMMON_ARGS=(
#   --distributed
#   --auto_device_map
#   --gpus_per_process 1
#   --hf_stage stage1
#   --hf_no_streaming
#   --t5_cpu
#   --mq_gradient_checkpointing
#   --aggressive_empty_cache
#   --local_openvid_video_root "${VIDEO_ROOT}"
#   --local_openvid_csv_path "${CSV_PATH}"
#   --local_openvid_hd_video_root "${VIDEO_HD_ROOT}"
#   --local_openvid_hd_csv_path "${CSV_HD_PATH}"
#   --local_video_cache_dir "${LOCAL_VIDEO_CACHE_DIR}"
#   --qwen3vl_model_id "${QWEN_MODEL}"
#   --num_metaqueries "${NUM_METAQUERIES}"
#   --num_train_steps "${NUM_TRAIN_STEPS}"
#   --save_steps "${SAVE_STEPS}"
#   --log_steps "${LOG_STEPS}"
# )

# if [[ -n "${LOCAL_LIMIT}" ]]; then
#   COMMON_ARGS+=(--local_openvid_limit "${LOCAL_LIMIT}")
# fi
# if [[ -n "${LOCAL_HD_LIMIT}" ]]; then
#   COMMON_ARGS+=(--local_openvid_hd_limit "${LOCAL_HD_LIMIT}")
# fi

# case "${TASK}" in
#   ti2v)
#     torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_wan_new.py \
#       --wan_checkpoint_dir "${WAN_TI2V_CKPT}" \
#       --output_dir "${OUTPUT_ROOT}/ti2v_stage1_openvid_local" \
#       "${COMMON_ARGS[@]}"
#     ;;
#   i2v)
#     torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_i2v_new.py \
#       --wan_checkpoint_dir "${WAN_I2V_CKPT}" \
#       --output_dir "${OUTPUT_ROOT}/i2v_stage1_openvid_local" \
#       "${COMMON_ARGS[@]}"
#     ;;
#   animate)
#     torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_animate_new.py \
#       --wan_checkpoint_dir "${WAN_ANIMATE_CKPT}" \
#       --output_dir "${OUTPUT_ROOT}/animate_stage1_openvid_local" \
#       "${COMMON_ARGS[@]}"
#     ;;
#   *)
#     echo "[ERROR] TASK 必须是: ti2v | i2v | animate"
#     exit 1
#     ;;
# esac

set -euo pipefail
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory
export http_proxy=10.130.130.6:56830
export https_proxy=10.130.130.6:56830
export HF_ENDPOINT=https://hf-mirror.com
: "${HF_TOKEN:?Set HF_TOKEN in the environment before running this script}"
# export WANDB_DISABLED=true
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="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"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTHONUNBUFFERED=1
# 用法:
#   bash train_stage1_openvid_local_metaquery_with_wan.sh
export OPENVID_VIDEO_ROOT=/home/liuzhirui/dataset/OpenVid-1M/video
export OPENVID_CSV_PATH=/home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVid-1M.csv
export OPENVID_HD_VIDEO_ROOT=/home/liuzhirui/dataset/OpenVid-1M/video_HD
export OPENVID_HD_CSV_PATH=/home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVidHD.csv

# 任务类型: ti2v | i2v | animate (直接改这里)
TASK="ti2v"
# SMOKE_TEST_1000=1
NUM_TRAIN_STEPS=200 
CUDA_LAUNCH_BLOCKING=1
WAN_DISABLE_FLASH_ATTN=0 
WAN_FLASH_ATTN_FORCE_VERSION=2 
WAN_FLASH_ATTN_FALLBACK_SDPA=1 
ANIMATE_FRAME_NUM=49
NUM_METAQUERIES=64
ANIMATE_MAX_AREA=262144
VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-/home/liuzhirui/dataset/OpenVid-1M/video}"
CSV_PATH="${OPENVID_CSV_PATH:-/home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVid-1M.csv}"
LOCAL_LIMIT="${OPENVID_LOCAL_LIMIT:-}"
VIDEO_HD_ROOT="${OPENVID_HD_VIDEO_ROOT:-/home/liuzhirui/dataset/OpenVid-1M/video_HD}"
CSV_HD_PATH="${OPENVID_HD_CSV_PATH:-/home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVidHD.csv}"
LOCAL_HD_LIMIT="${OPENVID_HD_LOCAL_LIMIT:-}"
LOCAL_TOTAL_LIMIT="${OPENVID_LOCAL_TOTAL_LIMIT:-}"
LOCAL_VIDEO_CACHE_DIR="${OPENVID_LOCAL_CACHE_DIR:-/home/liuzhirui/dataset/OpenVid-1M/.cache_video}"

# 1000条快速测试开关:
#   SMOKE_TEST_1000=1 时，默认只用普通 OpenVid 前1000条，且关闭 HD 源。
#   如需覆盖数量，可同时传 OPENVID_LOCAL_LIMIT。
SMOKE_TEST_1000="${SMOKE_TEST_1000:-0}"
if [[ "${SMOKE_TEST_1000}" == "1" ]]; then
  LOCAL_LIMIT="${OPENVID_LOCAL_LIMIT:-1000}"
  LOCAL_TOTAL_LIMIT="${OPENVID_LOCAL_TOTAL_LIMIT:-1000}"
  VIDEO_HD_ROOT=""
  CSV_HD_PATH=""
  LOCAL_HD_LIMIT=""
fi

if [[ -n "${LOCAL_TOTAL_LIMIT}" ]]; then
  export OPENVID_LOCAL_TOTAL_LIMIT="${LOCAL_TOTAL_LIMIT}"
fi

QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
WAN_TI2V_CKPT="${WAN_TI2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
WAN_I2V_CKPT="${WAN_I2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B}"
WAN_ANIMATE_CKPT="${WAN_ANIMATE_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-Animate-14B}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_openvid_stage1_full_training}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-50}"
LOG_STEPS="${LOG_STEPS:-50}"
LOG_EVERY_STEP="${LOG_EVERY_STEP:-1}"
WANDB_LOG_EVERY_STEP="${WANDB_LOG_EVERY_STEP:-1}"
LOG_CUDA_MEMORY="${LOG_CUDA_MEMORY:-1}"
NUM_METAQUERIES="${NUM_METAQUERIES:-64}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_LOG_CHECKPOINT="${WANDB_LOG_CHECKPOINT:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-wan-metaquery}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS="${WANDB_TAGS:-stage1,openvid,local}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
METRICS_JSONL_PATH="${METRICS_JSONL_PATH:-${OUTPUT_ROOT}/logs/${TASK}_train_metrics_steps${NUM_TRAIN_STEPS}_$(date +%Y%m%d-%H%M%S).jsonl}"

TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-49}"
TI2V_MAX_AREA="${TI2V_MAX_AREA:-262144}"        # 1280*720，显著降低 TI2V 显存
TI2V_MIN_DURATION_SEC="${TI2V_MIN_DURATION_SEC:-0.3}"
TI2V_NUM_METAQUERIES="${TI2V_NUM_METAQUERIES:-64}"
I2V_FRAME_NUM="${I2V_FRAME_NUM:-49}"
I2V_MAX_AREA="${I2V_MAX_AREA:-921600}"          # 384*384，降低 I2V 显存
I2V_MIN_DURATION_SEC="${I2V_MIN_DURATION_SEC:-0.3}"
I2V_NUM_METAQUERIES="${I2V_NUM_METAQUERIES:-64}"
ANIMATE_FRAME_NUM="${ANIMATE_FRAME_NUM:-49}"
ANIMATE_MIN_DURATION_SEC="${ANIMATE_MIN_DURATION_SEC:-0.3}"
ANIMATE_MAX_AREA="${ANIMATE_MAX_AREA:-262144}"   # 384*384，降低显存占用
ANIMATE_NUM_METAQUERIES="${ANIMATE_NUM_METAQUERIES:-64}"

# 固定 2x2 并行拓扑（显式写死）:
# - 2 个训练进程 (rank0, rank1)
# - 每个进程使用 2 张卡
# - 依赖 train_metaquery_*_new.py 中 --auto_device_map 的映射:
#   rank0 -> dit_device=0, encoder_device=1
#   rank1 -> dit_device=2, encoder_device=3
# 注意: 不显式指定 GPU 卡号；仍由平台提供 CUDA_VISIBLE_DEVICES。
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GPUS_PER_PROCESS="${GPUS_PER_PROCESS:-2}"
echo "[LAUNCH] TASK=${TASK} NPROC_PER_NODE=${NPROC_PER_NODE} GPUS_PER_PROCESS=${GPUS_PER_PROCESS}"
echo "[LAUNCH] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[LAUNCH] expected_topology: rank0[dit=0,enc=1], rank1[dit=2,enc=3]"
echo "[LAUNCH] LOG_STEPS=${LOG_STEPS} LOG_EVERY_STEP=${LOG_EVERY_STEP} WANDB_LOG_EVERY_STEP=${WANDB_LOG_EVERY_STEP}"
echo "[LAUNCH] LOG_CUDA_MEMORY=${LOG_CUDA_MEMORY} METRICS_JSONL_PATH=${METRICS_JSONL_PATH}"
echo "[LAUNCH] WANDB_LOG_CHECKPOINT=${WANDB_LOG_CHECKPOINT}"

COMMON_ARGS=(
  --distributed
  --auto_device_map
  --gpus_per_process "${GPUS_PER_PROCESS}"
  --hf_stage stage1
  --hf_no_streaming
  --t5_cpu
  --mq_gradient_checkpointing
  --aggressive_empty_cache
  --local_openvid_video_root "${VIDEO_ROOT}"
  --local_openvid_csv_path "${CSV_PATH}"
  --local_openvid_hd_video_root "${VIDEO_HD_ROOT}"
  --local_openvid_hd_csv_path "${CSV_HD_PATH}"
  --local_video_cache_dir "${LOCAL_VIDEO_CACHE_DIR}"
  --qwen3vl_model_id "${QWEN_MODEL}"
  --num_metaqueries "${NUM_METAQUERIES}"
  --num_train_steps "${NUM_TRAIN_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --log_steps "${LOG_STEPS}"
)

if [[ "${LOG_EVERY_STEP}" == "1" ]]; then
  COMMON_ARGS+=(--log_every_step)
fi
if [[ "${LOG_CUDA_MEMORY}" == "1" ]]; then
  COMMON_ARGS+=(--log_cuda_memory)
fi
if [[ -n "${METRICS_JSONL_PATH}" ]]; then
  COMMON_ARGS+=(--metrics_jsonl_path "${METRICS_JSONL_PATH}")
fi

if [[ -n "${LOCAL_LIMIT}" ]]; then
  COMMON_ARGS+=(--local_openvid_limit "${LOCAL_LIMIT}")
fi
if [[ -n "${LOCAL_HD_LIMIT}" ]]; then
  COMMON_ARGS+=(--local_openvid_hd_limit "${LOCAL_HD_LIMIT}")
fi

# W&B 适配：检测到 API Key 且显式启用时自动打开日志
if [[ "${WANDB_ENABLED}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
  COMMON_ARGS+=(--wandb_enabled)
  COMMON_ARGS+=(--wandb_project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    COMMON_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_MODE}" ]]; then
    COMMON_ARGS+=(--wandb_mode "${WANDB_MODE}")
  fi
  if [[ -n "${WANDB_API_KEY}" ]]; then
    COMMON_ARGS+=(--wandb_api_key "${WANDB_API_KEY}")
  fi
  if [[ -n "${WANDB_TAGS}" ]]; then
    COMMON_ARGS+=(--wandb_tags "${WANDB_TAGS}")
  fi
  if [[ "${WANDB_LOG_EVERY_STEP}" == "1" ]]; then
    COMMON_ARGS+=(--wandb_log_every_step)
  fi
  if [[ "${WANDB_LOG_CHECKPOINT}" == "1" ]]; then
    COMMON_ARGS+=(--wandb_log_checkpoint)
  fi
  if [[ -n "${WANDB_RUN_NAME}" ]]; then
    COMMON_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
  else
    COMMON_ARGS+=(--wandb_run_name "mq-${TASK}-stage1-openvid-local-$(date +%Y%m%d-%H%M%S)")
  fi
fi

case "${TASK}" in
  ti2v)
    export WAN_DISABLE_FLASH_ATTN="${WAN_DISABLE_FLASH_ATTN:-0}"   # TI2V 默认启用 flash-attn
    export WAN_FLASH_ATTN_FORCE_VERSION="${WAN_FLASH_ATTN_FORCE_VERSION:-2}"   # 强制 FA2
    export WAN_FLASH_ATTN_FALLBACK_SDPA="${WAN_FLASH_ATTN_FALLBACK_SDPA:-1}"
    export WAN_FLASH_ATTN_FORCE_CONTIGUOUS="${WAN_FLASH_ATTN_FORCE_CONTIGUOUS:-1}"
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
    export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

    echo "[LAUNCH] TI2V low-vram"
    echo "[LAUNCH] NPROC_PER_NODE=${NPROC_PER_NODE} GPUS_PER_PROCESS=${GPUS_PER_PROCESS}"
    echo "[LAUNCH] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "[LAUNCH] WAN_DISABLE_FLASH_ATTN=${WAN_DISABLE_FLASH_ATTN} WAN_FLASH_ATTN_FORCE_VERSION=${WAN_FLASH_ATTN_FORCE_VERSION}"
    echo "[LAUNCH] WAN_FLASH_ATTN_FALLBACK_SDPA=${WAN_FLASH_ATTN_FALLBACK_SDPA} CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}"
    echo "[LAUNCH] expected_topology: rank0[dit=0,enc=1], rank1[dit=2,enc=3]"

    torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_wan_new.py \
      --wan_checkpoint_dir "${WAN_TI2V_CKPT}" \
      --output_dir "${OUTPUT_ROOT}/ti2v_stage1_openvid_local_steps${NUM_TRAIN_STEPS}" \
      --frame_num "${TI2V_FRAME_NUM}" \
      --max_area "${TI2V_MAX_AREA}" \
      --min_duration_sec "${TI2V_MIN_DURATION_SEC}" \
      --num_metaqueries "${TI2V_NUM_METAQUERIES}" \
      "${COMMON_ARGS[@]}"
    ;;
  i2v)
    export WAN_DISABLE_FLASH_ATTN="${WAN_DISABLE_FLASH_ATTN:-0}"   # I2V 默认启用 flash-attn
    export WAN_FLASH_ATTN_FORCE_VERSION="${WAN_FLASH_ATTN_FORCE_VERSION:-2}"   # 强制 FA2
    export WAN_FLASH_ATTN_FALLBACK_SDPA="${WAN_FLASH_ATTN_FALLBACK_SDPA:-1}"
    export WAN_FLASH_ATTN_FORCE_CONTIGUOUS="${WAN_FLASH_ATTN_FORCE_CONTIGUOUS:-1}"
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
    export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

    echo "[LAUNCH] I2V low-vram"
    echo "[LAUNCH] NPROC_PER_NODE=${NPROC_PER_NODE} GPUS_PER_PROCESS=${GPUS_PER_PROCESS}"
    echo "[LAUNCH] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "[LAUNCH] WAN_DISABLE_FLASH_ATTN=${WAN_DISABLE_FLASH_ATTN} WAN_FLASH_ATTN_FORCE_VERSION=${WAN_FLASH_ATTN_FORCE_VERSION}"
    echo "[LAUNCH] WAN_FLASH_ATTN_FALLBACK_SDPA=${WAN_FLASH_ATTN_FALLBACK_SDPA} CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}"
    echo "[LAUNCH] expected_topology: rank0[dit=0,enc=1], rank1[dit=2,enc=3]"

    torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_i2v_new.py \
      --wan_checkpoint_dir "${WAN_I2V_CKPT}" \
      --output_dir "${OUTPUT_ROOT}/i2v_stage1_openvid_local_steps${NUM_TRAIN_STEPS}" \
      --frame_num "${I2V_FRAME_NUM}" \
      --max_area "${I2V_MAX_AREA}" \
      --min_duration_sec "${I2V_MIN_DURATION_SEC}" \
      --num_metaqueries "${I2V_NUM_METAQUERIES}" \
      "${COMMON_ARGS[@]}"
    ;;
  animate)
    export WAN_DISABLE_FLASH_ATTN="${WAN_DISABLE_FLASH_ATTN:-0}"   # Animate 默认启用 flash-attn
    export WAN_FLASH_ATTN_FORCE_VERSION="${WAN_FLASH_ATTN_FORCE_VERSION:-2}"   # 强制 FA2
    export WAN_FLASH_ATTN_FALLBACK_SDPA="${WAN_FLASH_ATTN_FALLBACK_SDPA:-1}"
    export WAN_FLASH_ATTN_FORCE_CONTIGUOUS="${WAN_FLASH_ATTN_FORCE_CONTIGUOUS:-1}"
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
    export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

    echo "[LAUNCH] ANIMATE low-vram"
    echo "[LAUNCH] NPROC_PER_NODE=${NPROC_PER_NODE} GPUS_PER_PROCESS=${GPUS_PER_PROCESS}"
    echo "[LAUNCH] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "[LAUNCH] WAN_DISABLE_FLASH_ATTN=${WAN_DISABLE_FLASH_ATTN} WAN_FLASH_ATTN_FORCE_VERSION=${WAN_FLASH_ATTN_FORCE_VERSION}"
    echo "[LAUNCH] WAN_FLASH_ATTN_FALLBACK_SDPA=${WAN_FLASH_ATTN_FALLBACK_SDPA} CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}"
    echo "[LAUNCH] expected_topology: rank0[dit=0,enc=1], rank1[dit=2,enc=3]"

    torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_animate_new.py \
      --wan_checkpoint_dir "${WAN_ANIMATE_CKPT}" \
      --output_dir "${OUTPUT_ROOT}/animate_stage1_openvid_local_steps${NUM_TRAIN_STEPS}" \
      --frame_num "${ANIMATE_FRAME_NUM}" \
      --max_area "${ANIMATE_MAX_AREA}" \
      --min_duration_sec "${ANIMATE_MIN_DURATION_SEC}" \
      --num_metaqueries "${ANIMATE_NUM_METAQUERIES}" \
      "${COMMON_ARGS[@]}"
    ;;
  *)
    echo "[ERROR] TASK 必须是: ti2v | i2v | animate"
    exit 1
    ;;
esac
