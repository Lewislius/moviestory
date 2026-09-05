#!/bin/bash
set -euo pipefail

# 2x96G 专用训练脚本（OpenVid local, stage1）
# 目标：在 2 张大显存卡上稳定跑通 MetaQuery + Wan 训练
#
# 默认拓扑:
#   NPROC_PER_NODE=1
#   GPUS_PER_PROCESS=2
# 即单进程双卡分工：dit_device=0, encoder_device=1（由 auto_device_map 解析）
#
# 用法:
#   TASK=animate bash train_stage1_openvid_local_metaquery_with_wan_2x96g.sh
#   TASK=animate PROFILE=aggressive NUM_TRAIN_STEPS=5000 bash ...
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# 网络与缓存
export http_proxy=10.130.130.6:56830
export https_proxy=10.130.130.6:56830
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export WAN_DIST_PROCESS_DEVICE="${WAN_DIST_PROCESS_DEVICE:-encoder}"

# Flash-attn 配置（默认打开，失败可手动 WAN_DISABLE_FLASH_ATTN=1）
# export WAN_DISABLE_FLASH_ATTN="${WAN_DISABLE_FLASH_ATTN:-0}"
export WAN_DISABLE_FLASH_ATTN="${WAN_DISABLE_FLASH_ATTN:-1}"
export WAN_FLASH_ATTN_FORCE_VERSION="${WAN_FLASH_ATTN_FORCE_VERSION:-2}"
export WAN_FLASH_ATTN_FALLBACK_SDPA="${WAN_FLASH_ATTN_FALLBACK_SDPA:-1}"
export WAN_FLASH_ATTN_FORCE_CONTIGUOUS="${WAN_FLASH_ATTN_FORCE_CONTIGUOUS:-1}"

# 数据路径
OPENVID_VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-/home/liuzhirui/dataset/OpenVid-1M/video}"
OPENVID_CSV_PATH="${OPENVID_CSV_PATH:-/home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVid-1M.csv}"
OPENVID_HD_VIDEO_ROOT="${OPENVID_HD_VIDEO_ROOT:-/home/liuzhirui/dataset/OpenVid-1M/video_HD}"
OPENVID_HD_CSV_PATH="${OPENVID_HD_CSV_PATH:-/home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVidHD.csv}"
OPENVID_LOCAL_CACHE_DIR="${OPENVID_LOCAL_CACHE_DIR:-/home/liuzhirui/dataset/OpenVid-1M/.cache_video}"
OPENVID_LOCAL_LIMIT="${OPENVID_LOCAL_LIMIT:-}"
OPENVID_HD_LOCAL_LIMIT="${OPENVID_HD_LOCAL_LIMIT:-}"
OPENVID_LOCAL_TOTAL_LIMIT="${OPENVID_LOCAL_TOTAL_LIMIT:-}"

# 模型路径
QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
WAN_TI2V_CKPT="${WAN_TI2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
WAN_I2V_CKPT="${WAN_I2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B}"
WAN_ANIMATE_CKPT="${WAN_ANIMATE_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-Animate-14B}"

# 任务/日志
TASK="${TASK:-ti2v}"                    # ti2v | i2v | animate
PROFILE="${PROFILE:-stable}"               # conservative | stable | aggressive
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_openvid_stage1_full_training}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-400}"
SAVE_STEPS="${SAVE_STEPS:-50}"
LOG_STEPS="${LOG_STEPS:-50}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

# 2x96G 默认拓扑
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
GPUS_PER_PROCESS="${GPUS_PER_PROCESS:-2}"

# W&B（可选）
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-wan-metaquery}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS="${WANDB_TAGS:-stage1,openvid,local,2x96g}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

# 采样配置（可按 PROFILE 覆盖）
case "${PROFILE}" in
  conservative)
    TI2V_FRAME_NUM_DEFAULT=13
    TI2V_MAX_AREA_DEFAULT=102400   # 320x320
    TI2V_NUM_MQ_DEFAULT=32

    I2V_FRAME_NUM_DEFAULT=17
    I2V_MAX_AREA_DEFAULT=147456    # 384x384
    I2V_NUM_MQ_DEFAULT=48

    ANIMATE_FRAME_NUM_DEFAULT=33
    ANIMATE_MAX_AREA_DEFAULT=196608  # 512x384
    ANIMATE_NUM_MQ_DEFAULT=48
    ;;
  aggressive)
    TI2V_FRAME_NUM_DEFAULT=29
    TI2V_MAX_AREA_DEFAULT=147456   # 384x384
    TI2V_NUM_MQ_DEFAULT=96

    I2V_FRAME_NUM_DEFAULT=49
    I2V_MAX_AREA_DEFAULT=262144    # 512x512
    I2V_NUM_MQ_DEFAULT=96

    ANIMATE_FRAME_NUM_DEFAULT=77
    ANIMATE_MAX_AREA_DEFAULT=262144
    ANIMATE_NUM_MQ_DEFAULT=128
    ;;
  stable)
    TI2V_FRAME_NUM_DEFAULT=33
    TI2V_MAX_AREA_DEFAULT=921600
    TI2V_NUM_MQ_DEFAULT=64

    I2V_FRAME_NUM_DEFAULT=33
    I2V_MAX_AREA_DEFAULT=196608
    I2V_NUM_MQ_DEFAULT=64

    ANIMATE_FRAME_NUM_DEFAULT=49
    ANIMATE_MAX_AREA_DEFAULT=262144
    ANIMATE_NUM_MQ_DEFAULT=64
    ;;
esac

TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-${TI2V_FRAME_NUM_DEFAULT}}"
TI2V_MAX_AREA="${TI2V_MAX_AREA:-${TI2V_MAX_AREA_DEFAULT}}"
TI2V_MIN_DURATION_SEC="${TI2V_MIN_DURATION_SEC:-0.3}"
TI2V_NUM_METAQUERIES="${TI2V_NUM_METAQUERIES:-${TI2V_NUM_MQ_DEFAULT}}"

I2V_FRAME_NUM="${I2V_FRAME_NUM:-${I2V_FRAME_NUM_DEFAULT}}"
I2V_MAX_AREA="${I2V_MAX_AREA:-${I2V_MAX_AREA_DEFAULT}}"
I2V_MIN_DURATION_SEC="${I2V_MIN_DURATION_SEC:-0.3}"
I2V_NUM_METAQUERIES="${I2V_NUM_METAQUERIES:-${I2V_NUM_MQ_DEFAULT}}"

ANIMATE_FRAME_NUM="${ANIMATE_FRAME_NUM:-${ANIMATE_FRAME_NUM_DEFAULT}}"
ANIMATE_MAX_AREA="${ANIMATE_MAX_AREA:-${ANIMATE_MAX_AREA_DEFAULT}}"
ANIMATE_MIN_DURATION_SEC="${ANIMATE_MIN_DURATION_SEC:-0.3}"
ANIMATE_NUM_METAQUERIES="${ANIMATE_NUM_METAQUERIES:-${ANIMATE_NUM_MQ_DEFAULT}}"

# smoke test 开关
SMOKE_TEST_1000="${SMOKE_TEST_1000:-0}"
if [[ "${SMOKE_TEST_1000}" == "1" ]]; then
  OPENVID_LOCAL_LIMIT="${OPENVID_LOCAL_LIMIT:-1000}"
  OPENVID_LOCAL_TOTAL_LIMIT="${OPENVID_LOCAL_TOTAL_LIMIT:-1000}"
  OPENVID_HD_VIDEO_ROOT=""
  OPENVID_HD_CSV_PATH=""
  OPENVID_HD_LOCAL_LIMIT=""
fi
if [[ -n "${OPENVID_LOCAL_TOTAL_LIMIT}" ]]; then
  export OPENVID_LOCAL_TOTAL_LIMIT
fi

# 自动检查可见 GPU 数，避免 2 卡机器误配成 4 卡拓扑
VISIBLE_GPU_COUNT=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  VISIBLE_GPU_COUNT="${#_GPU_ARR[@]}"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    VISIBLE_GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
  fi
fi
if [[ "${VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ ]] && (( VISIBLE_GPU_COUNT > 0 )); then
  REQUIRED=$(( NPROC_PER_NODE * GPUS_PER_PROCESS ))
  if (( REQUIRED > VISIBLE_GPU_COUNT )); then
    echo "[WARN] required_gpus=${REQUIRED} > visible_gpus=${VISIBLE_GPU_COUNT}, 自动降级拓扑"
    if (( VISIBLE_GPU_COUNT >= 2 )); then
      NPROC_PER_NODE=1
      GPUS_PER_PROCESS=2
    else
      NPROC_PER_NODE=1
      GPUS_PER_PROCESS=1
    fi
  fi
fi

echo "[LAUNCH] TASK=${TASK} PROFILE=${PROFILE}"
echo "[LAUNCH] NPROC_PER_NODE=${NPROC_PER_NODE} GPUS_PER_PROCESS=${GPUS_PER_PROCESS}"
echo "[LAUNCH] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} visible_gpu_count=${VISIBLE_GPU_COUNT}"
echo "[LAUNCH] flash_attn_disable=${WAN_DISABLE_FLASH_ATTN} force_ver=${WAN_FLASH_ATTN_FORCE_VERSION} fallback_sdpa=${WAN_FLASH_ATTN_FALLBACK_SDPA}"
echo "[LAUNCH] steps=${NUM_TRAIN_STEPS} save_steps=${SAVE_STEPS} log_steps=${LOG_STEPS} grad_accum=${GRAD_ACCUM_STEPS}"

COMMON_ARGS=(
  --distributed
  --auto_device_map
  --gpus_per_process "${GPUS_PER_PROCESS}"
  --hf_stage stage1
  --hf_no_streaming
  --t5_cpu
  --mq_gradient_checkpointing
  --aggressive_empty_cache
  --local_openvid_video_root "${OPENVID_VIDEO_ROOT}"
  --local_openvid_csv_path "${OPENVID_CSV_PATH}"
  --local_openvid_hd_video_root "${OPENVID_HD_VIDEO_ROOT}"
  --local_openvid_hd_csv_path "${OPENVID_HD_CSV_PATH}"
  --local_video_cache_dir "${OPENVID_LOCAL_CACHE_DIR}"
  --qwen3vl_model_id "${QWEN_MODEL}"
  --num_train_steps "${NUM_TRAIN_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --log_steps "${LOG_STEPS}"
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
)

if [[ -n "${OPENVID_LOCAL_LIMIT}" ]]; then
  COMMON_ARGS+=(--local_openvid_limit "${OPENVID_LOCAL_LIMIT}")
fi
if [[ -n "${OPENVID_HD_LOCAL_LIMIT}" ]]; then
  COMMON_ARGS+=(--local_openvid_hd_limit "${OPENVID_HD_LOCAL_LIMIT}")
fi

if [[ "${WANDB_ENABLED}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
  COMMON_ARGS+=(--wandb_enabled)
  COMMON_ARGS+=(--wandb_project "${WANDB_PROJECT}")
  COMMON_ARGS+=(--wandb_mode "${WANDB_MODE}")
  COMMON_ARGS+=(--wandb_api_key "${WANDB_API_KEY}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    COMMON_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_TAGS}" ]]; then
    COMMON_ARGS+=(--wandb_tags "${WANDB_TAGS}")
  fi
  if [[ -n "${WANDB_RUN_NAME}" ]]; then
    COMMON_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
  else
    COMMON_ARGS+=(--wandb_run_name "mq-${TASK}-stage1-openvid-local-${PROFILE}-$(date +%Y%m%d-%H%M%S)")
  fi
fi

case "${TASK}" in
  ti2v)
    torchrun --nproc_per_node "${NPROC_PER_NODE}" "/home/liuzhirui/model/Wan2.2/train_metaquery_wan_new.py" \
      --wan_checkpoint_dir "${WAN_TI2V_CKPT}" \
      --output_dir "${OUTPUT_ROOT}/ti2v_stage1_openvid_local_${PROFILE}_steps${NUM_TRAIN_STEPS}" \
      --frame_num "${TI2V_FRAME_NUM}" \
      --max_area "${TI2V_MAX_AREA}" \
      --min_duration_sec "${TI2V_MIN_DURATION_SEC}" \
      --num_metaqueries "${TI2V_NUM_METAQUERIES}" \
      "${COMMON_ARGS[@]}"
    ;;
  i2v)
    torchrun --nproc_per_node "${NPROC_PER_NODE}" "/home/liuzhirui/model/Wan2.2/train_metaquery_i2v_new.py" \
      --wan_checkpoint_dir "${WAN_I2V_CKPT}" \
      --output_dir "${OUTPUT_ROOT}/i2v_stage1_openvid_local_${PROFILE}_steps${NUM_TRAIN_STEPS}" \
      --frame_num "${I2V_FRAME_NUM}" \
      --max_area "${I2V_MAX_AREA}" \
      --min_duration_sec "${I2V_MIN_DURATION_SEC}" \
      --num_metaqueries "${I2V_NUM_METAQUERIES}" \
      "${COMMON_ARGS[@]}"
    ;;
  animate)
    torchrun --nproc_per_node "${NPROC_PER_NODE}" "/home/liuzhirui/model/Wan2.2/train_metaquery_animate_new.py" \
      --wan_checkpoint_dir "${WAN_ANIMATE_CKPT}" \
      --output_dir "${OUTPUT_ROOT}/animate_stage1_openvid_local_${PROFILE}_steps${NUM_TRAIN_STEPS}" \
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

