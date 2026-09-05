#!/bin/bash
set -euo pipefail
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory
export http_proxy=10.130.130.6:56830
export https_proxy=10.130.130.6:56830
export HF_ENDPOINT=https://hf-mirror.com
# Generic single-GPU launcher for inference_metaquery_wan_lora.py
# Supports checkpoints containing:
#   - wan_dit_lora.*
#   - wan_dit_trainable.*
# or both together.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_wan_lora_2x80g/ti2v_overfit10_lora_2x80g_steps1100_nummq64_rank16_alpha16/checkpoint-final}"
WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
QWEN3VL_MODEL_ID="${QWEN3VL_MODEL_ID:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"

PROMPT="${PROMPT:-A cinematic portrait shot of a woman turning her head and smiling softly.}"
REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/assets/demo_ref.png}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

MODE="${MODE:-i2v}"                           # i2v | t2v
FRAME_NUM="${FRAME_NUM:-21}"                 # must be 4n+1
SIZE_W="${SIZE_W:-640}"
SIZE_H="${SIZE_H:-384}"
I2V_FORCE_SIZE="${I2V_FORCE_SIZE:-0}"        # 1=强制使用 size，不按 ref 比例改
MAX_AREA="${MAX_AREA:-147456}"
SAMPLING_STEPS="${SAMPLING_STEPS:-32}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
SHIFT="${SHIFT:-5.0}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"      # unipc | dpm++
SEED="${SEED:-42}"

NUM_METAQUERIES="${NUM_METAQUERIES:-64}"
CONNECTOR_LAYERS="${CONNECTOR_LAYERS:-24}"
MQ_CFG_SCALE="${MQ_CFG_SCALE:-2.0}"
MQ_CONTEXT_SCALE="${MQ_CONTEXT_SCALE:-4.0}"
CFG_UNCOND_MODE="${CFG_UNCOND_MODE:-encode_negative}"  # encode_negative | zero_mq

LOAD_WAN_FINETUNE="${LOAD_WAN_FINETUNE:-1}"
VERIFY_LEVEL="${VERIFY_LEVEL:-basic}"                 # none | basic | full
VERIFY_FAIL_ON_WARNING="${VERIFY_FAIL_ON_WARNING:-1}" # 1=任何加载告警直接失败
VERIFY_REPORT_PATH="${VERIFY_REPORT_PATH:-}"
VERIFY_TRAIN_BEFORE_CHECKPOINT="${VERIFY_TRAIN_BEFORE_CHECKPOINT:-}"

DEVICE="${DEVICE:-0}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-/home/liuzhirui/model/Wan2.2/inference_outputs}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/wan_lora_${MODE}_seed${SEED}_${RUN_TS}.mp4}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/inference_metaquery_wan_lora.py"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --wan_checkpoint_dir "${WAN_CHECKPOINT_DIR}"
  --qwen3vl_model_id "${QWEN3VL_MODEL_ID}"
  --prompt "${PROMPT}"
  --negative_prompt "${NEGATIVE_PROMPT}"
  --mode "${MODE}"
  --frame_num "${FRAME_NUM}"
  --size "${SIZE_W}" "${SIZE_H}"
  --max_area "${MAX_AREA}"
  --sampling_steps "${SAMPLING_STEPS}"
  --guide_scale "${GUIDE_SCALE}"
  --mq_cfg_scale "${MQ_CFG_SCALE}"
  --mq_context_scale "${MQ_CONTEXT_SCALE}"
  --cfg_uncond_mode "${CFG_UNCOND_MODE}"
  --shift "${SHIFT}"
  --sample_solver "${SAMPLE_SOLVER}"
  --seed "${SEED}"
  --num_metaqueries "${NUM_METAQUERIES}"
  --connector_num_hidden_layers "${CONNECTOR_LAYERS}"
  --verify_level "${VERIFY_LEVEL}"
  --device "${DEVICE}"
  --output_path "${OUTPUT_PATH}"
)

if [[ -n "${VERIFY_REPORT_PATH}" ]]; then
  CMD+=(--verify_report_path "${VERIFY_REPORT_PATH}")
fi

if [[ -n "${VERIFY_TRAIN_BEFORE_CHECKPOINT}" ]]; then
  CMD+=(--verify_train_before_checkpoint "${VERIFY_TRAIN_BEFORE_CHECKPOINT}")
fi

if [[ "${VERIFY_FAIL_ON_WARNING}" == "1" ]]; then
  CMD+=(--verify_fail_on_warning)
fi

if [[ "${LOAD_WAN_FINETUNE}" == "1" ]]; then
  CMD+=(--load_wan_finetune)
else
  CMD+=(--disable_load_wan_finetune)
fi

if [[ "${MODE}" == "i2v" ]]; then
  CMD+=(--ref_image "${REF_IMAGE}")
  if [[ "${I2V_FORCE_SIZE}" == "1" ]]; then
    CMD+=(--i2v_force_size)
  fi
fi

if [[ "${OFFLOAD_MODEL}" == "1" ]]; then
  CMD+=(--offload_model)
fi

echo "[LORA-INFER] checkpoint=${CHECKPOINT_PATH}"
echo "[LORA-INFER] mode=${MODE} frame_num=${FRAME_NUM} size=${SIZE_W}x${SIZE_H} i2v_force_size=${I2V_FORCE_SIZE}"
echo "[LORA-INFER] num_metaqueries=${NUM_METAQUERIES} guide=${GUIDE_SCALE} mq_cfg=${MQ_CFG_SCALE} mq_ctx=${MQ_CONTEXT_SCALE}"
echo "[LORA-INFER] load_wan_finetune=${LOAD_WAN_FINETUNE} verify_level=${VERIFY_LEVEL} fail_on_warning=${VERIFY_FAIL_ON_WARNING}"
echo "[LORA-INFER] output=${OUTPUT_PATH}"
"${CMD[@]}"
