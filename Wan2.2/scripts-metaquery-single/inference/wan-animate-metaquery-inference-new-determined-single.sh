#!/bin/bash
set -euo pipefail
# Single-GPU inference launcher for inference_metaquery_animate.py
# eval "$(conda shell.bash hook)"
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory
export http_proxy=10.130.130.6:56830
export https_proxy=10.130.130.6:56830
export HF_ENDPOINT=https://hf-mirror.com

PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_openvid_stage1_full_training/animate_stage1_openvid_local/checkpoint-final}"
WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-Animate-14B}"
QWEN3VL_MODEL_ID="${QWEN3VL_MODEL_ID:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"

PROMPT="${PROMPT:-A girl is dancing.}"
REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/SCAIL/examples/004/ref.jpg}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

FRAME_NUM="${FRAME_NUM:-49}"     # must be 4n+1
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
SAMPLING_STEPS="${SAMPLING_STEPS:-40}"
GUIDE_SCALE="${GUIDE_SCALE:-1.0}"
SHIFT="${SHIFT:-5.0}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"   # unipc | dpm++
SEED="${SEED:-42}"

NUM_METAQUERIES="${NUM_METAQUERIES:-32}"
CONNECTOR_LAYERS="${CONNECTOR_LAYERS:-24}"
DEVICE="${DEVICE:-auto}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-1}"       # 1=enable --offload_model
MQ_REF_ONLY="${MQ_REF_ONLY:-0}"           # 1=ref image only for MQ encoder, not Wan y-condition
NO_REF_CONDITION="${NO_REF_CONDITION:-0}" # 1=ref image used by neither MQ nor Wan

OUTPUT_DIR="${OUTPUT_DIR:-/home/liuzhirui/model/Wan2.2/inference_outputs_step20}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/animate_single_seed${SEED}_${RUN_TS}.mp4}"

CMD=(
  "${PYTHON_BIN}" "/home/liuzhirui/model/Wan2.2/inference_metaquery_animate.py"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --wan_checkpoint_dir "${WAN_CHECKPOINT_DIR}"
  --qwen3vl_model_id "${QWEN3VL_MODEL_ID}"
  --prompt "${PROMPT}"
  --ref_image "${REF_IMAGE}"
  --negative_prompt "${NEGATIVE_PROMPT}"
  --frame_num "${FRAME_NUM}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --sampling_steps "${SAMPLING_STEPS}"
  --guide_scale "${GUIDE_SCALE}"
  --shift "${SHIFT}"
  --sample_solver "${SAMPLE_SOLVER}"
  --seed "${SEED}"
  --num_metaqueries "${NUM_METAQUERIES}"
  --connector_num_hidden_layers "${CONNECTOR_LAYERS}"
  # --device "${DEVICE}"
  --output_path "${OUTPUT_PATH}"
)

if [[ "${DEVICE}" != "auto" ]]; then
  CMD+=(--device "${DEVICE}")
fi
if [[ "${OFFLOAD_MODEL}" == "1" ]]; then
  CMD+=(--offload_model)
fi
if [[ "${MQ_REF_ONLY}" == "1" ]]; then
  CMD+=(--mq_ref_only)
fi
if [[ "${NO_REF_CONDITION}" == "1" ]]; then
  CMD+=(--no_ref_condition)
fi

echo "[SINGLE-ANIMATE] device=${DEVICE} frame_num=${FRAME_NUM} size=${WIDTH}x${HEIGHT} steps=${SAMPLING_STEPS}"
echo "[SINGLE-ANIMATE] mq_ref_only=${MQ_REF_ONLY}"
echo "[SINGLE-ANIMATE] no_ref_condition=${NO_REF_CONDITION}"
echo "[SINGLE-ANIMATE] output=${OUTPUT_PATH}"
"${CMD[@]}"
