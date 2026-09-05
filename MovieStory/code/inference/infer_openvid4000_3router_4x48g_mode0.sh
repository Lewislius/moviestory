#!/usr/bin/env bash
set -euo pipefail

INFERENCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${INFERENCE_ROOT}/.." && pwd)"
INFERENCE_TMPDIR="${INFERENCE_TMPDIR:-${CODE_ROOT}/tmp/inference_runtime}"
mkdir -p "${INFERENCE_TMPDIR}"
export TMPDIR="${INFERENCE_TMPDIR}"
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WAN_FLASH_ATTN_FORCE_VERSION=2
export WAN_FLASH_ATTN_FORCE_CONTIGUOUS=1

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CODE_ROOT}/checkpoint/three_router_mq-replaces-t5_strongbind_openvid4000_4x48g_steps150/checkpoint-final}"
WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
QWEN3VL_MODEL_ID="${QWEN3VL_MODEL_ID:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"

REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref5.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/动画图片/sango1.jpg}"
# PROMPT="${PROMPT:-The video is a close-up of a woman with long brown hair smiling at the camera. She turns her head naturally while the background remains stable.}"
PROMPT="${PROMPT:-The video shows the woman smiles and waves.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

# These defaults reproduce the training spatial/temporal contract.
FRAME_NUM="${FRAME_NUM:-49}"
MAX_AREA="${MAX_AREA:-262144}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
GUIDE_SCALE="${GUIDE_SCALE:-1.0}"
SHIFT="${SHIFT:-5.0}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"
SEED="${SEED:-42}"
FPS="${FPS:-24}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-0}"
ENCODER_DEVICE="${INFERENCE_DEVICE}"
DIT_DEVICE="${INFERENCE_DEVICE}"
RUNTIME_AUDIT="${RUNTIME_AUDIT:-full}"
AUDIT_FORWARD_RETRIES="${AUDIT_FORWARD_RETRIES:-1}"
AUDIT_GROWTH_LIMIT="${AUDIT_GROWTH_LIMIT:-20.0}"
MAX_NOISE_HIGH_FREQUENCY_RATIO="${MAX_NOISE_HIGH_FREQUENCY_RATIO:-0.9}"
MIN_REFERENCE_FRAME_CORRELATION="${MIN_REFERENCE_FRAME_CORRELATION:-0.5}"
MAX_REFERENCE_FRAME_MAE="${MAX_REFERENCE_FRAME_MAE:-0.4}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-1}"
T5_CPU="${T5_CPU:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
PARSE_ONLY="${PARSE_ONLY:-0}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_ROOT}/inference_outputs/openvid4000_3router_4x48g_mode0_strongbind}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/seed${SEED}_${RUN_TS}.mp4}"
VERIFY_REPORT_PATH="${VERIFY_REPORT_PATH:-${OUTPUT_PATH}.verify.json}"

cmd=(
  python "${INFERENCE_ROOT}/infer_3router_wan_4x48g_mode0.py"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --wan_checkpoint_dir "${WAN_CHECKPOINT_DIR}"
  --qwen3vl_model_id "${QWEN3VL_MODEL_ID}"
)

if [[ "${PARSE_ONLY}" == "1" ]]; then
  cmd+=(--parse_only)
  exec "${cmd[@]}"
fi
if [[ "${CHECK_ONLY}" == "1" ]]; then
  cmd+=(--check_only)
  exec "${cmd[@]}"
fi

if [[ ! -f "${REF_IMAGE}" ]]; then
  echo "[ERROR] Reference image not found: ${REF_IMAGE}" >&2
  exit 2
fi
if [[ -z "${PROMPT// }" ]]; then
  echo "[ERROR] PROMPT must be non-empty" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cmd+=(
  --prompt "${PROMPT}"
  --negative_prompt "${NEGATIVE_PROMPT}"
  --ref_image "${REF_IMAGE}"
  --output_path "${OUTPUT_PATH}"
  --verify_report_path "${VERIFY_REPORT_PATH}"
  --frame_num "${FRAME_NUM}"
  --max_area "${MAX_AREA}"
  --sampling_steps "${SAMPLING_STEPS}"
  --guide_scale "${GUIDE_SCALE}"
  --cfg_uncond_mode empty_mq
  --first_frame_mode preserved
  --shift "${SHIFT}"
  --sample_solver "${SAMPLE_SOLVER}"
  --seed "${SEED}"
  --fps "${FPS}"
  --encoder_device "${ENCODER_DEVICE}"
  --dit_device "${DIT_DEVICE}"
  --runtime_audit "${RUNTIME_AUDIT}"
  --audit_forward_retries "${AUDIT_FORWARD_RETRIES}"
  --audit_growth_limit "${AUDIT_GROWTH_LIMIT}"
  --max_noise_high_frequency_ratio "${MAX_NOISE_HIGH_FREQUENCY_RATIO}"
  --min_reference_frame_correlation "${MIN_REFERENCE_FRAME_CORRELATION}"
  --max_reference_frame_mae "${MAX_REFERENCE_FRAME_MAE}"
)

if [[ "${OFFLOAD_MODEL}" == "1" ]]; then
  cmd+=(--offload_model)
else
  cmd+=(--no-offload_model)
fi
if [[ "${T5_CPU}" == "1" ]]; then
  cmd+=(--t5_cpu)
else
  cmd+=(--no-t5_cpu)
fi

printf '[RUN] checkpoint=%s\n' "${CHECKPOINT_DIR}"
printf '[RUN] device=cuda:%s (Qwen encoder, connector, and Wan DiT)\n' "${INFERENCE_DEVICE}"
printf '[RUN] conditioning=three-router MQ(256), frozen-T5 RMS reference only, no T5 tokens in DiT\n'
printf '[RUN] reference=strongbind clean timestep-zero first slot, retained for VAE decode\n'
printf '[RUN] attention=flash_attention_2 (training-aligned) audit_retries=%s\n' "${AUDIT_FORWARD_RETRIES}"
printf '[RUN] output=%s\n' "${OUTPUT_PATH}"
printf '[RUN] verify_report=%s\n' "${VERIFY_REPORT_PATH}"
exec "${cmd[@]}"
