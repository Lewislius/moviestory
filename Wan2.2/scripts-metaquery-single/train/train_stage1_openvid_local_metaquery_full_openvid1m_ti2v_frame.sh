#!/bin/bash
set -euo pipefail

# Usage:
#   bash train_stage1_openvid_local_metaquery_full_openvid1m_ti2v_frame.sh
#
# Goal:
#   Full-data TI2V training on local OpenVid-1M + OpenVidHD, while also
#   enabling WAN-side first-frame conditioning during training.
#   This file is intentionally verbose and mostly standalone, so the original
#   train_stage1_openvid_local_metaquery_full_openvid1m.sh can stay untouched.
#
# Reference:
#   - base launcher:
#       train_stage1_openvid_local_metaquery_full_openvid1m.sh
#   - WAN first-frame conditioning style:
#       train_stage1_openvid_local_metaquery_overfit20_ti2v_frame.sh

source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export http_proxy="${http_proxy:-10.130.130.6:56830}"
export https_proxy="${https_proxy:-10.130.130.6:56830}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"

TASK="${TASK:-ti2v}"
if [[ "${TASK}" != "ti2v" ]]; then
  echo "[ERROR] This launcher is TI2V-only because WAN first-frame conditioning is only wired here."
  echo "[ERROR] Current TASK=${TASK}"
  exit 3
fi

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1200}"
SAVE_STEPS="${SAVE_STEPS:-600}"
LOG_STEPS="${LOG_STEPS:-1}"
LOG_EVERY_STEP="${LOG_EVERY_STEP:-1}"
LOG_CUDA_MEMORY="${LOG_CUDA_MEMORY:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
GPUS_PER_PROCESS="${GPUS_PER_PROCESS:-1}"
ENABLE_DISTRIBUTED_RUNTIME="${ENABLE_DISTRIBUTED_RUNTIME:-0}"
SEED="${SEED:-42}"

BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
COOLDOWN_STEPS="${COOLDOWN_STEPS:-${WARMUP_STEPS}}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-warmup_hold_cooldown}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.01}"

ENABLE_LOSS_EARLY_STOP="${ENABLE_LOSS_EARLY_STOP:-1}"
LOSS_EARLY_STOP_MIN_STEP="${LOSS_EARLY_STOP_MIN_STEP:-800}"
LOSS_EARLY_STOP_THRESHOLD="${LOSS_EARLY_STOP_THRESHOLD:-0.25}"

OPENVID_VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video}"
OPENVID_CSV_PATH="${OPENVID_CSV_PATH:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/data/train/OpenVid-1M.csv}"
OPENVID_HD_VIDEO_ROOT="${OPENVID_HD_VIDEO_ROOT:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video_HD}"
OPENVID_HD_CSV_PATH="${OPENVID_HD_CSV_PATH:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/data/train/OpenVidHD.csv}"
OPENVID_LOCAL_LIMIT="${OPENVID_LOCAL_LIMIT:-}"
OPENVID_HD_LOCAL_LIMIT="${OPENVID_HD_LOCAL_LIMIT:-}"
export OPENVID_LOCAL_TOTAL_LIMIT="${OPENVID_LOCAL_TOTAL_LIMIT:-}"

QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
WAN_TI2V_CKPT="${WAN_TI2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_openvid1m_full_ti2v_frame}"
OUTPUT_ROOT="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${OUTPUT_ROOT}")"
mkdir -p "${OUTPUT_ROOT}"

if [[ ! -d "${OPENVID_VIDEO_ROOT}" ]]; then
  echo "[ERROR] OPENVID_VIDEO_ROOT not found: ${OPENVID_VIDEO_ROOT}"
  exit 1
fi
if [[ ! -f "${OPENVID_CSV_PATH}" ]]; then
  echo "[ERROR] OPENVID_CSV_PATH not found: ${OPENVID_CSV_PATH}"
  exit 1
fi
if [[ ! -d "${OPENVID_HD_VIDEO_ROOT}" ]]; then
  echo "[ERROR] OPENVID_HD_VIDEO_ROOT not found: ${OPENVID_HD_VIDEO_ROOT}"
  exit 1
fi
if [[ ! -f "${OPENVID_HD_CSV_PATH}" ]]; then
  echo "[ERROR] OPENVID_HD_CSV_PATH not found: ${OPENVID_HD_CSV_PATH}"
  exit 1
fi

export WAN_DATA_PRECLEAN="${WAN_DATA_PRECLEAN:-1}"
export WAN_LOCAL_MISSING_CAPTION_PRINT_MAX="${WAN_LOCAL_MISSING_CAPTION_PRINT_MAX:-200}"
HF_NO_SUBSET_CACHE="${HF_NO_SUBSET_CACHE:-1}"

WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_API_KEY="${WANDB_API_KEY:-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}"
WANDB_PROJECT="${WANDB_PROJECT:-wan-metaquery-openvid1m-full-ti2v-frame}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-openvid1mfull-ti2v-frame-steps${NUM_TRAIN_STEPS}-$(date +%Y%m%d-%H%M%S)}"
WANDB_TAGS="${WANDB_TAGS:-openvid1m,openvidhd,full,ti2v,firstframe,wan-condition}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_LOG_EVERY_STEP="${WANDB_LOG_EVERY_STEP:-1}"
WANDB_LOG_CHECKPOINT="${WANDB_LOG_CHECKPOINT:-1}"

NUM_METAQUERIES="${NUM_METAQUERIES:-256}"
CONNECTOR_LAYERS="${CONNECTOR_LAYERS:-24}"
NULL_IMAGE_PROB="${NULL_IMAGE_PROB:-0.15}"
NULL_CAPTION_PROB="${NULL_CAPTION_PROB:-0.15}"

ENABLE_T5_ALIGNMENT="${ENABLE_T5_ALIGNMENT:-1}"
T5_ALIGN_MODE="${T5_ALIGN_MODE:-gram_cka}"
T5_ALIGN_ANCHOR_TOKENS="${T5_ALIGN_ANCHOR_TOKENS:-64}"
LAMBDA_T5_ALIGN_L2="${LAMBDA_T5_ALIGN_L2:-0.2}"
LAMBDA_T5_ALIGN_COS="${LAMBDA_T5_ALIGN_COS:-0.1}"
LAMBDA_T5_ALIGN_STATS="${LAMBDA_T5_ALIGN_STATS:-0.02}"
T5_ALIGN_OT_EPSILON="${T5_ALIGN_OT_EPSILON:-0.05}"
T5_ALIGN_OT_ITERS="${T5_ALIGN_OT_ITERS:-25}"

ENABLE_MQ_IMAGE_PRESERVE="${ENABLE_MQ_IMAGE_PRESERVE:-1}"
LAMBDA_MQ_IMAGE_PRESERVE="${LAMBDA_MQ_IMAGE_PRESERVE:-0.01}"
MQ_IMAGE_PRESERVE_MARGIN="${MQ_IMAGE_PRESERVE_MARGIN:-0.08}"
ENABLE_WAN_FUNC_DISTILL="${ENABLE_WAN_FUNC_DISTILL:-1}"
LAMBDA_WAN_FUNC_DISTILL="${LAMBDA_WAN_FUNC_DISTILL:-0.1}"
WAN_FUNC_TEACHER_MODE="${WAN_FUNC_TEACHER_MODE:-t5_only}"

TRAIN_MQ_INPUT_EMBEDDINGS="${TRAIN_MQ_INPUT_EMBEDDINGS:-1}"
REQUIRE_MQ_EMBEDDING_TRAIN="${REQUIRE_MQ_EMBEDDING_TRAIN:-1}"
WAN_TRAIN_MODE="${WAN_TRAIN_MODE:-cond_only}"                      # auto/full/cond_only/frozen
WAN_AUTO_FULL_MEM_GB="${WAN_AUTO_FULL_MEM_GB:-120}"
WAN_LR_RATIO="${WAN_LR_RATIO:-1.0}"
WAN_COND_NAME_PATTERN="${WAN_COND_NAME_PATTERN:-}"

TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-49}"
TI2V_MAX_AREA="${TI2V_MAX_AREA:-262144}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-0.3}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

DIT_CONDITION_MODE="${DIT_CONDITION_MODE:-mq_only}"
WAN_CONTEXT_MQ_ONLY="${WAN_CONTEXT_MQ_ONLY:-1}"

# WAN-side first-frame / preserved-prefix controls.
ENABLE_TI2V_FIRST_FRAME_CONDITION="${ENABLE_TI2V_FIRST_FRAME_CONDITION:-1}"
TRAIN_VIDEO_CONDITIONING_MODE="${TRAIN_VIDEO_CONDITIONING_MODE:-wan_animate_slot}"
TRAIN_ANIMATE_REF_FRAMES="${TRAIN_ANIMATE_REF_FRAMES:-1}"
TRAIN_ANIMATE_TEMPORAL_FRAMES="${TRAIN_ANIMATE_TEMPORAL_FRAMES:-0}"
TRAIN_ANIMATE_CONDITIONAL_FRAMES="${TRAIN_ANIMATE_CONDITIONAL_FRAMES:-0}"
TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO="${TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO:-1}"
TRAIN_ANIMATE_DROP_PREFIX_LOSS="${TRAIN_ANIMATE_DROP_PREFIX_LOSS:-1}"
TRAIN_REF_ANCHOR_MODE="${TRAIN_REF_ANCHOR_MODE:-animate_like}"
TRAIN_REF_ANCHOR_ALPHA0="${TRAIN_REF_ANCHOR_ALPHA0:-0.95}"
TRAIN_REF_ANCHOR_WARMUP_RATIO="${TRAIN_REF_ANCHOR_WARMUP_RATIO:-0.35}"

if [[ "${WAN_TRAIN_MODE}" != "frozen" && "${NPROC_PER_NODE}" != "1" ]]; then
  echo "[WARN] Wan train mode(${WAN_TRAIN_MODE}) currently supports single process only; force NPROC_PER_NODE=1"
  NPROC_PER_NODE="1"
  GPUS_PER_PROCESS="1"
fi
if [[ "${LR_SCHEDULER_TYPE}" == "warmup_hold_cooldown" ]]; then
  :
fi

METRICS_JSONL_PATH="${METRICS_JSONL_PATH:-${OUTPUT_ROOT}/logs/ti2v_openvid1mfull_frame_steps${NUM_TRAIN_STEPS}_$(date +%Y%m%d-%H%M%S).jsonl}"
METRICS_JSONL_PATH="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${METRICS_JSONL_PATH}")"

RUN_OUTPUT_DIR="${OUTPUT_ROOT}/ti2v_openvid1mfull_frame_steps${NUM_TRAIN_STEPS}_nummq${NUM_METAQUERIES}_nullimg${NULL_IMAGE_PROB}_nullcap${NULL_CAPTION_PROB}"

COMMON_ARGS=(
  --auto_device_map
  --gpus_per_process "${GPUS_PER_PROCESS}"
  --hf_stage stage1
  --hf_no_streaming
  --t5_cpu
  --mq_gradient_checkpointing
  --aggressive_empty_cache
  --wan_checkpoint_dir "${WAN_TI2V_CKPT}"
  --output_dir "${RUN_OUTPUT_DIR}"
  --qwen3vl_model_id "${QWEN_MODEL}"
  --dit_condition_mode "${DIT_CONDITION_MODE}"
  --local_openvid_video_root "${OPENVID_VIDEO_ROOT}"
  --local_openvid_csv_path "${OPENVID_CSV_PATH}"
  --local_openvid_hd_video_root "${OPENVID_HD_VIDEO_ROOT}"
  --local_openvid_hd_csv_path "${OPENVID_HD_CSV_PATH}"
  --num_metaqueries "${NUM_METAQUERIES}"
  --connector_num_hidden_layers "${CONNECTOR_LAYERS}"
  --null_image_prob "${NULL_IMAGE_PROB}"
  --null_caption_prob "${NULL_CAPTION_PROB}"
  --num_train_steps "${NUM_TRAIN_STEPS}"
  --warmup_steps "${WARMUP_STEPS}"
  --cooldown_steps "${COOLDOWN_STEPS}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --lr_min_ratio "${LR_MIN_RATIO}"
  --save_steps "${SAVE_STEPS}"
  --log_steps "${LOG_STEPS}"
  --loss_early_stop_min_step "${LOSS_EARLY_STOP_MIN_STEP}"
  --loss_early_stop_threshold "${LOSS_EARLY_STOP_THRESHOLD}"
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --frame_num "${TI2V_FRAME_NUM}"
  --max_area "${TI2V_MAX_AREA}"
  --min_duration_sec "${MIN_DURATION_SEC}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --wan_train_mode "${WAN_TRAIN_MODE}"
  --wan_auto_full_mem_gb "${WAN_AUTO_FULL_MEM_GB}"
  --wan_lr_ratio "${WAN_LR_RATIO}"
  --t5_align_anchor_tokens "${T5_ALIGN_ANCHOR_TOKENS}"
  --lambda_t5_align_l2 "${LAMBDA_T5_ALIGN_L2}"
  --lambda_t5_align_cos "${LAMBDA_T5_ALIGN_COS}"
  --lambda_t5_align_stats "${LAMBDA_T5_ALIGN_STATS}"
  --t5_align_mode "${T5_ALIGN_MODE}"
  --t5_align_ot_epsilon "${T5_ALIGN_OT_EPSILON}"
  --t5_align_ot_iters "${T5_ALIGN_OT_ITERS}"
  --lambda_mq_image_preserve "${LAMBDA_MQ_IMAGE_PRESERVE}"
  --mq_image_preserve_margin "${MQ_IMAGE_PRESERVE_MARGIN}"
  --lambda_wan_func_distill "${LAMBDA_WAN_FUNC_DISTILL}"
  --wan_func_teacher_mode "${WAN_FUNC_TEACHER_MODE}"
  --train_video_conditioning_mode "${TRAIN_VIDEO_CONDITIONING_MODE}"
  --train_animate_ref_frames "${TRAIN_ANIMATE_REF_FRAMES}"
  --train_animate_temporal_frames "${TRAIN_ANIMATE_TEMPORAL_FRAMES}"
  --train_animate_conditional_frames "${TRAIN_ANIMATE_CONDITIONAL_FRAMES}"
  --train_ref_anchor_mode "${TRAIN_REF_ANCHOR_MODE}"
  --train_ref_anchor_alpha0 "${TRAIN_REF_ANCHOR_ALPHA0}"
  --train_ref_anchor_warmup_ratio "${TRAIN_REF_ANCHOR_WARMUP_RATIO}"
  --seed "${SEED}"
  --metrics_jsonl_path "${METRICS_JSONL_PATH}"
)

if [[ "${ENABLE_DISTRIBUTED_RUNTIME}" == "1" || "${NPROC_PER_NODE}" != "1" ]]; then
  COMMON_ARGS+=(--distributed)
fi

if [[ "${HF_NO_SUBSET_CACHE}" == "1" ]]; then
  COMMON_ARGS+=(--hf_no_subset_cache)
fi
if [[ -n "${OPENVID_LOCAL_LIMIT}" ]]; then
  COMMON_ARGS+=(--local_openvid_limit "${OPENVID_LOCAL_LIMIT}")
fi
if [[ -n "${OPENVID_HD_LOCAL_LIMIT}" ]]; then
  COMMON_ARGS+=(--local_openvid_hd_limit "${OPENVID_HD_LOCAL_LIMIT}")
fi
if [[ -n "${WAN_COND_NAME_PATTERN}" ]]; then
  COMMON_ARGS+=(--wan_cond_name_pattern "${WAN_COND_NAME_PATTERN}")
fi

if [[ "${TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO}" == "1" ]]; then
  COMMON_ARGS+=(--train_animate_preserve_timestep_zero)
else
  COMMON_ARGS+=(--train_animate_no_preserve_timestep_zero)
fi

if [[ "${TRAIN_ANIMATE_DROP_PREFIX_LOSS}" == "1" ]]; then
  COMMON_ARGS+=(--train_animate_drop_prefix_loss)
else
  COMMON_ARGS+=(--train_animate_no_drop_prefix_loss)
fi

if [[ "${TRAIN_MQ_INPUT_EMBEDDINGS}" == "1" ]]; then
  COMMON_ARGS+=(--train_mq_input_embeddings)
else
  COMMON_ARGS+=(--freeze_mq_input_embeddings)
fi

if [[ "${REQUIRE_MQ_EMBEDDING_TRAIN}" == "1" && "${TRAIN_MQ_INPUT_EMBEDDINGS}" != "1" ]]; then
  echo "[FATAL] REQUIRE_MQ_EMBEDDING_TRAIN=1 but TRAIN_MQ_INPUT_EMBEDDINGS=${TRAIN_MQ_INPUT_EMBEDDINGS}"
  exit 11
fi

if [[ "${ENABLE_T5_ALIGNMENT}" == "1" ]]; then
  COMMON_ARGS+=(--enable_t5_alignment)
else
  COMMON_ARGS+=(--disable_t5_alignment)
fi

if [[ "${ENABLE_LOSS_EARLY_STOP}" == "1" ]]; then
  COMMON_ARGS+=(--enable_loss_early_stop)
else
  COMMON_ARGS+=(--disable_loss_early_stop)
fi

if [[ "${ENABLE_MQ_IMAGE_PRESERVE}" == "1" ]]; then
  COMMON_ARGS+=(--enable_mq_image_preserve)
fi

if [[ "${ENABLE_WAN_FUNC_DISTILL}" == "1" ]]; then
  COMMON_ARGS+=(--enable_wan_func_distill)
else
  COMMON_ARGS+=(--disable_wan_func_distill)
fi

if [[ "${LOG_EVERY_STEP}" == "1" ]]; then
  COMMON_ARGS+=(--log_every_step)
fi
if [[ "${LOG_CUDA_MEMORY}" == "1" ]]; then
  COMMON_ARGS+=(--log_cuda_memory)
fi
if [[ -n "${RESUME_MQ_ENCODER_PATH:-}" ]]; then
  COMMON_ARGS+=(--resume_mq_encoder_path "${RESUME_MQ_ENCODER_PATH}")
fi

if [[ "${WANDB_ENABLED}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
  COMMON_ARGS+=(--wandb_enabled)
  COMMON_ARGS+=(--wandb_project "${WANDB_PROJECT}")
  COMMON_ARGS+=(--wandb_mode "${WANDB_MODE}")
  COMMON_ARGS+=(--wandb_api_key "${WANDB_API_KEY}")
  COMMON_ARGS+=(--wandb_tags "${WANDB_TAGS}")
  COMMON_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    COMMON_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  if [[ "${WANDB_LOG_EVERY_STEP}" == "1" ]]; then
    COMMON_ARGS+=(--wandb_log_every_step)
  fi
  if [[ "${WANDB_LOG_CHECKPOINT}" == "1" ]]; then
    COMMON_ARGS+=(--wandb_log_checkpoint)
  fi
fi

TRAIN_ENTRY="${SCRIPT_DIR}/train_metaquery_wan_new.py"
export WAN_BASE_TI2V_MODULE="${WAN_BASE_TI2V_MODULE:-train_metaquery_wan}"
export WAN_CONNECTOR_FILE="${WAN_CONNECTOR_FILE:-${SCRIPT_DIR}/train_connector_for_wan.py}"

BASE_TI2V_FILE_CANDIDATE="${WAN_BASE_TI2V_FILE:-${SCRIPT_DIR}/${WAN_BASE_TI2V_MODULE}.py}"
TI2V_FIRSTFRAME_ARG_SUPPORTED=0
if [[ -f "${BASE_TI2V_FILE_CANDIDATE}" ]]; then
  if grep -q -- "--enable_ti2v_first_frame_condition" "${BASE_TI2V_FILE_CANDIDATE}"; then
    TI2V_FIRSTFRAME_ARG_SUPPORTED=1
  fi
fi
if [[ "${TI2V_FIRSTFRAME_ARG_SUPPORTED}" == "1" ]]; then
  if [[ "${ENABLE_TI2V_FIRST_FRAME_CONDITION}" == "1" ]]; then
    COMMON_ARGS+=(--enable_ti2v_first_frame_condition)
  else
    COMMON_ARGS+=(--disable_ti2v_first_frame_condition)
  fi
else
  echo "[WARN] base module does not support --enable_ti2v_first_frame_condition, skip this flag."
  echo "[WARN] checked_file=${BASE_TI2V_FILE_CANDIDATE}"
fi

echo "[LAUNCH][OPENVID1M_FULL_FRAME] TASK=${TASK} steps=${NUM_TRAIN_STEPS}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] openvid_root=${OPENVID_VIDEO_ROOT}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] openvid_csv=${OPENVID_CSV_PATH}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] openvid_hd_root=${OPENVID_HD_VIDEO_ROOT}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] openvid_hd_csv=${OPENVID_HD_CSV_PATH}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] openvid_limit=${OPENVID_LOCAL_LIMIT:-all} hd_limit=${OPENVID_HD_LOCAL_LIMIT:-all} total_limit=${OPENVID_LOCAL_TOTAL_LIMIT:-all}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] run_output_dir=${RUN_OUTPUT_DIR}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] metrics_jsonl=${METRICS_JSONL_PATH}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] WAN first-frame condition=$( [[ "${ENABLE_TI2V_FIRST_FRAME_CONDITION}" == "1" ]] && echo ENABLED || echo DISABLED )"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] video_conditioning_mode=${TRAIN_VIDEO_CONDITIONING_MODE} ref_frames=${TRAIN_ANIMATE_REF_FRAMES} temporal_frames=${TRAIN_ANIMATE_TEMPORAL_FRAMES} conditional_frames=${TRAIN_ANIMATE_CONDITIONAL_FRAMES}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] preserve_timestep_zero=${TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO} drop_prefix_loss=${TRAIN_ANIMATE_DROP_PREFIX_LOSS}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] ref_anchor_mode=${TRAIN_REF_ANCHOR_MODE} alpha0=${TRAIN_REF_ANCHOR_ALPHA0} warmup_ratio=${TRAIN_REF_ANCHOR_WARMUP_RATIO}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] WAN context mode=$( [[ "${WAN_CONTEXT_MQ_ONLY}" == "1" ]] && echo mq_only || echo mq_plus_t5 )"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] num_metaqueries=${NUM_METAQUERIES} null_image_prob=${NULL_IMAGE_PROB} null_caption_prob=${NULL_CAPTION_PROB}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] warmup=${WARMUP_STEPS} cooldown=${COOLDOWN_STEPS} lr_scheduler=${LR_SCHEDULER_TYPE} lr_min_ratio=${LR_MIN_RATIO}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] batch_size=${BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} lr=${LEARNING_RATE}"
echo "[LAUNCH][OPENVID1M_FULL_FRAME] wan_train_mode=${WAN_TRAIN_MODE} auto_full_mem_gb=${WAN_AUTO_FULL_MEM_GB} wan_lr_ratio=${WAN_LR_RATIO} wan_cond_name_pattern=${WAN_COND_NAME_PATTERN:-<default>}"

if [[ "${ENABLE_DISTRIBUTED_RUNTIME}" == "1" || "${NPROC_PER_NODE}" != "1" ]]; then
  TRAIN_LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}")
  LAUNCH_MODE="torchrun+distributed"
else
  TRAIN_LAUNCHER=("${PYTHON_BIN}")
  LAUNCH_MODE="python-single-process"
fi
echo "[LAUNCH][OPENVID1M_FULL_FRAME] launch_mode=${LAUNCH_MODE} nproc_per_node=${NPROC_PER_NODE} gpus_per_process=${GPUS_PER_PROCESS} enable_distributed_runtime=${ENABLE_DISTRIBUTED_RUNTIME}"

"${TRAIN_LAUNCHER[@]}" "${TRAIN_ENTRY}" "${COMMON_ARGS[@]}"

RUN_OUTPUT_DIR="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${RUN_OUTPUT_DIR}")"
echo "[OPENVID1M_FULL_FRAME][VERIFY] before checkpoint: ${RUN_OUTPUT_DIR}/checkpoint-before-training"
echo "[OPENVID1M_FULL_FRAME][VERIFY] final checkpoint : ${RUN_OUTPUT_DIR}/checkpoint-final"
echo "[OPENVID1M_FULL_FRAME][VERIFY] chain manifest   : ${RUN_OUTPUT_DIR}/training_chain_manifest.json"
echo "[OPENVID1M_FULL_FRAME][VERIFY] audit cmd:"
echo "python ${SCRIPT_DIR}/verify_metaquery_chain.py \\"
echo "  --before_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-before-training \\"
echo "  --after_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-final \\"
echo "  --output_json ${RUN_OUTPUT_DIR}/chain_audit_report.json --strict"
