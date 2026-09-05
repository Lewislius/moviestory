#!/bin/bash
# set -euo pipefail

# # Usage:
# #   bash train_stage1_openvid_local_metaquery_full_openvid1m.sh
# #
# # Goal:
# #   Full-data training on local OpenVid-1M + OpenVidHD, reusing the same
# #   core training hyperparameters from the overfit script.

# source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
# conda activate /home/liuzhirui/miniconda3/envs/moviestory

# export http_proxy="${http_proxy:-10.130.130.6:56830}"
# export https_proxy="${https_proxy:-10.130.130.6:56830}"
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# export HF_TOKEN="${HF_TOKEN:-}"
# export PYTHONUNBUFFERED=1
# export TORCH_NCCL_BLOCKING_WAIT=1
# export WANDB_ENABLED=1
# export WANDB_API_KEY="${WANDB_API_KEY:-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}"
# export WANDB_PROJECT="${WANDB_PROJECT:-wan-metaquery-openvid1m-full}"
# export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"

# TASK="${TASK:-ti2v}"  # ti2v | i2v | animate
# NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000}"
# SAVE_STEPS="${SAVE_STEPS:-1000}"
# LOG_STEPS="${LOG_STEPS:-1}"
# LOG_EVERY_STEP="${LOG_EVERY_STEP:-1}"
# NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
# GPUS_PER_PROCESS="${GPUS_PER_PROCESS:-1}"
# SEED="${SEED:-42}"
# BATCH_SIZE="${BATCH_SIZE:-1}"
# GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
# LEARNING_RATE="${LEARNING_RATE:-1e-5}"
# WARMUP_STEPS="${WARMUP_STEPS:-200}"
# COOLDOWN_STEPS="${COOLDOWN_STEPS:-${WARMUP_STEPS}}"
# LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-warmup_hold_cooldown}"
# LR_MIN_RATIO="${LR_MIN_RATIO:-0.01}"
# ENABLE_LOSS_EARLY_STOP="${ENABLE_LOSS_EARLY_STOP:-1}"
# LOSS_EARLY_STOP_MIN_STEP="${LOSS_EARLY_STOP_MIN_STEP:-900}"
# LOSS_EARLY_STOP_THRESHOLD="${LOSS_EARLY_STOP_THRESHOLD:-0.25}"

# OPENVID_VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video}"
# OPENVID_CSV_PATH="${OPENVID_CSV_PATH:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/data/train/OpenVid-1M.csv}"
# OPENVID_HD_VIDEO_ROOT="${OPENVID_HD_VIDEO_ROOT:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video_HD}"
# OPENVID_HD_CSV_PATH="${OPENVID_HD_CSV_PATH:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/data/train/OpenVidHD.csv}"
# OPENVID_LOCAL_LIMIT="${OPENVID_LOCAL_LIMIT:-}"
# OPENVID_HD_LOCAL_LIMIT="${OPENVID_HD_LOCAL_LIMIT:-}"
# export OPENVID_LOCAL_TOTAL_LIMIT="${OPENVID_LOCAL_TOTAL_LIMIT:-}"

# QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
# WAN_TI2V_CKPT="${WAN_TI2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
# WAN_I2V_CKPT="${WAN_I2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B}"
# WAN_ANIMATE_CKPT="${WAN_ANIMATE_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-Animate-14B}"

# OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_openvid1m_full}"
# OUTPUT_ROOT="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${OUTPUT_ROOT}")"
# mkdir -p "${OUTPUT_ROOT}"

# if [[ ! -d "${OPENVID_VIDEO_ROOT}" ]]; then
#   echo "[ERROR] OPENVID_VIDEO_ROOT not found: ${OPENVID_VIDEO_ROOT}"
#   exit 1
# fi
# if [[ ! -f "${OPENVID_CSV_PATH}" ]]; then
#   echo "[ERROR] OPENVID_CSV_PATH not found: ${OPENVID_CSV_PATH}"
#   exit 1
# fi
# if [[ ! -d "${OPENVID_HD_VIDEO_ROOT}" ]]; then
#   echo "[ERROR] OPENVID_HD_VIDEO_ROOT not found: ${OPENVID_HD_VIDEO_ROOT}"
#   exit 1
# fi
# if [[ ! -f "${OPENVID_HD_CSV_PATH}" ]]; then
#   echo "[ERROR] OPENVID_HD_CSV_PATH not found: ${OPENVID_HD_CSV_PATH}"
#   exit 1
# fi

# export WAN_DATA_PRECLEAN="${WAN_DATA_PRECLEAN:-1}"
# export WAN_LOCAL_MISSING_CAPTION_PRINT_MAX="${WAN_LOCAL_MISSING_CAPTION_PRINT_MAX:-200}"
# HF_NO_SUBSET_CACHE="${HF_NO_SUBSET_CACHE:-1}"

# WANDB_ENABLED="${WANDB_ENABLED:-0}"
# WANDB_ENTITY="${WANDB_ENTITY:-}"
# WANDB_RUN_NAME="${WANDB_RUN_NAME:-openvid1mfull-${TASK}-steps${NUM_TRAIN_STEPS}-$(date +%Y%m%d-%H%M%S)}"
# WANDB_TAGS="${WANDB_TAGS:-openvid1m,openvidhd,full}"
# WANDB_MODE="${WANDB_MODE:-offline}"

# LOG_CUDA_MEMORY="${LOG_CUDA_MEMORY:-1}"
# WANDB_LOG_EVERY_STEP="${WANDB_LOG_EVERY_STEP:-1}"
# WANDB_LOG_CHECKPOINT="${WANDB_LOG_CHECKPOINT:-1}"

# NUM_METAQUERIES="${NUM_METAQUERIES:-256}"
# NULL_IMAGE_PROB="${NULL_IMAGE_PROB:-0.1}"
# NULL_CAPTION_PROB="${NULL_CAPTION_PROB:-0.1}"
# ENABLE_T5_ALIGNMENT="${ENABLE_T5_ALIGNMENT:-1}"
# T5_ALIGN_MODE="${T5_ALIGN_MODE:-gram_cka}"
# T5_ALIGN_ANCHOR_TOKENS="${T5_ALIGN_ANCHOR_TOKENS:-64}"
# LAMBDA_T5_ALIGN_L2="${LAMBDA_T5_ALIGN_L2:-0.2}"
# LAMBDA_T5_ALIGN_COS="${LAMBDA_T5_ALIGN_COS:-0.1}"
# LAMBDA_T5_ALIGN_STATS="${LAMBDA_T5_ALIGN_STATS:-0.02}"
# T5_ALIGN_OT_EPSILON="${T5_ALIGN_OT_EPSILON:-0.05}"
# T5_ALIGN_OT_ITERS="${T5_ALIGN_OT_ITERS:-25}"
# ENABLE_MQ_IMAGE_PRESERVE="${ENABLE_MQ_IMAGE_PRESERVE:-1}"
# LAMBDA_MQ_IMAGE_PRESERVE="${LAMBDA_MQ_IMAGE_PRESERVE:-0.01}"
# MQ_IMAGE_PRESERVE_MARGIN="${MQ_IMAGE_PRESERVE_MARGIN:-0.08}"
# ENABLE_WAN_FUNC_DISTILL="${ENABLE_WAN_FUNC_DISTILL:-1}"
# LAMBDA_WAN_FUNC_DISTILL="${LAMBDA_WAN_FUNC_DISTILL:-0.1}"
# WAN_FUNC_TEACHER_MODE="${WAN_FUNC_TEACHER_MODE:-t5_only}"
# TRAIN_MQ_INPUT_EMBEDDINGS="${TRAIN_MQ_INPUT_EMBEDDINGS:-1}"  # 1=train MQ token embeddings + connector
# REQUIRE_MQ_EMBEDDING_TRAIN="${REQUIRE_MQ_EMBEDDING_TRAIN:-1}" # 1=guardrail against accidental freeze
# WAN_TRAIN_MODE="${WAN_TRAIN_MODE:-cond_only}"                      # auto/full/cond_only/frozen
# WAN_AUTO_FULL_MEM_GB="${WAN_AUTO_FULL_MEM_GB:-120}"
# WAN_LR_RATIO="${WAN_LR_RATIO:-0.4}"
# WAN_COND_NAME_PATTERN="${WAN_COND_NAME_PATTERN:-}"

# if [[ "${TASK}" == "ti2v" && "${WAN_TRAIN_MODE}" != "frozen" && "${NPROC_PER_NODE}" != "1" ]]; then
#   echo "[WARN] Wan train mode(${WAN_TRAIN_MODE}) currently supports single process only; force NPROC_PER_NODE=1"
#   NPROC_PER_NODE="1"
#   GPUS_PER_PROCESS="1"
# fi
# if [[ "${TASK}" != "ti2v" && "${LR_SCHEDULER_TYPE}" == "warmup_hold_cooldown" ]]; then
#   echo "[WARN] warmup_hold_cooldown is only implemented in ti2v; fallback to constant_with_warmup for ${TASK}"
#   LR_SCHEDULER_TYPE="constant_with_warmup"
# fi

# TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-49}"
# I2V_FRAME_NUM="${I2V_FRAME_NUM:-49}"
# ANIMATE_FRAME_NUM="${ANIMATE_FRAME_NUM:-49}"
# TI2V_MAX_AREA="${TI2V_MAX_AREA:-262144}"
# I2V_MAX_AREA="${I2V_MAX_AREA:-262144}"
# ANIMATE_MAX_AREA="${ANIMATE_MAX_AREA:-262144}"
# MIN_DURATION_SEC="${MIN_DURATION_SEC:-0.3}"
# DIT_CONDITION_MODE="${DIT_CONDITION_MODE:-mq_only}"
# TRAIN_REF_ANCHOR_MODE="${TRAIN_REF_ANCHOR_MODE:-none}"

# METRICS_JSONL_PATH="${METRICS_JSONL_PATH:-${OUTPUT_ROOT}/logs/${TASK}_openvid1mfull_steps${NUM_TRAIN_STEPS}_$(date +%Y%m%d-%H%M%S).jsonl}"
# METRICS_JSONL_PATH="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${METRICS_JSONL_PATH}")"

# COMMON_ARGS=(
#   --distributed
#   --auto_device_map
#   --gpus_per_process "${GPUS_PER_PROCESS}"
#   --hf_stage stage1
#   --hf_no_streaming
#   --t5_cpu
#   --mq_gradient_checkpointing
#   --aggressive_empty_cache
#   --local_openvid_video_root "${OPENVID_VIDEO_ROOT}"
#   --local_openvid_csv_path "${OPENVID_CSV_PATH}"
#   --local_openvid_hd_video_root "${OPENVID_HD_VIDEO_ROOT}"
#   --local_openvid_hd_csv_path "${OPENVID_HD_CSV_PATH}"
#   --qwen3vl_model_id "${QWEN_MODEL}"
#   --dit_condition_mode "${DIT_CONDITION_MODE}"
#   --num_metaqueries "${NUM_METAQUERIES}"
#   --null_image_prob "${NULL_IMAGE_PROB}"
#   --null_caption_prob "${NULL_CAPTION_PROB}"
#   --t5_align_anchor_tokens "${T5_ALIGN_ANCHOR_TOKENS}"
#   --lambda_t5_align_l2 "${LAMBDA_T5_ALIGN_L2}"
#   --lambda_t5_align_cos "${LAMBDA_T5_ALIGN_COS}"
#   --lambda_t5_align_stats "${LAMBDA_T5_ALIGN_STATS}"
#   --lambda_mq_image_preserve "${LAMBDA_MQ_IMAGE_PRESERVE}"
#   --mq_image_preserve_margin "${MQ_IMAGE_PRESERVE_MARGIN}"
#   --num_train_steps "${NUM_TRAIN_STEPS}"
#   --warmup_steps "${WARMUP_STEPS}"
#   --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
#   --lr_min_ratio "${LR_MIN_RATIO}"
#   --save_steps "${SAVE_STEPS}"
#   --log_steps "${LOG_STEPS}"
#   --loss_early_stop_min_step "${LOSS_EARLY_STOP_MIN_STEP}"
#   --loss_early_stop_threshold "${LOSS_EARLY_STOP_THRESHOLD}"
#   --batch_size "${BATCH_SIZE}"
#   --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
#   --learning_rate "${LEARNING_RATE}"
#   --seed "${SEED}"
#   --metrics_jsonl_path "${METRICS_JSONL_PATH}"
# )

# if [[ "${HF_NO_SUBSET_CACHE}" == "1" ]]; then
#   COMMON_ARGS+=(--hf_no_subset_cache)
# fi
# if [[ -n "${OPENVID_LOCAL_LIMIT}" ]]; then
#   COMMON_ARGS+=(--local_openvid_limit "${OPENVID_LOCAL_LIMIT}")
# fi
# if [[ -n "${OPENVID_HD_LOCAL_LIMIT}" ]]; then
#   COMMON_ARGS+=(--local_openvid_hd_limit "${OPENVID_HD_LOCAL_LIMIT}")
# fi

# if [[ "${TRAIN_MQ_INPUT_EMBEDDINGS}" == "1" ]]; then
#   COMMON_ARGS+=(--train_mq_input_embeddings)
# else
#   COMMON_ARGS+=(--freeze_mq_input_embeddings)
# fi

# if [[ "${TASK}" == "ti2v" ]]; then
#   COMMON_ARGS+=(--cooldown_steps "${COOLDOWN_STEPS}")
#   COMMON_ARGS+=(--wan_train_mode "${WAN_TRAIN_MODE}")
#   COMMON_ARGS+=(--wan_auto_full_mem_gb "${WAN_AUTO_FULL_MEM_GB}")
#   COMMON_ARGS+=(--wan_lr_ratio "${WAN_LR_RATIO}")
#   if [[ -n "${WAN_COND_NAME_PATTERN}" ]]; then
#     COMMON_ARGS+=(--wan_cond_name_pattern "${WAN_COND_NAME_PATTERN}")
#   fi
#   COMMON_ARGS+=(--t5_align_mode "${T5_ALIGN_MODE}")
#   COMMON_ARGS+=(--t5_align_ot_epsilon "${T5_ALIGN_OT_EPSILON}")
#   COMMON_ARGS+=(--t5_align_ot_iters "${T5_ALIGN_OT_ITERS}")
#   COMMON_ARGS+=(--lambda_wan_func_distill "${LAMBDA_WAN_FUNC_DISTILL}")
#   COMMON_ARGS+=(--wan_func_teacher_mode "${WAN_FUNC_TEACHER_MODE}")
# fi

# if [[ "${ENABLE_T5_ALIGNMENT}" == "1" ]]; then
#   COMMON_ARGS+=(--enable_t5_alignment)
# else
#   COMMON_ARGS+=(--disable_t5_alignment)
# fi

# if [[ "${ENABLE_LOSS_EARLY_STOP}" == "1" ]]; then
#   COMMON_ARGS+=(--enable_loss_early_stop)
# else
#   COMMON_ARGS+=(--disable_loss_early_stop)
# fi

# if [[ "${ENABLE_MQ_IMAGE_PRESERVE}" == "1" ]]; then
#   COMMON_ARGS+=(--enable_mq_image_preserve)
# fi

# if [[ "${ENABLE_WAN_FUNC_DISTILL}" == "1" ]]; then
#   if [[ "${TASK}" == "ti2v" ]]; then
#     COMMON_ARGS+=(--enable_wan_func_distill)
#   fi
# else
#   if [[ "${TASK}" == "ti2v" ]]; then
#     COMMON_ARGS+=(--disable_wan_func_distill)
#   fi
# fi

# if [[ "${REQUIRE_MQ_EMBEDDING_TRAIN}" == "1" && "${TRAIN_MQ_INPUT_EMBEDDINGS}" != "1" ]]; then
#   echo "[FATAL] REQUIRE_MQ_EMBEDDING_TRAIN=1 but TRAIN_MQ_INPUT_EMBEDDINGS=${TRAIN_MQ_INPUT_EMBEDDINGS}"
#   exit 11
# fi

# if [[ "${LOG_EVERY_STEP}" == "1" ]]; then
#   COMMON_ARGS+=(--log_every_step)
# fi
# if [[ "${LOG_CUDA_MEMORY}" == "1" ]]; then
#   COMMON_ARGS+=(--log_cuda_memory)
# fi
# if [[ "${WANDB_ENABLED}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
#   COMMON_ARGS+=(--wandb_enabled)
#   COMMON_ARGS+=(--wandb_project "${WANDB_PROJECT}")
#   if [[ -n "${WANDB_ENTITY}" ]]; then
#     COMMON_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
#   fi
#   COMMON_ARGS+=(--wandb_mode "${WANDB_MODE}")
#   COMMON_ARGS+=(--wandb_api_key "${WANDB_API_KEY}")
#   COMMON_ARGS+=(--wandb_tags "${WANDB_TAGS}")
#   COMMON_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
#   if [[ "${WANDB_LOG_EVERY_STEP}" == "1" ]]; then
#     COMMON_ARGS+=(--wandb_log_every_step)
#   fi
#   if [[ "${WANDB_LOG_CHECKPOINT}" == "1" ]]; then
#     COMMON_ARGS+=(--wandb_log_checkpoint)
#   fi
# fi

# echo "[LAUNCH][OPENVID1M_FULL] TASK=${TASK} steps=${NUM_TRAIN_STEPS}"
# echo "[LAUNCH][OPENVID1M_FULL] openvid_root=${OPENVID_VIDEO_ROOT}"
# echo "[LAUNCH][OPENVID1M_FULL] openvid_csv=${OPENVID_CSV_PATH}"
# echo "[LAUNCH][OPENVID1M_FULL] openvid_hd_root=${OPENVID_HD_VIDEO_ROOT}"
# echo "[LAUNCH][OPENVID1M_FULL] openvid_hd_csv=${OPENVID_HD_CSV_PATH}"
# echo "[LAUNCH][OPENVID1M_FULL] openvid_limit=${OPENVID_LOCAL_LIMIT:-all} hd_limit=${OPENVID_HD_LOCAL_LIMIT:-all} total_limit=${OPENVID_LOCAL_TOTAL_LIMIT:-all}"
# echo "[LAUNCH][OPENVID1M_FULL] metrics_jsonl=${METRICS_JSONL_PATH}"
# echo "[LAUNCH][OPENVID1M_FULL] warmup=${WARMUP_STEPS} cooldown=${COOLDOWN_STEPS} lr_scheduler=${LR_SCHEDULER_TYPE} lr_min_ratio=${LR_MIN_RATIO}"
# echo "[LAUNCH][OPENVID1M_FULL] early_stop enabled=${ENABLE_LOSS_EARLY_STOP} min_step=${LOSS_EARLY_STOP_MIN_STEP} threshold=${LOSS_EARLY_STOP_THRESHOLD}"
# echo "[LAUNCH][OPENVID1M_FULL] batch_size=${BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} lr=${LEARNING_RATE}"
# echo "[LAUNCH][OPENVID1M_FULL] num_metaqueries=${NUM_METAQUERIES} null_image_prob=${NULL_IMAGE_PROB} null_caption_prob=${NULL_CAPTION_PROB}"
# echo "[LAUNCH][OPENVID1M_FULL] dit_condition_mode=${DIT_CONDITION_MODE} train_ref_anchor_mode=${TRAIN_REF_ANCHOR_MODE}"

# case "${TASK}" in
#   ti2v)
#     RUN_OUTPUT_DIR="${OUTPUT_ROOT}/ti2v_openvid1mfull_steps${NUM_TRAIN_STEPS}_nummq${NUM_METAQUERIES}_nullimg${NULL_IMAGE_PROB}_nullcap${NULL_CAPTION_PROB}"
#     torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_wan_new.py \
#       --wan_checkpoint_dir "${WAN_TI2V_CKPT}" \
#       --output_dir "${RUN_OUTPUT_DIR}" \
#       --frame_num "${TI2V_FRAME_NUM}" \
#       --max_area "${TI2V_MAX_AREA}" \
#       --min_duration_sec "${MIN_DURATION_SEC}" \
#       --train_ref_anchor_mode "${TRAIN_REF_ANCHOR_MODE}" \
#       "${COMMON_ARGS[@]}"
#     ;;
#   i2v)
#     RUN_OUTPUT_DIR="${OUTPUT_ROOT}/i2v_openvid1mfull_steps${NUM_TRAIN_STEPS}_nummq${NUM_METAQUERIES}_nullimg${NULL_IMAGE_PROB}_nullcap${NULL_CAPTION_PROB}"
#     torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_i2v_new.py \
#       --wan_checkpoint_dir "${WAN_I2V_CKPT}" \
#       --output_dir "${RUN_OUTPUT_DIR}" \
#       --frame_num "${I2V_FRAME_NUM}" \
#       --max_area "${I2V_MAX_AREA}" \
#       --min_duration_sec "${MIN_DURATION_SEC}" \
#       "${COMMON_ARGS[@]}"
#     ;;
#   animate)
#     RUN_OUTPUT_DIR="${OUTPUT_ROOT}/animate_openvid1mfull_steps${NUM_TRAIN_STEPS}"
#     torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_animate_new.py \
#       --wan_checkpoint_dir "${WAN_ANIMATE_CKPT}" \
#       --output_dir "${RUN_OUTPUT_DIR}" \
#       --frame_num "${ANIMATE_FRAME_NUM}" \
#       --max_area "${ANIMATE_MAX_AREA}" \
#       --min_duration_sec "${MIN_DURATION_SEC}" \
#       "${COMMON_ARGS[@]}"
#     ;;
#   *)
#     echo "[ERROR] TASK must be one of: ti2v | i2v | animate"
#     exit 3
#     ;;
# esac

# RUN_OUTPUT_DIR="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${RUN_OUTPUT_DIR}")"
# echo "[OPENVID1M_FULL][VERIFY] before checkpoint: ${RUN_OUTPUT_DIR}/checkpoint-before-training"
# echo "[OPENVID1M_FULL][VERIFY] final checkpoint : ${RUN_OUTPUT_DIR}/checkpoint-final"
# echo "[OPENVID1M_FULL][VERIFY] chain manifest   : ${RUN_OUTPUT_DIR}/training_chain_manifest.json"
# echo "[OPENVID1M_FULL][VERIFY] audit cmd:"
# echo "python /home/liuzhirui/model/Wan2.2/verify_metaquery_chain.py \\"
# echo "  --before_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-before-training \\"
# echo "  --after_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-final \\"
# echo "  --output_json ${RUN_OUTPUT_DIR}/chain_audit_report.json --strict"

set -euo pipefail

# Usage:
#   bash train_stage1_openvid_local_metaquery_full_openvid1m.sh
#
# Goal:
#   Full-data training on local OpenVid-1M + OpenVidHD, reusing the same
#   core training hyperparameters from the overfit script.

source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

export http_proxy="${http_proxy:-10.130.130.6:56830}"
export https_proxy="${https_proxy:-10.130.130.6:56830}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=1
export WANDB_ENABLED=1
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_PROJECT="${WANDB_PROJECT:-wan-metaquery-openvid1m-full}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"

TASK="${TASK:-ti2v}"  # ti2v | i2v | animate
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-4000}"
SAVE_STEPS="${SAVE_STEPS:-1200}"
LOG_STEPS="${LOG_STEPS:-1}"
LOG_EVERY_STEP="${LOG_EVERY_STEP:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
GPUS_PER_PROCESS="${GPUS_PER_PROCESS:-1}"
ENABLE_DISTRIBUTED_RUNTIME="${ENABLE_DISTRIBUTED_RUNTIME:-0}"  # 单卡默认不初始化 torch.distributed / NCCL
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
COOLDOWN_STEPS="${COOLDOWN_STEPS:-${WARMUP_STEPS}}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-warmup_hold_cooldown}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.01}"
ENABLE_LOSS_EARLY_STOP="${ENABLE_LOSS_EARLY_STOP:-1}"
LOSS_EARLY_STOP_MIN_STEP="${LOSS_EARLY_STOP_MIN_STEP:-3980}"
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
WAN_I2V_CKPT="${WAN_I2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B}"
WAN_ANIMATE_CKPT="${WAN_ANIMATE_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-Animate-14B}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_openvid1m_full}"
OUTPUT_ROOT="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${OUTPUT_ROOT}")"
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

WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-openvid1mfull-${TASK}-steps${NUM_TRAIN_STEPS}-$(date +%Y%m%d-%H%M%S)}"
WANDB_TAGS="${WANDB_TAGS:-openvid1m,openvidhd,full}"
WANDB_MODE="${WANDB_MODE:-offline}"

LOG_CUDA_MEMORY="${LOG_CUDA_MEMORY:-1}"
WANDB_LOG_EVERY_STEP="${WANDB_LOG_EVERY_STEP:-1}"
WANDB_LOG_CHECKPOINT="${WANDB_LOG_CHECKPOINT:-1}"

NUM_METAQUERIES="${NUM_METAQUERIES:-256}"
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
TRAIN_MQ_INPUT_EMBEDDINGS="${TRAIN_MQ_INPUT_EMBEDDINGS:-1}"  # 1=train MQ token embeddings + connector
REQUIRE_MQ_EMBEDDING_TRAIN="${REQUIRE_MQ_EMBEDDING_TRAIN:-1}" # 1=guardrail against accidental freeze
WAN_TRAIN_MODE="${WAN_TRAIN_MODE:-cond_only}"                      # auto/full/cond_only/frozen
WAN_AUTO_FULL_MEM_GB="${WAN_AUTO_FULL_MEM_GB:-120}"
WAN_LR_RATIO="${WAN_LR_RATIO:-1.0}"
WAN_COND_NAME_PATTERN="${WAN_COND_NAME_PATTERN:-}"

if [[ "${TASK}" == "ti2v" && "${WAN_TRAIN_MODE}" != "frozen" && "${NPROC_PER_NODE}" != "1" ]]; then
  echo "[WARN] Wan train mode(${WAN_TRAIN_MODE}) currently supports single process only; force NPROC_PER_NODE=1"
  NPROC_PER_NODE="1"
  GPUS_PER_PROCESS="1"
fi
if [[ "${TASK}" != "ti2v" && "${LR_SCHEDULER_TYPE}" == "warmup_hold_cooldown" ]]; then
  echo "[WARN] warmup_hold_cooldown is only implemented in ti2v; fallback to constant_with_warmup for ${TASK}"
  LR_SCHEDULER_TYPE="constant_with_warmup"
fi

TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-49}"
I2V_FRAME_NUM="${I2V_FRAME_NUM:-49}"
ANIMATE_FRAME_NUM="${ANIMATE_FRAME_NUM:-49}"
TI2V_MAX_AREA="${TI2V_MAX_AREA:-262144}"
I2V_MAX_AREA="${I2V_MAX_AREA:-262144}"
ANIMATE_MAX_AREA="${ANIMATE_MAX_AREA:-262144}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-0.3}"
DIT_CONDITION_MODE="${DIT_CONDITION_MODE:-mq_only}"
TRAIN_REF_ANCHOR_MODE="${TRAIN_REF_ANCHOR_MODE:-none}"

METRICS_JSONL_PATH="${METRICS_JSONL_PATH:-${OUTPUT_ROOT}/logs/${TASK}_openvid1mfull_steps${NUM_TRAIN_STEPS}_$(date +%Y%m%d-%H%M%S).jsonl}"
METRICS_JSONL_PATH="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${METRICS_JSONL_PATH}")"

COMMON_ARGS=(
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
  --qwen3vl_model_id "${QWEN_MODEL}"
  --dit_condition_mode "${DIT_CONDITION_MODE}"
  --num_metaqueries "${NUM_METAQUERIES}"
  --null_image_prob "${NULL_IMAGE_PROB}"
  --null_caption_prob "${NULL_CAPTION_PROB}"
  --t5_align_anchor_tokens "${T5_ALIGN_ANCHOR_TOKENS}"
  --lambda_t5_align_l2 "${LAMBDA_T5_ALIGN_L2}"
  --lambda_t5_align_cos "${LAMBDA_T5_ALIGN_COS}"
  --lambda_t5_align_stats "${LAMBDA_T5_ALIGN_STATS}"
  --lambda_mq_image_preserve "${LAMBDA_MQ_IMAGE_PRESERVE}"
  --mq_image_preserve_margin "${MQ_IMAGE_PRESERVE_MARGIN}"
  --num_train_steps "${NUM_TRAIN_STEPS}"
  --warmup_steps "${WARMUP_STEPS}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --lr_min_ratio "${LR_MIN_RATIO}"
  --save_steps "${SAVE_STEPS}"
  --log_steps "${LOG_STEPS}"
  --loss_early_stop_min_step "${LOSS_EARLY_STOP_MIN_STEP}"
  --loss_early_stop_threshold "${LOSS_EARLY_STOP_THRESHOLD}"
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
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

if [[ "${TRAIN_MQ_INPUT_EMBEDDINGS}" == "1" ]]; then
  COMMON_ARGS+=(--train_mq_input_embeddings)
else
  COMMON_ARGS+=(--freeze_mq_input_embeddings)
fi

if [[ "${TASK}" == "ti2v" ]]; then
  COMMON_ARGS+=(--cooldown_steps "${COOLDOWN_STEPS}")
  COMMON_ARGS+=(--wan_train_mode "${WAN_TRAIN_MODE}")
  COMMON_ARGS+=(--wan_auto_full_mem_gb "${WAN_AUTO_FULL_MEM_GB}")
  COMMON_ARGS+=(--wan_lr_ratio "${WAN_LR_RATIO}")
  if [[ -n "${WAN_COND_NAME_PATTERN}" ]]; then
    COMMON_ARGS+=(--wan_cond_name_pattern "${WAN_COND_NAME_PATTERN}")
  fi
  COMMON_ARGS+=(--t5_align_mode "${T5_ALIGN_MODE}")
  COMMON_ARGS+=(--t5_align_ot_epsilon "${T5_ALIGN_OT_EPSILON}")
  COMMON_ARGS+=(--t5_align_ot_iters "${T5_ALIGN_OT_ITERS}")
  COMMON_ARGS+=(--lambda_wan_func_distill "${LAMBDA_WAN_FUNC_DISTILL}")
  COMMON_ARGS+=(--wan_func_teacher_mode "${WAN_FUNC_TEACHER_MODE}")
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
  if [[ "${TASK}" == "ti2v" ]]; then
    COMMON_ARGS+=(--enable_wan_func_distill)
  fi
else
  if [[ "${TASK}" == "ti2v" ]]; then
    COMMON_ARGS+=(--disable_wan_func_distill)
  fi
fi

if [[ "${REQUIRE_MQ_EMBEDDING_TRAIN}" == "1" && "${TRAIN_MQ_INPUT_EMBEDDINGS}" != "1" ]]; then
  echo "[FATAL] REQUIRE_MQ_EMBEDDING_TRAIN=1 but TRAIN_MQ_INPUT_EMBEDDINGS=${TRAIN_MQ_INPUT_EMBEDDINGS}"
  exit 11
fi

if [[ "${LOG_EVERY_STEP}" == "1" ]]; then
  COMMON_ARGS+=(--log_every_step)
fi
if [[ "${LOG_CUDA_MEMORY}" == "1" ]]; then
  COMMON_ARGS+=(--log_cuda_memory)
fi
if [[ "${WANDB_ENABLED}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
  COMMON_ARGS+=(--wandb_enabled)
  COMMON_ARGS+=(--wandb_project "${WANDB_PROJECT}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    COMMON_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  COMMON_ARGS+=(--wandb_mode "${WANDB_MODE}")
  COMMON_ARGS+=(--wandb_api_key "${WANDB_API_KEY}")
  COMMON_ARGS+=(--wandb_tags "${WANDB_TAGS}")
  COMMON_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
  if [[ "${WANDB_LOG_EVERY_STEP}" == "1" ]]; then
    COMMON_ARGS+=(--wandb_log_every_step)
  fi
  if [[ "${WANDB_LOG_CHECKPOINT}" == "1" ]]; then
    COMMON_ARGS+=(--wandb_log_checkpoint)
  fi
fi

echo "[LAUNCH][OPENVID1M_FULL] TASK=${TASK} steps=${NUM_TRAIN_STEPS}"
echo "[LAUNCH][OPENVID1M_FULL] openvid_root=${OPENVID_VIDEO_ROOT}"
echo "[LAUNCH][OPENVID1M_FULL] openvid_csv=${OPENVID_CSV_PATH}"
echo "[LAUNCH][OPENVID1M_FULL] openvid_hd_root=${OPENVID_HD_VIDEO_ROOT}"
echo "[LAUNCH][OPENVID1M_FULL] openvid_hd_csv=${OPENVID_HD_CSV_PATH}"
echo "[LAUNCH][OPENVID1M_FULL] openvid_limit=${OPENVID_LOCAL_LIMIT:-all} hd_limit=${OPENVID_HD_LOCAL_LIMIT:-all} total_limit=${OPENVID_LOCAL_TOTAL_LIMIT:-all}"
echo "[LAUNCH][OPENVID1M_FULL] metrics_jsonl=${METRICS_JSONL_PATH}"
echo "[LAUNCH][OPENVID1M_FULL] warmup=${WARMUP_STEPS} cooldown=${COOLDOWN_STEPS} lr_scheduler=${LR_SCHEDULER_TYPE} lr_min_ratio=${LR_MIN_RATIO}"
echo "[LAUNCH][OPENVID1M_FULL] early_stop enabled=${ENABLE_LOSS_EARLY_STOP} min_step=${LOSS_EARLY_STOP_MIN_STEP} threshold=${LOSS_EARLY_STOP_THRESHOLD}"
echo "[LAUNCH][OPENVID1M_FULL] batch_size=${BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} lr=${LEARNING_RATE}"
echo "[LAUNCH][OPENVID1M_FULL] num_metaqueries=${NUM_METAQUERIES} null_image_prob=${NULL_IMAGE_PROB} null_caption_prob=${NULL_CAPTION_PROB}"
echo "[LAUNCH][OPENVID1M_FULL] dit_condition_mode=${DIT_CONDITION_MODE} train_ref_anchor_mode=${TRAIN_REF_ANCHOR_MODE}"

if [[ "${ENABLE_DISTRIBUTED_RUNTIME}" == "1" || "${NPROC_PER_NODE}" != "1" ]]; then
  TRAIN_LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}")
  LAUNCH_MODE="torchrun+distributed"
else
  TRAIN_LAUNCHER=(python)
  LAUNCH_MODE="python-single-process"
fi
echo "[LAUNCH][OPENVID1M_FULL] launch_mode=${LAUNCH_MODE} nproc_per_node=${NPROC_PER_NODE} gpus_per_process=${GPUS_PER_PROCESS} enable_distributed_runtime=${ENABLE_DISTRIBUTED_RUNTIME}"

case "${TASK}" in
  ti2v)
    RUN_OUTPUT_DIR="${OUTPUT_ROOT}/ti2v_openvid1mfull_steps${NUM_TRAIN_STEPS}_nummq${NUM_METAQUERIES}_nullimg${NULL_IMAGE_PROB}_nullcap${NULL_CAPTION_PROB}"
    "${TRAIN_LAUNCHER[@]}" /home/liuzhirui/model/Wan2.2/train_metaquery_wan_new.py \
      --wan_checkpoint_dir "${WAN_TI2V_CKPT}" \
      --output_dir "${RUN_OUTPUT_DIR}" \
      --frame_num "${TI2V_FRAME_NUM}" \
      --max_area "${TI2V_MAX_AREA}" \
      --min_duration_sec "${MIN_DURATION_SEC}" \
      --train_ref_anchor_mode "${TRAIN_REF_ANCHOR_MODE}" \
      "${COMMON_ARGS[@]}"
    ;;
  i2v)
    RUN_OUTPUT_DIR="${OUTPUT_ROOT}/i2v_openvid1mfull_steps${NUM_TRAIN_STEPS}_nummq${NUM_METAQUERIES}_nullimg${NULL_IMAGE_PROB}_nullcap${NULL_CAPTION_PROB}"
    "${TRAIN_LAUNCHER[@]}" /home/liuzhirui/model/Wan2.2/train_metaquery_i2v_new.py \
      --wan_checkpoint_dir "${WAN_I2V_CKPT}" \
      --output_dir "${RUN_OUTPUT_DIR}" \
      --frame_num "${I2V_FRAME_NUM}" \
      --max_area "${I2V_MAX_AREA}" \
      --min_duration_sec "${MIN_DURATION_SEC}" \
      "${COMMON_ARGS[@]}"
    ;;
  animate)
    RUN_OUTPUT_DIR="${OUTPUT_ROOT}/animate_openvid1mfull_steps${NUM_TRAIN_STEPS}"
    "${TRAIN_LAUNCHER[@]}" /home/liuzhirui/model/Wan2.2/train_metaquery_animate_new.py \
      --wan_checkpoint_dir "${WAN_ANIMATE_CKPT}" \
      --output_dir "${RUN_OUTPUT_DIR}" \
      --frame_num "${ANIMATE_FRAME_NUM}" \
      --max_area "${ANIMATE_MAX_AREA}" \
      --min_duration_sec "${MIN_DURATION_SEC}" \
      "${COMMON_ARGS[@]}"
    ;;
  *)
    echo "[ERROR] TASK must be one of: ti2v | i2v | animate"
    exit 3
    ;;
esac

RUN_OUTPUT_DIR="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${RUN_OUTPUT_DIR}")"
echo "[OPENVID1M_FULL][VERIFY] before checkpoint: ${RUN_OUTPUT_DIR}/checkpoint-before-training"
echo "[OPENVID1M_FULL][VERIFY] final checkpoint : ${RUN_OUTPUT_DIR}/checkpoint-final"
echo "[OPENVID1M_FULL][VERIFY] chain manifest   : ${RUN_OUTPUT_DIR}/training_chain_manifest.json"
echo "[OPENVID1M_FULL][VERIFY] audit cmd:"
echo "python /home/liuzhirui/model/Wan2.2/verify_metaquery_chain.py \\"
echo "  --before_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-before-training \\"
echo "  --after_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-final \\"
echo "  --output_json ${RUN_OUTPUT_DIR}/chain_audit_report.json --strict"
