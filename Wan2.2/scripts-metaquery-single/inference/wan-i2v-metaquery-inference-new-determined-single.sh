#!/bin/bash
set -euo pipefail
eval "$(conda shell.bash hook)"
conda env list
conda activate /home/liuzhirui/miniconda3/envs/moviestory
export http_proxy=10.130.130.6:56830
export https_proxy=10.130.130.6:56830
export HF_ENDPOINT=https://hf-mirror.com
# Determined: FSDP inference launcher (ti2v / i2v / animate)

TASK="${TASK:-animate}"         # ti2v | i2v | animate
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

DIST_TIMEOUT_SEC="${DIST_TIMEOUT_SEC:-1800}"
WAN_DIST_WARMUP="${WAN_DIST_WARMUP:-none}"   # none | barrier | all_reduce
WAN_LOAD_STAGGER_SEC="${WAN_LOAD_STAGGER_SEC:-8}"

DIT_FSDP="${DIT_FSDP:-1}"
T5_FSDP="${T5_FSDP:-1}"
USE_SP="${USE_SP:-0}"
T5_CPU="${T5_CPU:-0}"

QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
TI2V_WAN_CKPT="${TI2V_WAN_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
I2V_WAN_CKPT="${I2V_WAN_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B}"
ANIMATE_WAN_CKPT="${ANIMATE_WAN_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-Animate-14B}"
MQ_CHECKPOINT="${MQ_CHECKPOINT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_openvid_stage1_full_training/animate_stage1_openvid_local/checkpoint-final}"

PROMPT="${PROMPT:-A girl is dancing.}"
NEG_PROMPT="${NEG_PROMPT:-}"
REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/SCAIL/examples/004/ref.jpg}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/liuzhirui/model/Wan2.2/inference_outputs_step20}"
mkdir -p "${OUTPUT_DIR}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"

SEED="${SEED:-42}"
NUM_METAQUERIES="${NUM_METAQUERIES:-32}"
CONNECTOR_LAYERS="${CONNECTOR_LAYERS:-24}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"      # unipc | dpm++

TI2V_MODE="${TI2V_MODE:-i2v}"                # t2v | i2v
TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-81}"
TI2V_SIZE_W="${TI2V_SIZE_W:-832}"
TI2V_SIZE_H="${TI2V_SIZE_H:-480}"
TI2V_MAX_AREA="${TI2V_MAX_AREA:-399360}"
TI2V_STEPS="${TI2V_STEPS:-50}"
TI2V_GUIDE_SCALE="${TI2V_GUIDE_SCALE:-5.0}"
TI2V_SHIFT="${TI2V_SHIFT:-5.0}"

I2V_FRAME_NUM="${I2V_FRAME_NUM:-81}"
I2V_MAX_AREA="${I2V_MAX_AREA:-921600}"
I2V_STEPS="${I2V_STEPS:-40}"
I2V_GUIDE_LOW="${I2V_GUIDE_LOW:-3.5}"
I2V_GUIDE_HIGH="${I2V_GUIDE_HIGH:-3.5}"
I2V_SHIFT="${I2V_SHIFT:-5.0}"

ANIMATE_FRAME_NUM="${ANIMATE_FRAME_NUM:-77}"
ANIMATE_H="${ANIMATE_H:-384}"
ANIMATE_W="${ANIMATE_W:-384}"
ANIMATE_STEPS="${ANIMATE_STEPS:-40}"
ANIMATE_GUIDE_SCALE="${ANIMATE_GUIDE_SCALE:-1.0}"
ANIMATE_SHIFT="${ANIMATE_SHIFT:-5.0}"

COMMON_FLAGS=(--distributed)
if [[ "${DIT_FSDP}" == "1" ]]; then COMMON_FLAGS+=(--dit_fsdp); fi
if [[ "${T5_FSDP}" == "1" ]]; then COMMON_FLAGS+=(--t5_fsdp); fi
if [[ "${USE_SP}" == "1" ]]; then COMMON_FLAGS+=(--use_sp); fi
if [[ "${T5_CPU}" == "1" ]]; then COMMON_FLAGS+=(--t5_cpu); fi

echo "[FSDP-INFER] TASK=${TASK} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[FSDP-INFER] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[FSDP-INFER] DIT_FSDP=${DIT_FSDP} T5_FSDP=${T5_FSDP} USE_SP=${USE_SP} T5_CPU=${T5_CPU}"
echo "[FSDP-INFER] checkpoint=${MQ_CHECKPOINT}"
echo "[FSDP-INFER] DIST_TIMEOUT_SEC=${DIST_TIMEOUT_SEC} WAN_DIST_WARMUP=${WAN_DIST_WARMUP}"
echo "[FSDP-INFER] WAN_LOAD_STAGGER_SEC=${WAN_LOAD_STAGGER_SEC} HF_HUB_OFFLINE=${HF_HUB_OFFLINE} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"

case "${TASK}" in
  animate)
    OUT="${OUTPUT_DIR}/animate_fsdp_seed${SEED}_${RUN_TS}.mp4"
    torchrun --standalone --nnodes=1 --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/inference_metaquery_animate_fsdp.py \
      "${COMMON_FLAGS[@]}" \
      --dist_timeout_sec "${DIST_TIMEOUT_SEC}" \
      --dist_warmup "${WAN_DIST_WARMUP}" \
      --load_stagger_sec "${WAN_LOAD_STAGGER_SEC}" \
      --checkpoint_path "${MQ_CHECKPOINT}" \
      --wan_checkpoint_dir "${ANIMATE_WAN_CKPT}" \
      --qwen3vl_model_id "${QWEN_MODEL}" \
      --prompt "${PROMPT}" \
      --ref_image "${REF_IMAGE}" \
      --negative_prompt "${NEG_PROMPT}" \
      --frame_num "${ANIMATE_FRAME_NUM}" \
      --height "${ANIMATE_H}" \
      --width "${ANIMATE_W}" \
      --sampling_steps "${ANIMATE_STEPS}" \
      --guide_scale "${ANIMATE_GUIDE_SCALE}" \
      --shift "${ANIMATE_SHIFT}" \
      --sample_solver "${SAMPLE_SOLVER}" \
      --seed "${SEED}" \
      --num_metaqueries "${NUM_METAQUERIES}" \
      --connector_num_hidden_layers "${CONNECTOR_LAYERS}" \
      --output_path "${OUT}"
    ;;

  i2v)
    OUT="${OUTPUT_DIR}/i2v_fsdp_seed${SEED}_${RUN_TS}.mp4"
    torchrun --standalone --nnodes=1 --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/inference_metaquery_i2v_fsdp.py \
      "${COMMON_FLAGS[@]}" \
      --dist_timeout_sec "${DIST_TIMEOUT_SEC}" \
      --dist_warmup "${WAN_DIST_WARMUP}" \
      --load_stagger_sec "${WAN_LOAD_STAGGER_SEC}" \
      --checkpoint_path "${MQ_CHECKPOINT}" \
      --wan_checkpoint_dir "${I2V_WAN_CKPT}" \
      --qwen3vl_model_id "${QWEN_MODEL}" \
      --prompt "${PROMPT}" \
      --ref_image "${REF_IMAGE}" \
      --negative_prompt "${NEG_PROMPT}" \
      --frame_num "${I2V_FRAME_NUM}" \
      --max_area "${I2V_MAX_AREA}" \
      --sampling_steps "${I2V_STEPS}" \
      --guide_scale "${I2V_GUIDE_LOW}" "${I2V_GUIDE_HIGH}" \
      --shift "${I2V_SHIFT}" \
      --sample_solver "${SAMPLE_SOLVER}" \
      --seed "${SEED}" \
      --num_metaqueries "${NUM_METAQUERIES}" \
      --connector_num_hidden_layers "${CONNECTOR_LAYERS}" \
      --output_path "${OUT}"
    ;;

  ti2v)
    OUT="${OUTPUT_DIR}/ti2v_fsdp_${TI2V_MODE}_seed${SEED}_${RUN_TS}.mp4"
    CMD=(
      torchrun --standalone --nnodes=1 --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/inference_metaquery_ti2v_fsdp.py
      "${COMMON_FLAGS[@]}"
      --dist_timeout_sec "${DIST_TIMEOUT_SEC}"
      --dist_warmup "${WAN_DIST_WARMUP}"
      --load_stagger_sec "${WAN_LOAD_STAGGER_SEC}"
      --checkpoint_path "${MQ_CHECKPOINT}"
      --wan_checkpoint_dir "${TI2V_WAN_CKPT}"
      --qwen3vl_model_id "${QWEN_MODEL}"
      --prompt "${PROMPT}"
      --negative_prompt "${NEG_PROMPT}"
      --mode "${TI2V_MODE}"
      --frame_num "${TI2V_FRAME_NUM}"
      --size "${TI2V_SIZE_W}" "${TI2V_SIZE_H}"
      --max_area "${TI2V_MAX_AREA}"
      --sampling_steps "${TI2V_STEPS}"
      --guide_scale "${TI2V_GUIDE_SCALE}"
      --shift "${TI2V_SHIFT}"
      --sample_solver "${SAMPLE_SOLVER}"
      --seed "${SEED}"
      --num_metaqueries "${NUM_METAQUERIES}"
      --connector_num_hidden_layers "${CONNECTOR_LAYERS}"
      --output_path "${OUT}"
    )
    if [[ "${TI2V_MODE}" == "i2v" ]]; then
      CMD+=(--ref_image "${REF_IMAGE}")
    fi
    "${CMD[@]}"
    ;;

  *)
    echo "[ERROR] TASK must be one of: ti2v | i2v | animate"
    exit 1
    ;;
esac

echo "[FSDP-INFER] Done."
