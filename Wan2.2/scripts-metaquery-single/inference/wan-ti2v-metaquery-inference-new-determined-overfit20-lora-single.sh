#!/bin/bash
set -euo pipefail
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory
export http_proxy=10.130.130.6:56830
export https_proxy=10.130.130.6:56830
export HF_ENDPOINT=https://hf-mirror.com
# Single-GPU inference launcher for Wan LoRA + extra-trainables checkpoints.
# Designed for checkpoints produced by:
#   train_stage1_openvid_local_metaquery_overfit10_wan_lora_2x80g.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_wan_lora_2x80g/ti2v_overfit10_lora_2x80g_steps900_nummq256_rank16_alpha16/checkpoint-earlystop-step869-denoise0.2393}"
WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
QWEN3VL_MODEL_ID="${QWEN3VL_MODEL_ID:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"

# PROMPT="A man is smiling."
# PROMPT="A girl is laughing."
# PROMPT="The video features a young man with dark hair and a blue shirt. He is standing in a forested area with trees and foliage in the background. The man appears to be in a thoughtful or contemplative mood, as he gazes off to the side. The lighting in the scene is natural, suggesting it might be daytime. The overall style of the video is realistic, with a focus on the man and his surroundings."
# PROMPT="The video features a woman with gray hair, wearing a yellow and black patterned blouse, sitting in a room with a white door and a light switch on the wall. She appears to be in a state of surprise or shock, as her mouth is open and her eyes are wide. The style of the video is realistic, with a focus on the woman\'s facial expression and the room\'s interior. The lighting in the room is bright, and the colors are vivid. The woman\'s position in the room and her facial expression suggest that she is reacting to something unexpected or startling. The overall mood of the video is tense and dramatic."
# PROMPT="The video features a woman with long, dark hair and a serious expression. She is wearing a black leather jacket and gold earrings. The lighting in the video is warm and soft, highlighting her features. The background is blurred, but it appears to be an indoor setting with a window. The woman's gaze is directed off to the side, and she seems to be deep in thought. The overall style of the video is moody and introspective."
PROMPT="The video is a close-up of a woman with long brown hair smiling at the camera. She is wearing a black top and has a pair of sunglasses on top of her head. The background is blurred, but it appears to be an outdoor setting with trees and a fence. The style of the video is casual and candid, capturing a moment of the woman's day."
# PROMPT="The video features a bald man with a beard, wearing a fur coat. He is shown in three different expressions, each one more intense than the last. The man is looking off to the side, his face contorted in a serious expression. The background is dark and moody, with a hint of a cityscape. The overall style of the video is dramatic and intense, with a focus on the man's expressions and the dark, moody atmosphere."
# PROMPT=""
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref0.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref1.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref2.jpg}"
REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref3.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref4.jpg}"

MODE="${MODE:-i2v}"                           # i2v | t2v
FRAME_NUM="${FRAME_NUM:-49}"                 # must be 4n+1
SIZE_W="${SIZE_W:-512}"
SIZE_H="${SIZE_H:-512}"
MAX_AREA="${MAX_AREA:-262144}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
GUIDE_SCALE="${GUIDE_SCALE:-12}"
SHIFT="${SHIFT:-5.0}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"      # unipc | dpm++
SEED="${SEED:-42}"

NUM_METAQUERIES="${NUM_METAQUERIES:-256}"
CONNECTOR_LAYERS="${CONNECTOR_LAYERS:-24}"
MQ_CFG_SCALE="${MQ_CFG_SCALE:-2.0}"
MQ_CONTEXT_SCALE="${MQ_CONTEXT_SCALE:-4.0}"
CFG_UNCOND_MODE="${CFG_UNCOND_MODE:-encode_negative}"  # encode_negative | zero_mq

LOAD_WAN_FINETUNE="${LOAD_WAN_FINETUNE:-1}"
VERIFY_LEVEL="${VERIFY_LEVEL:-basic}"                 # none | basic | full
VERIFY_FAIL_ON_WARNING="${VERIFY_FAIL_ON_WARNING:-1}" # 1=任何加载告警直接失败

DEVICE="${DEVICE:-0}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-/home/liuzhirui/model/Wan2.2/inference_outputs_lora_0402}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/wan_lora_${MODE}_seed${SEED}_${RUN_TS}.mp4}"
echo "[LORA-INFER] GUIDE_SCALE=${GUIDE_SCALE}"
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
fi

if [[ "${OFFLOAD_MODEL}" == "1" ]]; then
  CMD+=(--offload_model)
fi

echo "[LORA-INFER] checkpoint=${CHECKPOINT_PATH}"
echo "[LORA-INFER] mode=${MODE} frame_num=${FRAME_NUM} size=${SIZE_W}x${SIZE_H} steps=${SAMPLING_STEPS}"
echo "[LORA-INFER] num_metaqueries=${NUM_METAQUERIES} guide=${GUIDE_SCALE} mq_cfg=${MQ_CFG_SCALE} mq_ctx=${MQ_CONTEXT_SCALE}"
echo "[LORA-INFER] load_wan_finetune=${LOAD_WAN_FINETUNE} verify_level=${VERIFY_LEVEL} fail_on_warning=${VERIFY_FAIL_ON_WARNING}"
echo "[LORA-INFER] output=${OUTPUT_PATH}"
"${CMD[@]}"
