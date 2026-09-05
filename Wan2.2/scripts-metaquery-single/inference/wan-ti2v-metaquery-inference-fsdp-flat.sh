#!/bin/bash
set -euo pipefail

# Optional environment bootstrap:
# eval "$(conda shell.bash hook)"
# conda activate /path/to/your/env
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export http_proxy="${http_proxy:-10.130.130.6:56830}"
export https_proxy="${https_proxy:-10.130.130.6:56830}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_full_4gpu/ti2v_overfit30_steps800_nummq256_nullimg0.1_nullcap0.1/checkpoint-600}"
WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
QWEN3VL_MODEL_ID="${QWEN3VL_MODEL_ID:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"


# PROMPT="The video features a young man with dark hair and a blue shirt. He is standing in a forested area with trees and foliage in the background. The man appears to be in a thoughtful or contemplative mood, as he gazes off to the side. The lighting in the scene is natural, suggesting it might be daytime. The overall style of the video is realistic, with a focus on the man and his surroundings."
# PROMPT="The video features a woman with gray hair, wearing a yellow and black patterned blouse, sitting in a room with a white door and a light switch on the wall. She appears to be in a state of surprise or shock, as her mouth is open and her eyes are wide. The style of the video is realistic, with a focus on the woman\'s facial expression and the room\'s interior. The lighting in the room is bright, and the colors are vivid. The woman\'s position in the room and her facial expression suggest that she is reacting to something unexpected or startling. The overall mood of the video is tense and dramatic."
# PROMPT="The video features a woman with long, dark hair and a serious expression. She is wearing a black leather jacket and gold earrings. The lighting in the video is warm and soft, highlighting her features. The background is blurred, but it appears to be an indoor setting with a window. The woman's gaze is directed off to the side, and she seems to be deep in thought. The overall style of the video is moody and introspective."
PROMPT="The video is a close-up of a woman with long brown hair smiling at the camera. She is wearing a black top and has a pair of sunglasses on top of her head. The background is blurred, but it appears to be an outdoor setting with trees and a fence. The style of the video is casual and candid, capturing a moment of the woman's day."
# PROMPT="The video features a bald man with a beard, wearing a fur coat. He is shown in three different expressions, each one more intense than the last. The man is looking off to the side, his face contorted in a serious expression. The background is dark and moody, with a hint of a cityscape. The overall style of the video is dramatic and intense, with a focus on the man's expressions and the dark, moody atmosphere."

NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
# NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,杂乱的背景,三条腿,背景人很多,倒着走}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref0.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref1.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref2.jpg}"
REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref3.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref4.jpg}"

MODE="${MODE:-i2v}"                   # i2v | t2v
FRAME_NUM="${FRAME_NUM:-49}"          # must be 4n+1
SIZE_W="${SIZE_W:-512}"
SIZE_H="${SIZE_H:-512}"
MAX_AREA="${MAX_AREA:-1000000}"
SAMPLING_STEPS="${SAMPLING_STEPS:-40}"
GUIDE_SCALE="${GUIDE_SCALE:-7.5}"
SHIFT="${SHIFT:-5.0}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}"   # unipc | dpm++
SEED="${SEED:-42}"

NUM_METAQUERIES="${NUM_METAQUERIES:-256}"
CONNECTOR_NUM_HIDDEN_LAYERS="${CONNECTOR_NUM_HIDDEN_LAYERS:-24}"
VERIFY_LEVEL="${VERIFY_LEVEL:-basic}"     # none | basic | full
VERIFY_FAIL_ON_WARNING="${VERIFY_FAIL_ON_WARNING:-1}"
LOAD_WAN_FINETUNE="${LOAD_WAN_FINETUNE:-1}"
I2V_FORCE_SIZE="${I2V_FORCE_SIZE:-1}"
VERIFY_TRAIN_BEFORE_CHECKPOINT="${VERIFY_TRAIN_BEFORE_CHECKPOINT:-}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29561}"
DIST_TIMEOUT_SEC="${DIST_TIMEOUT_SEC:-1800}"
WAN_DIST_WARMUP="${WAN_DIST_WARMUP:-none}"         # none | barrier | all_reduce
WAN_LOAD_STAGGER_SEC="${WAN_LOAD_STAGGER_SEC:-0}"
T5_FSDP="${T5_FSDP:-0}"
USE_SP="${USE_SP:-0}"
T5_CPU="${T5_CPU:-0}"
INIT_ON_CPU="${INIT_ON_CPU:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs_fsdp}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/ti2v_fsdp_${MODE}_seed${SEED}_${RUN_TS}.mp4}"
VERIFY_REPORT_PATH="${VERIFY_REPORT_PATH:-${OUTPUT_PATH}.verify.json}"

if [[ ! -e "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] CHECKPOINT_PATH not found: ${CHECKPOINT_PATH}"
  exit 1
fi

if [[ ! -d "${WAN_CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] WAN_CHECKPOINT_DIR not found: ${WAN_CHECKPOINT_DIR}"
  exit 1
fi

if [[ "${MODE}" == "i2v" && ! -f "${REF_IMAGE}" ]]; then
  echo "[ERROR] MODE=i2v requires REF_IMAGE: ${REF_IMAGE}"
  exit 1
fi

if command -v torchrun >/dev/null 2>&1; then
  TORCHRUN_CMD=(
    torchrun
    --standalone
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_port "${MASTER_PORT}"
  )
  LAUNCHER_NAME="torchrun"
else
  TORCHRUN_CMD=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_port "${MASTER_PORT}"
  )
  LAUNCHER_NAME="python -m torch.distributed.run"
fi

CMD=(
  "${SCRIPT_DIR}/inference_metaquery_ti2v_fsdp.py"
  --distributed
  --dit_fsdp
  --dist_timeout_sec "${DIST_TIMEOUT_SEC}"
  --dist_warmup "${WAN_DIST_WARMUP}"
  --load_stagger_sec "${WAN_LOAD_STAGGER_SEC}"
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
  --shift "${SHIFT}"
  --sample_solver "${SAMPLE_SOLVER}"
  --seed "${SEED}"
  --output_path "${OUTPUT_PATH}"
  --verify_level "${VERIFY_LEVEL}"
  --verify_report_path "${VERIFY_REPORT_PATH}"
  --num_metaqueries "${NUM_METAQUERIES}"
  --connector_num_hidden_layers "${CONNECTOR_NUM_HIDDEN_LAYERS}"
)

if [[ "${MODE}" == "i2v" ]]; then
  CMD+=(--ref_image "${REF_IMAGE}")
fi

if [[ "${VERIFY_FAIL_ON_WARNING}" == "1" ]]; then
  CMD+=(--verify_fail_on_warning)
fi

if [[ "${LOAD_WAN_FINETUNE}" == "1" ]]; then
  CMD+=(--load_wan_finetune)
else
  CMD+=(--disable_load_wan_finetune)
fi

if [[ "${I2V_FORCE_SIZE}" == "1" && "${MODE}" == "i2v" ]]; then
  CMD+=(--i2v_force_size)
fi

if [[ -n "${VERIFY_TRAIN_BEFORE_CHECKPOINT}" ]]; then
  CMD+=(--verify_train_before_checkpoint "${VERIFY_TRAIN_BEFORE_CHECKPOINT}")
fi

if [[ "${T5_FSDP}" == "1" ]]; then
  CMD+=(--t5_fsdp)
fi

if [[ "${USE_SP}" == "1" ]]; then
  CMD+=(--use_sp)
fi

if [[ "${T5_CPU}" == "1" ]]; then
  CMD+=(--t5_cpu)
fi

if [[ "${INIT_ON_CPU}" == "0" ]]; then
  CMD+=(--no_init_on_cpu)
fi

echo "[FSDP-TI2V] launcher=${LAUNCHER_NAME}"
echo "[FSDP-TI2V] checkpoint=${CHECKPOINT_PATH}"
echo "[FSDP-TI2V] wan_ckpt=${WAN_CHECKPOINT_DIR}"
echo "[FSDP-TI2V] mode=${MODE} nproc_per_node=${NPROC_PER_NODE} frame_num=${FRAME_NUM}"
echo "[FSDP-TI2V] size=${SIZE_W}x${SIZE_H} steps=${SAMPLING_STEPS} guide_scale=${GUIDE_SCALE}"
echo "[FSDP-TI2V] output=${OUTPUT_PATH}"
echo "[FSDP-TI2V] verify_report=${VERIFY_REPORT_PATH}"

printf '[FSDP-TI2V] launch command:'
printf ' %q' "${TORCHRUN_CMD[@]}"
printf ' %q' "${CMD[@]}"
echo

"${TORCHRUN_CMD[@]}" "${CMD[@]}"
