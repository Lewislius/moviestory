#!/usr/bin/env bash
set -euo pipefail
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory
export http_proxy=10.130.130.6:56830
export https_proxy=10.130.130.6:56830
export HF_ENDPOINT=https://hf-mirror.com
# Demo launcher (animate-like i2v):
# 该脚本一次生成 5 个对照结果（同 prompt、同 ref_image）：
#   1) mq_i2v_ref_full      : MetaQuery i2v + ref_image (MQ + Wan 第一帧条件)
#   2) mq_t2v_mq_ref_only   : MetaQuery t2v + ref_image (仅 MQ 使用参考图)
#   3) mq_t2v_no_ref        : MetaQuery t2v + no ref_image (完全无参考图)
#   4) ti2v_plain_with_ref  : 纯 Wan TI2V（无 MetaQuery）+ ref_image
#   5) ti2v_plain_no_ref    : 纯 Wan TI2V（无 MetaQuery）+ no ref_image
#
# Usage:
#   chmod +x wan-ti2v-metaquery-inference-animate-like-single.sh
#   ./wan-ti2v-metaquery-inference-animate-like-single.sh
# Optional override:
#   PROMPT="A panda riding a bike" REF_IMAGE="/path/ref.png" ./wan-ti2v-metaquery-inference-animate-like-single.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# -----------------------------
# Required paths (edit as needed)
# -----------------------------
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10/ti2v_overfit10_steps50/checkpoint-final}"
# CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10/ti2v_overfit10_steps52/checkpoint-final}"
WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
QWEN3VL_MODEL_ID="${QWEN3VL_MODEL_ID:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"

# -----------------------------
# Prompt / reference
# -----------------------------
# PROMPT="The video features a young man with dark hair and a blue shirt. He is standing in a forested area with trees and foliage in the background. The man appears to be in a thoughtful or contemplative mood, as he gazes off to the side. The lighting in the scene is natural, suggesting it might be daytime. The overall style of the video is realistic, with a focus on the man and his surroundings."
# PROMPT="The video features a woman with gray hair, wearing a yellow and black patterned blouse, sitting in a room with a white door and a light switch on the wall. She appears to be in a state of surprise or shock, as her mouth is open and her eyes are wide. The style of the video is realistic, with a focus on the woman\'s facial expression and the room\'s interior. The lighting in the room is bright, and the colors are vivid. The woman\'s position in the room and her facial expression suggest that she is reacting to something unexpected or startling. The overall mood of the video is tense and dramatic."
# PROMPT="The video features a woman with long, dark hair and a serious expression. She is wearing a black leather jacket and gold earrings. The lighting in the video is warm and soft, highlighting her features. The background is blurred, but it appears to be an indoor setting with a window. The woman's gaze is directed off to the side, and she seems to be deep in thought. The overall style of the video is moody and introspective."
PROMPT="The video is a close-up of a woman with long brown hair smiling at the camera. She is wearing a black top and has a pair of sunglasses on top of her head. The background is blurred, but it appears to be an outdoor setting with trees and a fence. The style of the video is casual and candid, capturing a moment of the woman's day."
# PROMPT="The video features a bald man with a beard, wearing a fur coat. He is shown in three different expressions, each one more intense than the last. The man is looking off to the side, his face contorted in a serious expression. The background is dark and moody, with a hint of a cityscape. The overall style of the video is dramatic and intense, with a focus on the man's expressions and the dark, moody atmosphere."

# NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,杂乱的背景,三条腿,背景人很多,倒着走}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref0.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref1.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref2.jpg}"
REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref3.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref4.jpg}"
# -----------------------------
# Sampling / generation
# -----------------------------
FRAME_NUM="${FRAME_NUM:-49}" # must be 4n+1
SIZE_W="${SIZE_W:-512}"
SIZE_H="${SIZE_H:-512}"
MAX_AREA="${MAX_AREA:-1000000}" # 1280*704
SAMPLING_STEPS="${SAMPLING_STEPS:-32}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
SHIFT="${SHIFT:-5.0}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}" # unipc | dpm++
SEED="${SEED:-42}"

# -----------------------------
# MetaQuery / runtime
# -----------------------------
NUM_METAQUERIES="${NUM_METAQUERIES:-64}"
CONNECTOR_NUM_HIDDEN_LAYERS="${CONNECTOR_NUM_HIDDEN_LAYERS:-24}"
DEVICE="${DEVICE:-0}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-1}" # 1 => enable --offload_model
I2V_FORCE_SIZE="${I2V_FORCE_SIZE:-1}" # 1 => i2v 强制使用 SIZE_W/SIZE_H
I2V_REF_STRATEGY="${I2V_REF_STRATEGY:-animate_like}" # animate_like | hard_lock
VERIFY_LEVEL="${VERIFY_LEVEL:-basic}"   # none | basic | full
VERIFY_FAIL_ON_WARNING="${VERIFY_FAIL_ON_WARNING:-0}" # 1 => warning 直接失败
CHECKPOINT_LAYOUT_STRICT="${CHECKPOINT_LAYOUT_STRICT:-1}" # 1 => 关键文件缺失直接退出
VERIFY_TRAIN_BEFORE_CHECKPOINT="${VERIFY_TRAIN_BEFORE_CHECKPOINT:-}" # 训练前基线checkpoint(可选)
CHAIN_AUDIT_EPS="${CHAIN_AUDIT_EPS:-1e-7}"
CHAIN_AUDIT_STRICT="${CHAIN_AUDIT_STRICT:-0}" # 1 => 审计失败直接退出
RUN_CHAIN_AUDIT="${RUN_CHAIN_AUDIT:-1}"       # 1 => 在脚本结尾汇总训练/推理链路

# -----------------------------
# Pure TI2V (no MetaQuery) runtime
# -----------------------------
TI2V_TASK="${TI2V_TASK:-ti2v-5B}"
TI2V_SIZE="${TI2V_SIZE:-1280*704}"    # must be in SIZE_CONFIGS keys of generate.py
TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-${FRAME_NUM}}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs_overfit10_animate_like}"
mkdir -p "${OUTPUT_DIR}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-animate_like}"

if [[ ! -f "${REF_IMAGE}" ]]; then
  echo "[ERROR] REF_IMAGE not found: ${REF_IMAGE}"
  exit 1
fi

verify_checkpoint_layout() {
  local ckpt_path="$1"
  local strict="$2"
  local ckpt_dir="${ckpt_path}"

  if [[ -f "${ckpt_path}" ]]; then
    ckpt_dir="$(dirname "${ckpt_path}")"
  fi

  if [[ ! -d "${ckpt_dir}" ]]; then
    echo "[ERROR][CKPT] checkpoint dir not found: ${ckpt_dir}"
    exit 1
  fi

  echo "[VERIFY][CKPT] checkpoint dir: ${ckpt_dir}"

  local has_model=0
  if [[ -f "${ckpt_dir}/model.safetensors" || -f "${ckpt_dir}/mq_encoder_full.pt" || -f "${ckpt_dir}/mq_encoder_full.safetensors" ]]; then
    has_model=1
  fi
  local has_trainable=0
  if [[ -f "${ckpt_dir}/mq_encoder_trainable.pt" || -f "${ckpt_dir}/mq_encoder_trainable.safetensors" ]]; then
    has_trainable=1
  fi

  local files=(
    "config.json"
    "trainer_state.json"
    "optimizer.pt"
    "scheduler.pt"
    "training_args.bin"
  )
  local missing=0
  local f=""
  for f in "${files[@]}"; do
    if [[ -f "${ckpt_dir}/${f}" ]]; then
      echo "  [OK] ${f}"
    else
      echo "  [MISSING] ${f}"
      missing=1
    fi
  done

  if [[ "${has_model}" == "1" ]]; then
    echo "  [OK] model file: found (model.safetensors / mq_encoder_full.pt)"
  else
    echo "  [MISSING] model file: model.safetensors 或 mq_encoder_full.pt"
    missing=1
  fi

  if [[ "${has_trainable}" == "1" ]]; then
    echo "  [OK] trainable file: found (mq_encoder_trainable.pt / .safetensors)"
  else
    echo "  [MISSING] trainable file: mq_encoder_trainable.pt / .safetensors"
    missing=1
  fi

  if [[ -f "${ckpt_dir}/training_state.pt" ]]; then
    echo "  [OK] training_state.pt"
  else
    echo "  [WARN] training_state.pt not found (可选)"
  fi

  if [[ "${missing}" == "1" && "${strict}" == "1" ]]; then
    echo "[ERROR][CKPT] checkpoint 布局检查失败（CHECKPOINT_LAYOUT_STRICT=1）"
    exit 1
  fi

  if [[ "${missing}" == "1" ]]; then
    echo "[WARN][CKPT] checkpoint 存在缺失文件，但继续运行（CHECKPOINT_LAYOUT_STRICT=0）"
  else
    echo "[VERIFY][CKPT] checkpoint 布局检查通过"
  fi
}

run_mq_case() {
  local case_name="$1"
  local mode="$2"
  local use_ref="$3"   # 1|0
  local output_path="${OUTPUT_DIR}/${case_name}_seed${SEED}_W${SIZE_W}_H${SIZE_H}_frame${FRAME_NUM}_${RUN_TS}.mp4"
  local verify_report_path="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${case_name}_seed${SEED}_${RUN_TS}.verify.json"

  local cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/inference_metaquery_wan_animate_like.py"
    --checkpoint_path "${CHECKPOINT_PATH}"
    --wan_checkpoint_dir "${WAN_CHECKPOINT_DIR}"
    --qwen3vl_model_id "${QWEN3VL_MODEL_ID}"
    --prompt "${PROMPT}"
    --negative_prompt "${NEGATIVE_PROMPT}"
    --mode "${mode}"
    --frame_num "${FRAME_NUM}"
    --size "${SIZE_W}" "${SIZE_H}"
    --max_area "${MAX_AREA}"
    --sampling_steps "${SAMPLING_STEPS}"
    --guide_scale "${GUIDE_SCALE}"
    --shift "${SHIFT}"
    --sample_solver "${SAMPLE_SOLVER}"
    --seed "${SEED}"
    --output_path "${output_path}"
    --num_metaqueries "${NUM_METAQUERIES}"
    --connector_num_hidden_layers "${CONNECTOR_NUM_HIDDEN_LAYERS}"
    --device "${DEVICE}"
    --verify_level "${VERIFY_LEVEL}"
    --verify_report_path "${verify_report_path}"
  )

  if [[ "${use_ref}" == "1" ]]; then
    cmd+=(--ref_image "${REF_IMAGE}")
  fi

  if [[ "${OFFLOAD_MODEL}" == "1" ]]; then
    cmd+=(--offload_model)
  fi
  if [[ "${mode}" == "i2v" && "${I2V_FORCE_SIZE}" == "1" ]]; then
    cmd+=(--i2v_force_size)
  fi
  if [[ "${mode}" == "i2v" ]]; then
    cmd+=(--i2v_ref_strategy "${I2V_REF_STRATEGY}")
  fi
  if [[ "${VERIFY_FAIL_ON_WARNING}" == "1" ]]; then
    cmd+=(--verify_fail_on_warning)
  fi
  if [[ -n "${VERIFY_TRAIN_BEFORE_CHECKPOINT}" ]]; then
    cmd+=(--verify_train_before_checkpoint "${VERIFY_TRAIN_BEFORE_CHECKPOINT}")
  fi

  echo "[DEMO][MQ] case=${case_name} mode=${mode} use_ref=${use_ref} device=${DEVICE}"
  echo "[DEMO] size=${SIZE_W}x${SIZE_H} frames=${FRAME_NUM} steps=${SAMPLING_STEPS}"
  echo "[DEMO] output=${output_path}"
  echo "[DEMO] verify_level=${VERIFY_LEVEL} verify_report=${verify_report_path}"
  "${cmd[@]}"
}

run_chain_audit() {
  if [[ "${RUN_CHAIN_AUDIT}" != "1" ]]; then
    return
  fi
  if [[ -z "${VERIFY_TRAIN_BEFORE_CHECKPOINT}" ]]; then
    echo "[CHAIN-AUDIT] skip: VERIFY_TRAIN_BEFORE_CHECKPOINT is empty"
    return
  fi
  local audit_json="${OUTPUT_DIR}/${OUTPUT_PREFIX}_chain_audit_seed${SEED}_${RUN_TS}.json"
  local report_glob="${OUTPUT_DIR}/${OUTPUT_PREFIX}_*_seed${SEED}_${RUN_TS}.verify.json"
  local cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_metaquery_chain.py"
    --before_checkpoint "${VERIFY_TRAIN_BEFORE_CHECKPOINT}"
    --after_checkpoint "${CHECKPOINT_PATH}"
    --inference_report_glob "${report_glob}"
    --eps "${CHAIN_AUDIT_EPS}"
    --output_json "${audit_json}"
  )
  if [[ "${CHAIN_AUDIT_STRICT}" == "1" ]]; then
    cmd+=(--strict)
  fi
  echo "[CHAIN-AUDIT] before=${VERIFY_TRAIN_BEFORE_CHECKPOINT}"
  echo "[CHAIN-AUDIT] after=${CHECKPOINT_PATH}"
  echo "[CHAIN-AUDIT] reports=${report_glob}"
  echo "[CHAIN-AUDIT] output=${audit_json}"
  "${cmd[@]}"
}

run_plain_ti2v_case() {
  local case_name="$1"
  local use_ref="$2"   # 1|0
  local output_path="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${case_name}_seed${SEED}_${RUN_TS}.mp4"

  local cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/generate.py"
    --task "${TI2V_TASK}"
    --ckpt_dir "${WAN_CHECKPOINT_DIR}"
    --prompt "${PROMPT}"
    --size "${TI2V_SIZE}"
    --frame_num "${TI2V_FRAME_NUM}"
    --sample_solver "${SAMPLE_SOLVER}"
    --sample_steps "${SAMPLING_STEPS}"
    --sample_shift "${SHIFT}"
    --sample_guide_scale "${GUIDE_SCALE}"
    --base_seed "${SEED}"
    --save_file "${output_path}"
  )

  if [[ "${use_ref}" == "1" ]]; then
    cmd+=(--image "${REF_IMAGE}")
  fi

  if [[ "${OFFLOAD_MODEL}" == "1" ]]; then
    cmd+=(--offload_model true)
  else
    cmd+=(--offload_model false)
  fi

  echo "[DEMO][TI2V] case=${case_name} use_ref=${use_ref}"
  echo "[DEMO][TI2V] task=${TI2V_TASK} size=${TI2V_SIZE} frames=${TI2V_FRAME_NUM} steps=${SAMPLING_STEPS}"
  echo "[DEMO][TI2V] output=${output_path}"
  "${cmd[@]}"
}

echo "[DEMO] checkpoint=${CHECKPOINT_PATH}"
echo "[DEMO] wan_ckpt=${WAN_CHECKPOINT_DIR}"
echo "[DEMO] prompt=${PROMPT}"
echo "[DEMO] ref_image=${REF_IMAGE}"
echo "[DEMO] i2v_force_size=${I2V_FORCE_SIZE} size=${SIZE_W}x${SIZE_H}"
echo "[DEMO] i2v_ref_strategy=${I2V_REF_STRATEGY}"
echo "[DEMO] verify_level=${VERIFY_LEVEL}"
echo "[DEMO] verify_train_before_checkpoint=${VERIFY_TRAIN_BEFORE_CHECKPOINT:-<none>}"
echo "[DEMO] generating 5 cases ..."

verify_checkpoint_layout "${CHECKPOINT_PATH}" "${CHECKPOINT_LAYOUT_STRICT}"
if [[ -n "${VERIFY_TRAIN_BEFORE_CHECKPOINT}" ]]; then
  verify_checkpoint_layout "${VERIFY_TRAIN_BEFORE_CHECKPOINT}" "${CHECKPOINT_LAYOUT_STRICT}"
fi


# run_plain_ti2v_case "ti2v_plain_with_ref" "1"

# run_plain_ti2v_case "ti2v_plain_no_ref" "0"

run_mq_case "mq_i2v_ref_full" "i2v" "1"

run_mq_case "mq_t2v_mq_ref_only" "t2v" "1"

run_mq_case "mq_t2v_no_ref" "t2v" "0"

run_chain_audit


echo "[DEMO] done. outputs in ${OUTPUT_DIR}"
