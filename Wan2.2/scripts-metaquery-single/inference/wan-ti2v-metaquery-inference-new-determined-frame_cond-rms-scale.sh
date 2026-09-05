#!/usr/bin/env bash
set -euo pipefail

# # Guide-only frame-cond 版本：
# # 1) 保留 WAN 首帧条件 (animate_ref_slot)
# # 2) 仅保留 GUIDE_SCALE 作为采样强度控制
# source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
# conda activate /home/liuzhirui/miniconda3/envs/moviestory
# export http_proxy=10.130.130.6:56830
# export https_proxy=10.130.130.6:56830
# export HF_ENDPOINT=https://hf-mirror.com

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PYTHON_BIN="${PYTHON_BIN:-python}"

# # CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_ti2v_firstframe_wan_cond/ti2v_firstframe_wancond_overfit10_steps803_nummq256_nullimg0.1_nullcap0.1/checkpoint-final}"
# # CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_ti2v_firstframe_wan_cond/ti2v_firstframe_wancond_overfit10_steps888_nummq256_nullimg0.1_nullcap0.1/checkpoint-final}"
# # CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_ti2v_firstframe_wan_cond/ti2v_firstframe_wancond_overfit10_steps890_nummq256_nullimg0.1_nullcap0.1/checkpoint-final}"
# CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_ti2v_firstframe_wan_cond/ti2v_firstframe_wancond_overfit10_steps889_nummq256_nullimg0.1_nullcap0.1/checkpoint-680}"
# WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
# QWEN3VL_MODEL_ID="${QWEN3VL_MODEL_ID:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"

# # PROMPT="A man is smiling."
# # PROMPT="A girl is laughing."
# # PROMPT="A girl is sad and crying."
# # PROMPT="The video features a young man with dark hair and a blue shirt. He is standing in a forested area with trees and foliage in the background. The man appears to be in a thoughtful or contemplative mood, as he gazes off to the side. The lighting in the scene is natural, suggesting it might be daytime. The overall style of the video is realistic, with a focus on the man and his surroundings."
# # PROMPT="The video features a woman with gray hair, wearing a yellow and black patterned blouse, sitting in a room with a white door and a light switch on the wall. She appears to be in a state of surprise or shock, as her mouth is open and her eyes are wide. The style of the video is realistic, with a focus on the woman\'s facial expression and the room\'s interior. The lighting in the room is bright, and the colors are vivid. The woman\'s position in the room and her facial expression suggest that she is reacting to something unexpected or startling. The overall mood of the video is tense and dramatic."
# # PROMPT="The video features a woman with long, dark hair and a serious expression. She is wearing a black leather jacket and gold earrings. The lighting in the video is warm and soft, highlighting her features. The background is blurred, but it appears to be an indoor setting with a window. The woman's gaze is directed off to the side, and she seems to be deep in thought. The overall style of the video is moody and introspective."
# # PROMPT="The video is a close-up of a woman with long brown hair smiling at the camera. She is wearing a black top and has a pair of sunglasses on top of her head. The background is blurred, but it appears to be an outdoor setting with trees and a fence. The style of the video is casual and candid, capturing a moment of the woman's day."
# PROMPT="The video features a bald man with a beard, wearing a fur coat. He is shown in three different expressions, each one more intense than the last. The man is looking off to the side, his face contorted in a serious expression. The background is dark and moody, with a hint of a cityscape. The overall style of the video is dramatic and intense, with a focus on the man's expressions and the dark, moody atmosphere."
# # PROMPT=""
# NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
# # REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref0.jpg}"
# # REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref1.jpg}"
# # REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref2.jpg}"
# # REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref3.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref4.jpg}"

# # Option B: provide SOURCE_VIDEO and let this script extract frame-0.
# SOURCE_VIDEO="${SOURCE_VIDEO:-}"

# FRAME_NUM="${FRAME_NUM:-49}"            # 4n+1
# MAX_AREA="${MAX_AREA:-262144}"
# SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
# GUIDE_SCALE="${GUIDE_SCALE:-1}"
# CFG_UNCOND_MODE="${CFG_UNCOND_MODE:-encode_negative}" # encode_negative | zero_mq
# USE_CHECKPOINT_RMS_HINTS="${USE_CHECKPOINT_RMS_HINTS:-1}" # 1=未显式传参时，自动跟随checkpoint中的rms训练参数
# MQ_NORM_PROBE_WITH_T5_USER_SET=0a
# if [[ -n "${MQ_NORM_PROBE_WITH_T5+x}" ]]; then MQ_NORM_PROBE_WITH_T5_USER_SET=1; fi
# MQ_NORM_MATCH_T5_USER_SET=0
# if [[ -n "${MQ_NORM_MATCH_T5+x}" ]]; then MQ_NORM_MATCH_T5_USER_SET=1; fi
# MQ_NORM_WARN_RATIO_LOW_USER_SET=0
# if [[ -n "${MQ_NORM_WARN_RATIO_LOW+x}" ]]; then MQ_NORM_WARN_RATIO_LOW_USER_SET=1; fi
# MQ_NORM_WARN_RATIO_HIGH_USER_SET=0
# if [[ -n "${MQ_NORM_WARN_RATIO_HIGH+x}" ]]; then MQ_NORM_WARN_RATIO_HIGH_USER_SET=1; fi
# MQ_NORM_MATCH_CLIP_MIN_USER_SET=0
# if [[ -n "${MQ_NORM_MATCH_CLIP_MIN+x}" ]]; then MQ_NORM_MATCH_CLIP_MIN_USER_SET=1; fi
# MQ_NORM_MATCH_CLIP_MAX_USER_SET=0
# if [[ -n "${MQ_NORM_MATCH_CLIP_MAX+x}" ]]; then MQ_NORM_MATCH_CLIP_MAX_USER_SET=1; fi
# MQ_NORM_PROBE_WITH_T5="${MQ_NORM_PROBE_WITH_T5:-1}"   # 1=对比 MQ/T5 token RMS
# MQ_NORM_MATCH_T5="${MQ_NORM_MATCH_T5:-1}"             # 1=按 T5 RMS 对齐 MQ RMS（默认开启）
# MQ_NORM_WARN_RATIO_LOW="${MQ_NORM_WARN_RATIO_LOW:-0.25}"
# MQ_NORM_WARN_RATIO_HIGH="${MQ_NORM_WARN_RATIO_HIGH:-4.0}"
# MQ_NORM_MATCH_CLIP_MIN="${MQ_NORM_MATCH_CLIP_MIN:-0.03}"
# MQ_NORM_MATCH_CLIP_MAX="${MQ_NORM_MATCH_CLIP_MAX:-4.0}"
# MQ_NORM_MATCH_EVERY_STEP="${MQ_NORM_MATCH_EVERY_STEP:-1}" # 1=每个去噪step都重新执行MQ RMS缩放
# MQ_NORM_LOG_EACH_STEP="${MQ_NORM_LOG_EACH_STEP:-1}"       # 1=每步打印缩放前后RMS与scale
# MQ_NORM_LOG_INTERVAL="${MQ_NORM_LOG_INTERVAL:-1}"         # 1=每步打印
# SHIFT="${SHIFT:-5.0}"
# SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}" # unipc | dpm++
# SEED="${SEED:-42}"
# VERIFY_LEVEL="${VERIFY_LEVEL:-basic}"   # none | basic | full
# VERIFY_FAIL_ON_WARNING="${VERIFY_FAIL_ON_WARNING:-0}" # 1 => warning直接失败

# NUM_METAQUERIES="${NUM_METAQUERIES:-256}"
# CONNECTOR_LAYERS="${CONNECTOR_LAYERS:-24}"
# DEVICE="${DEVICE:-0}"
# OFFLOAD_MODEL="${OFFLOAD_MODEL:-1}"     # 1=enable --offload_model
# WAN_CONTEXT_MQ_ONLY="${WAN_CONTEXT_MQ_ONLY:-1}" # 1: WAN context仅保留MQ特征(不拼接T5)
# SHOW_CHECKPOINT_TRAINING_HINTS="${SHOW_CHECKPOINT_TRAINING_HINTS:-1}"

# OUTPUT_DIR="${OUTPUT_DIR:-/home/liuzhirui/model/Wan2.2/inference_outputs_20260407}"
# RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
# mkdir -p "${OUTPUT_DIR}"
# OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/ti2v_firstframe_wancond_guide_only_seed${SEED}_${RUN_TS}.mp4}"
# VERIFY_REPORT_PATH="${VERIFY_REPORT_PATH:-${OUTPUT_PATH}.verify.json}"
# I2V_INFER_ENTRY="${SCRIPT_DIR}/inference_metaquery_wan_rms_aligned.py"

# if [[ -n "${SOURCE_VIDEO}" ]]; then
#   EXTRACTED_REF_IMAGE="${EXTRACTED_REF_IMAGE:-${OUTPUT_DIR}/firstframe_wancond_${RUN_TS}.png}"
#   export SOURCE_VIDEO EXTRACTED_REF_IMAGE
#   "${PYTHON_BIN}" - << 'PY'
# import cv2
# import os
# from pathlib import Path

# src = Path(os.environ["SOURCE_VIDEO"]).expanduser()
# dst = Path(os.environ["EXTRACTED_REF_IMAGE"]).expanduser()
# dst.parent.mkdir(parents=True, exist_ok=True)

# if not src.exists():
#     raise FileNotFoundError(f"SOURCE_VIDEO not found: {src}")

# cap = cv2.VideoCapture(str(src))
# if not cap.isOpened():
#     raise RuntimeError(f"Failed to open SOURCE_VIDEO: {src}")
# ok, frame = cap.read()
# cap.release()
# if (not ok) or frame is None:
#     raise RuntimeError(f"Failed to read first frame from: {src}")

# if not cv2.imwrite(str(dst), frame):
#     raise RuntimeError(f"Failed to write extracted frame to: {dst}")

# print(f"[I2V-WANCOND] extracted_ref_image={dst}")
# PY
#   REF_IMAGE="${EXTRACTED_REF_IMAGE}"
# fi

# if [[ -z "${REF_IMAGE}" ]]; then
#   echo "[ERROR] REF_IMAGE is empty. Set REF_IMAGE or SOURCE_VIDEO."
#   exit 2
# fi
# if [[ ! -f "${REF_IMAGE}" ]]; then
#   echo "[ERROR] REF_IMAGE not found: ${REF_IMAGE}"
#   exit 3
# fi

# if [[ "${SHOW_CHECKPOINT_TRAINING_HINTS}" == "1" ]]; then
#   export CHECKPOINT_PATH
#   "${PYTHON_BIN}" - << 'PY'
# import json
# import os
# from pathlib import Path

# ckpt = Path(os.environ["CHECKPOINT_PATH"]).expanduser()
# candidates = []
# if ckpt.is_dir():
#     candidates.extend([ckpt / "config.json", ckpt / "training_args.json"])
# else:
#     candidates.extend([ckpt.parent / "config.json", ckpt.parent / "training_args.json"])

# cfg = None
# cfg_path = None
# for p in candidates:
#     if p.exists():
#         try:
#             cfg = json.loads(p.read_text(encoding="utf-8"))
#             cfg_path = p
#             break
#         except Exception:
#             pass

# if cfg is None:
#     print("[I2V-WANCOND][INFER] checkpoint training-hints: not found (config.json/training_args.json)")
# else:
#     def g(k, d="N/A"):
#         v = cfg.get(k, d)
#         return v if v is not None else d

#     print(f"[I2V-WANCOND][INFER] checkpoint cfg file={cfg_path}")
#     print(
#         "[I2V-WANCOND][INFER] checkpoint training-hints "
#         f"lr_scheduler_type={g('lr_scheduler_type')} "
#         f"warmup_steps={g('warmup_steps')} cooldown_steps={g('cooldown_steps')} "
#         f"enable_loss_early_stop={g('enable_loss_early_stop')} "
#         f"loss_early_stop_min_step={g('loss_early_stop_min_step')} "
#         f"loss_early_stop_threshold={g('loss_early_stop_threshold')} "
#         f"wan_train_mode={g('wan_train_mode')} "
#         f"wan_lr_ratio={g('wan_lr_ratio')}"
#     )
# PY
# fi

# if [[ "${USE_CHECKPOINT_RMS_HINTS}" == "1" ]]; then
#   export CHECKPOINT_PATH
#   CKPT_RMS_HINTS_RAW="$("${PYTHON_BIN}" - << 'PY'
# import json
# import os
# from pathlib import Path

# ckpt = Path(os.environ["CHECKPOINT_PATH"]).expanduser()
# candidates = []
# if ckpt.is_dir():
#     candidates.extend([ckpt / "training_args.json", ckpt / "config.json"])
# else:
#     candidates.extend([ckpt.parent / "training_args.json", ckpt.parent / "config.json"])

# cfg = None
# for p in candidates:
#     if p.exists():
#         try:
#             cfg = json.loads(p.read_text(encoding="utf-8"))
#             break
#         except Exception:
#             pass

# if not isinstance(cfg, dict):
#     raise SystemExit(0)

# def emit_bool(src_key: str, dst_key: str) -> None:
#     if src_key in cfg:
#         v = cfg.get(src_key)
#         if isinstance(v, bool):
#             print(f"{dst_key}={1 if v else 0}")
#         elif isinstance(v, (int, float)):
#             print(f"{dst_key}={1 if float(v) != 0.0 else 0}")

# def emit_float(src_key: str, dst_key: str) -> None:
#     if src_key in cfg:
#         v = cfg.get(src_key)
#         if isinstance(v, (int, float)):
#             print(f"{dst_key}={float(v)}")
#         else:
#             try:
#                 print(f"{dst_key}={float(str(v).strip())}")
#             except Exception:
#                 pass

# emit_bool("mq_norm_probe_with_t5", "CKPT_MQ_NORM_PROBE_WITH_T5")
# emit_bool("mq_norm_match_t5", "CKPT_MQ_NORM_MATCH_T5")
# emit_float("mq_norm_warn_ratio_low", "CKPT_MQ_NORM_WARN_RATIO_LOW")
# emit_float("mq_norm_warn_ratio_high", "CKPT_MQ_NORM_WARN_RATIO_HIGH")
# emit_float("mq_norm_match_clip_min", "CKPT_MQ_NORM_MATCH_CLIP_MIN")
# emit_float("mq_norm_match_clip_max", "CKPT_MQ_NORM_MATCH_CLIP_MAX")
# PY
# )"

#   if [[ -n "${CKPT_RMS_HINTS_RAW}" ]]; then
#     eval "${CKPT_RMS_HINTS_RAW}"
#     if [[ "${MQ_NORM_PROBE_WITH_T5_USER_SET}" != "1" && -n "${CKPT_MQ_NORM_PROBE_WITH_T5:-}" ]]; then
#       MQ_NORM_PROBE_WITH_T5="${CKPT_MQ_NORM_PROBE_WITH_T5}"
#     fi
#     if [[ "${MQ_NORM_MATCH_T5_USER_SET}" != "1" && -n "${CKPT_MQ_NORM_MATCH_T5:-}" ]]; then
#       MQ_NORM_MATCH_T5="${CKPT_MQ_NORM_MATCH_T5}"
#     fi
#     if [[ "${MQ_NORM_WARN_RATIO_LOW_USER_SET}" != "1" && -n "${CKPT_MQ_NORM_WARN_RATIO_LOW:-}" ]]; then
#       MQ_NORM_WARN_RATIO_LOW="${CKPT_MQ_NORM_WARN_RATIO_LOW}"
#     fi
#     if [[ "${MQ_NORM_WARN_RATIO_HIGH_USER_SET}" != "1" && -n "${CKPT_MQ_NORM_WARN_RATIO_HIGH:-}" ]]; then
#       MQ_NORM_WARN_RATIO_HIGH="${CKPT_MQ_NORM_WARN_RATIO_HIGH}"
#     fi
#     if [[ "${MQ_NORM_MATCH_CLIP_MIN_USER_SET}" != "1" && -n "${CKPT_MQ_NORM_MATCH_CLIP_MIN:-}" ]]; then
#       MQ_NORM_MATCH_CLIP_MIN="${CKPT_MQ_NORM_MATCH_CLIP_MIN}"
#     fi
#     if [[ "${MQ_NORM_MATCH_CLIP_MAX_USER_SET}" != "1" && -n "${CKPT_MQ_NORM_MATCH_CLIP_MAX:-}" ]]; then
#       MQ_NORM_MATCH_CLIP_MAX="${CKPT_MQ_NORM_MATCH_CLIP_MAX}"
#     fi
#   fi
# fi

# INFER_HELP_CACHE=""
# load_infer_help() {
#   if [[ -n "${INFER_HELP_CACHE}" ]]; then
#     return
#   fi
#   INFER_HELP_CACHE="$("${PYTHON_BIN}" "${I2V_INFER_ENTRY}" --help 2>&1 || true)"
# }

# infer_supports_arg() {
#   local arg_name="$1"
#   load_infer_help
#   if [[ "${INFER_HELP_CACHE}" == *"${arg_name}"* ]]; then
#     return 0
#   fi
#   return 1
# }

# CMD=(
#   "${PYTHON_BIN}" "${I2V_INFER_ENTRY}"
#   --checkpoint_path "${CHECKPOINT_PATH}"
#   --wan_checkpoint_dir "${WAN_CHECKPOINT_DIR}"
#   --qwen3vl_model_id "${QWEN3VL_MODEL_ID}"
#   --mode i2v
#   --prompt "${PROMPT}"
#   --ref_image "${REF_IMAGE}"
#   --negative_prompt "${NEGATIVE_PROMPT}"
#   --frame_num "${FRAME_NUM}"
#   --max_area "${MAX_AREA}"
#   --sampling_steps "${SAMPLING_STEPS}"
#   --guide_scale "${GUIDE_SCALE}"
#   --shift "${SHIFT}"
#   --sample_solver "${SAMPLE_SOLVER}"
#   --seed "${SEED}"
#   --num_metaqueries "${NUM_METAQUERIES}"
#   --connector_num_hidden_layers "${CONNECTOR_LAYERS}"
#   --device "${DEVICE}"
#   --output_path "${OUTPUT_PATH}"
# )

# if infer_supports_arg "--i2v_method"; then
#   CMD+=(--i2v_method animate_ref_slot)
# fi
# if infer_supports_arg "--cfg_uncond_mode"; then
#   CMD+=(--cfg_uncond_mode "${CFG_UNCOND_MODE}")
# fi
# if infer_supports_arg "--mq_norm_warn_ratio_low"; then
#   CMD+=(--mq_norm_warn_ratio_low "${MQ_NORM_WARN_RATIO_LOW}")
# fi
# if infer_supports_arg "--mq_norm_warn_ratio_high"; then
#   CMD+=(--mq_norm_warn_ratio_high "${MQ_NORM_WARN_RATIO_HIGH}")
# fi
# if infer_supports_arg "--mq_norm_match_clip_min"; then
#   CMD+=(--mq_norm_match_clip_min "${MQ_NORM_MATCH_CLIP_MIN}")
# fi
# if infer_supports_arg "--mq_norm_match_clip_max"; then
#   CMD+=(--mq_norm_match_clip_max "${MQ_NORM_MATCH_CLIP_MAX}")
# fi
# if infer_supports_arg "--mq_norm_log_interval"; then
#   CMD+=(--mq_norm_log_interval "${MQ_NORM_LOG_INTERVAL}")
# fi
# if infer_supports_arg "--verify_level"; then
#   CMD+=(--verify_level "${VERIFY_LEVEL}")
# fi
# if infer_supports_arg "--verify_report_path"; then
#   CMD+=(--verify_report_path "${VERIFY_REPORT_PATH}")
# fi

# if [[ "${OFFLOAD_MODEL}" == "1" ]]; then
#   CMD+=(--offload_model)
# fi
# if infer_supports_arg "--mq_norm_probe_with_t5" && [[ "${MQ_NORM_PROBE_WITH_T5}" == "1" ]]; then
#   CMD+=(--mq_norm_probe_with_t5)
# elif infer_supports_arg "--disable_mq_norm_probe_with_t5"; then
#   CMD+=(--disable_mq_norm_probe_with_t5)
# fi
# if infer_supports_arg "--mq_norm_match_t5" && [[ "${MQ_NORM_MATCH_T5}" == "1" ]]; then
#   CMD+=(--mq_norm_match_t5)
# elif infer_supports_arg "--disable_mq_norm_match_t5"; then
#   CMD+=(--disable_mq_norm_match_t5)
# fi
# if infer_supports_arg "--mq_norm_match_every_step" && [[ "${MQ_NORM_MATCH_EVERY_STEP}" == "1" ]]; then
#   CMD+=(--mq_norm_match_every_step)
# fi
# if infer_supports_arg "--mq_norm_log_each_step" && [[ "${MQ_NORM_LOG_EACH_STEP}" == "1" ]]; then
#   CMD+=(--mq_norm_log_each_step)
# fi
# if infer_supports_arg "--verify_fail_on_warning" && [[ "${VERIFY_FAIL_ON_WARNING}" == "1" ]]; then
#   CMD+=(--verify_fail_on_warning)
# fi

# echo "[I2V-WANCOND][GUIDE-ONLY] checkpoint=${CHECKPOINT_PATH}"
# echo "[I2V-WANCOND][GUIDE-ONLY] WAN first-frame condition=ENABLED (TI2V animate_ref_slot)"
# echo "[I2V-WANCOND][GUIDE-ONLY] WAN context mode=$( [[ "${WAN_CONTEXT_MQ_ONLY}" == "1" ]] && echo mq_only || echo mq_plus_t5 )"
# echo "[I2V-WANCOND][GUIDE-ONLY] prompt=${PROMPT}"
# echo "[I2V-WANCOND][GUIDE-ONLY] ref_image=${REF_IMAGE}"
# echo "[I2V-WANCOND][GUIDE-ONLY] guide_scale=${GUIDE_SCALE} seed=${SEED}"
# echo "[I2V-WANCOND][GUIDE-ONLY] cfg_uncond_mode=${CFG_UNCOND_MODE}"
# echo "[I2V-WANCOND][GUIDE-ONLY] use_checkpoint_rms_hints=${USE_CHECKPOINT_RMS_HINTS}"
# echo "[I2V-WANCOND][GUIDE-ONLY] mq_norm_probe_with_t5=${MQ_NORM_PROBE_WITH_T5} mq_norm_match_t5=${MQ_NORM_MATCH_T5} warn=[${MQ_NORM_WARN_RATIO_LOW},${MQ_NORM_WARN_RATIO_HIGH}] clip=[${MQ_NORM_MATCH_CLIP_MIN},${MQ_NORM_MATCH_CLIP_MAX}]"
# echo "[I2V-WANCOND][GUIDE-ONLY] mq_norm_match_every_step=${MQ_NORM_MATCH_EVERY_STEP} mq_norm_log_each_step=${MQ_NORM_LOG_EACH_STEP} log_interval=${MQ_NORM_LOG_INTERVAL}"
# echo "[I2V-WANCOND][GUIDE-ONLY] verify_level=${VERIFY_LEVEL} verify_report=${VERIFY_REPORT_PATH}"
# echo "[I2V-WANCOND][GUIDE-ONLY] output=${OUTPUT_PATH}"
# "${CMD[@]}"










# 下面这个是默认首帧强绑定！
set -euo pipefail

# Guide-only frame-cond 版本：
# 1) 默认启用首帧强绑定 (legacy_ref_lock + hard_lock)
# 2) 仅保留 GUIDE_SCALE 作为采样强度控制
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory
export http_proxy=10.130.130.6:56830
export https_proxy=10.130.130.6:56830
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_ti2v_firstframe_wan_cond/ti2v_firstframe_wancond_overfit10_steps803_nummq256_nullimg0.1_nullcap0.1/checkpoint-final}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_ti2v_firstframe_wan_cond/ti2v_firstframe_wancond_overfit10_steps888_nummq256_nullimg0.1_nullcap0.1/checkpoint-final}"
# CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_ti2v_firstframe_wan_cond/ti2v_firstframe_wancond_overfit10_steps889_nummq256_nullimg0.1_nullcap0.1/checkpoint-final}"
# CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_ti2v_firstframe_wan_cond/ti2v_firstframe_wancond_overfit10_steps890_nummq256_nullimg0.1_nullcap0.1/checkpoint-final}"

WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
QWEN3VL_MODEL_ID="${QWEN3VL_MODEL_ID:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"

# PROMPT="A man is smiling."
# PROMPT="A girl is laughing."
# PROMPT="A girl is sad and crying."
# PROMPT="The video features a young man with dark hair and a blue shirt. He is standing in a forested area with trees and foliage in the background. The man appears to be in a thoughtful or contemplative mood, as he gazes off to the side. The lighting in the scene is natural, suggesting it might be daytime. The overall style of the video is realistic, with a focus on the man and his surroundings."
# PROMPT="The video features a woman with gray hair, wearing a yellow and black patterned blouse, sitting in a room with a white door and a light switch on the wall. She appears to be in a state of surprise or shock, as her mouth is open and her eyes are wide. The style of the video is realistic, with a focus on the woman\'s facial expression and the room\'s interior. The lighting in the room is bright, and the colors are vivid. The woman\'s position in the room and her facial expression suggest that she is reacting to something unexpected or startling. The overall mood of the video is tense and dramatic."
PROMPT="The video features a woman with long, dark hair and a serious expression. She is wearing a black leather jacket and gold earrings. The lighting in the video is warm and soft, highlighting her features. The background is blurred, but it appears to be an indoor setting with a window. The woman's gaze is directed off to the side, and she seems to be deep in thought. The overall style of the video is moody and introspective."
# PROMPT="The video is a close-up of a woman with long brown hair smiling at the camera. She is wearing a black top and has a pair of sunglasses on top of her head. The background is blurred, but it appears to be an outdoor setting with trees and a fence. The style of the video is casual and candid, capturing a moment of the woman's day."
# PROMPT="The video features a bald man with a beard, wearing a fur coat. He is shown in three different expressions, each one more intense than the last. The man is looking off to the side, his face contorted in a serious expression. The background is dark and moody, with a hint of a cityscape. The overall style of the video is dramatic and intense, with a focus on the man's expressions and the dark, moody atmosphere."
# PROMPT="The video features a young woman with long black hair, styled in an updo, wearing a white blouse with a high collar. She is looking off to the side with a thoughtful expression. The background is a blurred, warm-toned wall with a subtle floral pattern. The lighting is soft and natural, suggesting an indoor setting with a window nearby. The overall style of the video is elegant and serene, with a focus on the woman's contemplative demeanor."
# PROMPT="The video features a young woman with blonde hair and glasses, smiling and looking to her right. She is wearing a blue shirt and appears to be outdoors, as suggested by the natural light and the blurred background. The style of the video is casual and candid, capturing a moment of the woman's day. The focus is on her face and expression, with the background being out of focus and not the main subject of the video. The lighting suggests it might be daytime."

# PROMPT=""
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref0.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref1.jpg}"
REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref2.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref3.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref4.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref5.jpg}"
# REF_IMAGE="${REF_IMAGE:-/home/liuzhirui/model/Wan2.2/overfit_ref/overfit_ref6.jpg}"

# Option B: provide SOURCE_VIDEO and let this script extract frame-0.
SOURCE_VIDEO="${SOURCE_VIDEO:-}"

FRAME_NUM="${FRAME_NUM:-49}"            # 4n+1
MAX_AREA="${MAX_AREA:-262144}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
GUIDE_SCALE="${GUIDE_SCALE:-1}"
CFG_UNCOND_MODE="${CFG_UNCOND_MODE:-encode_negative}" # encode_negative | zero_mq
USE_CHECKPOINT_RMS_HINTS="${USE_CHECKPOINT_RMS_HINTS:-1}" # 1=未显式传参时，自动跟随checkpoint中的rms训练参数
MQ_NORM_PROBE_WITH_T5_USER_SET=0
if [[ -n "${MQ_NORM_PROBE_WITH_T5+x}" ]]; then MQ_NORM_PROBE_WITH_T5_USER_SET=1; fi
MQ_NORM_MATCH_T5_USER_SET=0
if [[ -n "${MQ_NORM_MATCH_T5+x}" ]]; then MQ_NORM_MATCH_T5_USER_SET=1; fi
MQ_NORM_WARN_RATIO_LOW_USER_SET=0
if [[ -n "${MQ_NORM_WARN_RATIO_LOW+x}" ]]; then MQ_NORM_WARN_RATIO_LOW_USER_SET=1; fi
MQ_NORM_WARN_RATIO_HIGH_USER_SET=0
if [[ -n "${MQ_NORM_WARN_RATIO_HIGH+x}" ]]; then MQ_NORM_WARN_RATIO_HIGH_USER_SET=1; fi
MQ_NORM_MATCH_CLIP_MIN_USER_SET=0
if [[ -n "${MQ_NORM_MATCH_CLIP_MIN+x}" ]]; then MQ_NORM_MATCH_CLIP_MIN_USER_SET=1; fi
MQ_NORM_MATCH_CLIP_MAX_USER_SET=0
if [[ -n "${MQ_NORM_MATCH_CLIP_MAX+x}" ]]; then MQ_NORM_MATCH_CLIP_MAX_USER_SET=1; fi
MQ_NORM_PROBE_WITH_T5="${MQ_NORM_PROBE_WITH_T5:-1}"   # 1=对比 MQ/T5 token RMS
MQ_NORM_MATCH_T5="${MQ_NORM_MATCH_T5:-1}"             # 1=按 T5 RMS 对齐 MQ RMS（默认开启）
MQ_NORM_WARN_RATIO_LOW="${MQ_NORM_WARN_RATIO_LOW:-0.25}"
MQ_NORM_WARN_RATIO_HIGH="${MQ_NORM_WARN_RATIO_HIGH:-4.0}"
MQ_NORM_MATCH_CLIP_MIN="${MQ_NORM_MATCH_CLIP_MIN:-0.03}"
MQ_NORM_MATCH_CLIP_MAX="${MQ_NORM_MATCH_CLIP_MAX:-4.0}"
MQ_NORM_MATCH_EVERY_STEP="${MQ_NORM_MATCH_EVERY_STEP:-1}" # 1=每个去噪step都重新执行MQ RMS缩放
MQ_NORM_LOG_EACH_STEP="${MQ_NORM_LOG_EACH_STEP:-1}"       # 1=每步打印缩放前后RMS与scale
MQ_NORM_LOG_INTERVAL="${MQ_NORM_LOG_INTERVAL:-1}"         # 1=每步打印
SHIFT="${SHIFT:-5.0}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}" # unipc | dpm++
SEED="${SEED:-42}"
FIRST_FRAME_STRONG_BIND="${FIRST_FRAME_STRONG_BIND:-1}" # 1=legacy_ref_lock + hard_lock
VERIFY_LEVEL="${VERIFY_LEVEL:-basic}"   # none | basic | full
VERIFY_FAIL_ON_WARNING="${VERIFY_FAIL_ON_WARNING:-0}" # 1 => warning直接失败

NUM_METAQUERIES="${NUM_METAQUERIES:-256}"
CONNECTOR_LAYERS="${CONNECTOR_LAYERS:-24}"
DEVICE="${DEVICE:-0}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-1}"     # 1=enable --offload_model
WAN_CONTEXT_MQ_ONLY="${WAN_CONTEXT_MQ_ONLY:-1}" # 1: WAN context仅保留MQ特征(不拼接T5)
SHOW_CHECKPOINT_TRAINING_HINTS="${SHOW_CHECKPOINT_TRAINING_HINTS:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-/home/liuzhirui/model/Wan2.2/inference_outputs_20260407}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/ti2v_firstframe_wancond_guide_only_seed${SEED}_${RUN_TS}.mp4}"
VERIFY_REPORT_PATH="${VERIFY_REPORT_PATH:-${OUTPUT_PATH}.verify.json}"
I2V_INFER_ENTRY="${SCRIPT_DIR}/inference_metaquery_wan_rms_aligned.py"

if [[ -n "${SOURCE_VIDEO}" ]]; then
  EXTRACTED_REF_IMAGE="${EXTRACTED_REF_IMAGE:-${OUTPUT_DIR}/firstframe_wancond_${RUN_TS}.png}"
  export SOURCE_VIDEO EXTRACTED_REF_IMAGE
  "${PYTHON_BIN}" - << 'PY'
import cv2
import os
from pathlib import Path

src = Path(os.environ["SOURCE_VIDEO"]).expanduser()
dst = Path(os.environ["EXTRACTED_REF_IMAGE"]).expanduser()
dst.parent.mkdir(parents=True, exist_ok=True)

if not src.exists():
    raise FileNotFoundError(f"SOURCE_VIDEO not found: {src}")

cap = cv2.VideoCapture(str(src))
if not cap.isOpened():
    raise RuntimeError(f"Failed to open SOURCE_VIDEO: {src}")
ok, frame = cap.read()
cap.release()
if (not ok) or frame is None:
    raise RuntimeError(f"Failed to read first frame from: {src}")

if not cv2.imwrite(str(dst), frame):
    raise RuntimeError(f"Failed to write extracted frame to: {dst}")

print(f"[I2V-WANCOND] extracted_ref_image={dst}")
PY
  REF_IMAGE="${EXTRACTED_REF_IMAGE}"
fi

if [[ -z "${REF_IMAGE}" ]]; then
  echo "[ERROR] REF_IMAGE is empty. Set REF_IMAGE or SOURCE_VIDEO."
  exit 2
fi
if [[ ! -f "${REF_IMAGE}" ]]; then
  echo "[ERROR] REF_IMAGE not found: ${REF_IMAGE}"
  exit 3
fi

if [[ "${SHOW_CHECKPOINT_TRAINING_HINTS}" == "1" ]]; then
  export CHECKPOINT_PATH
  "${PYTHON_BIN}" - << 'PY'
import json
import os
from pathlib import Path

ckpt = Path(os.environ["CHECKPOINT_PATH"]).expanduser()
candidates = []
if ckpt.is_dir():
    candidates.extend([ckpt / "config.json", ckpt / "training_args.json"])
else:
    candidates.extend([ckpt.parent / "config.json", ckpt.parent / "training_args.json"])

cfg = None
cfg_path = None
for p in candidates:
    if p.exists():
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            cfg_path = p
            break
        except Exception:
            pass

if cfg is None:
    print("[I2V-WANCOND][INFER] checkpoint training-hints: not found (config.json/training_args.json)")
else:
    def g(k, d="N/A"):
        v = cfg.get(k, d)
        return v if v is not None else d

    print(f"[I2V-WANCOND][INFER] checkpoint cfg file={cfg_path}")
    print(
        "[I2V-WANCOND][INFER] checkpoint training-hints "
        f"lr_scheduler_type={g('lr_scheduler_type')} "
        f"warmup_steps={g('warmup_steps')} cooldown_steps={g('cooldown_steps')} "
        f"enable_loss_early_stop={g('enable_loss_early_stop')} "
        f"loss_early_stop_min_step={g('loss_early_stop_min_step')} "
        f"loss_early_stop_threshold={g('loss_early_stop_threshold')} "
        f"wan_train_mode={g('wan_train_mode')} "
        f"wan_lr_ratio={g('wan_lr_ratio')}"
    )
PY
fi

if [[ "${USE_CHECKPOINT_RMS_HINTS}" == "1" ]]; then
  export CHECKPOINT_PATH
  CKPT_RMS_HINTS_RAW="$("${PYTHON_BIN}" - << 'PY'
import json
import os
from pathlib import Path

ckpt = Path(os.environ["CHECKPOINT_PATH"]).expanduser()
candidates = []
if ckpt.is_dir():
    candidates.extend([ckpt / "training_args.json", ckpt / "config.json"])
else:
    candidates.extend([ckpt.parent / "training_args.json", ckpt.parent / "config.json"])

cfg = None
for p in candidates:
    if p.exists():
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            break
        except Exception:
            pass

if not isinstance(cfg, dict):
    raise SystemExit(0)

def emit_bool(src_key: str, dst_key: str) -> None:
    if src_key in cfg:
        v = cfg.get(src_key)
        if isinstance(v, bool):
            print(f"{dst_key}={1 if v else 0}")
        elif isinstance(v, (int, float)):
            print(f"{dst_key}={1 if float(v) != 0.0 else 0}")

def emit_float(src_key: str, dst_key: str) -> None:
    if src_key in cfg:
        v = cfg.get(src_key)
        if isinstance(v, (int, float)):
            print(f"{dst_key}={float(v)}")
        else:
            try:
                print(f"{dst_key}={float(str(v).strip())}")
            except Exception:
                pass

emit_bool("mq_norm_probe_with_t5", "CKPT_MQ_NORM_PROBE_WITH_T5")
emit_bool("mq_norm_match_t5", "CKPT_MQ_NORM_MATCH_T5")
emit_float("mq_norm_warn_ratio_low", "CKPT_MQ_NORM_WARN_RATIO_LOW")
emit_float("mq_norm_warn_ratio_high", "CKPT_MQ_NORM_WARN_RATIO_HIGH")
emit_float("mq_norm_match_clip_min", "CKPT_MQ_NORM_MATCH_CLIP_MIN")
emit_float("mq_norm_match_clip_max", "CKPT_MQ_NORM_MATCH_CLIP_MAX")
PY
)"

  if [[ -n "${CKPT_RMS_HINTS_RAW}" ]]; then
    eval "${CKPT_RMS_HINTS_RAW}"
    if [[ "${MQ_NORM_PROBE_WITH_T5_USER_SET}" != "1" && -n "${CKPT_MQ_NORM_PROBE_WITH_T5:-}" ]]; then
      MQ_NORM_PROBE_WITH_T5="${CKPT_MQ_NORM_PROBE_WITH_T5}"
    fi
    if [[ "${MQ_NORM_MATCH_T5_USER_SET}" != "1" && -n "${CKPT_MQ_NORM_MATCH_T5:-}" ]]; then
      MQ_NORM_MATCH_T5="${CKPT_MQ_NORM_MATCH_T5}"
    fi
    if [[ "${MQ_NORM_WARN_RATIO_LOW_USER_SET}" != "1" && -n "${CKPT_MQ_NORM_WARN_RATIO_LOW:-}" ]]; then
      MQ_NORM_WARN_RATIO_LOW="${CKPT_MQ_NORM_WARN_RATIO_LOW}"
    fi
    if [[ "${MQ_NORM_WARN_RATIO_HIGH_USER_SET}" != "1" && -n "${CKPT_MQ_NORM_WARN_RATIO_HIGH:-}" ]]; then
      MQ_NORM_WARN_RATIO_HIGH="${CKPT_MQ_NORM_WARN_RATIO_HIGH}"
    fi
    if [[ "${MQ_NORM_MATCH_CLIP_MIN_USER_SET}" != "1" && -n "${CKPT_MQ_NORM_MATCH_CLIP_MIN:-}" ]]; then
      MQ_NORM_MATCH_CLIP_MIN="${CKPT_MQ_NORM_MATCH_CLIP_MIN}"
    fi
    if [[ "${MQ_NORM_MATCH_CLIP_MAX_USER_SET}" != "1" && -n "${CKPT_MQ_NORM_MATCH_CLIP_MAX:-}" ]]; then
      MQ_NORM_MATCH_CLIP_MAX="${CKPT_MQ_NORM_MATCH_CLIP_MAX}"
    fi
  fi
fi

INFER_HELP_CACHE=""
load_infer_help() {
  if [[ -n "${INFER_HELP_CACHE}" ]]; then
    return
  fi
  INFER_HELP_CACHE="$("${PYTHON_BIN}" "${I2V_INFER_ENTRY}" --help 2>&1 || true)"
}

infer_supports_arg() {
  local arg_name="$1"
  load_infer_help
  if [[ "${INFER_HELP_CACHE}" == *"${arg_name}"* ]]; then
    return 0
  fi
  return 1
}

CMD=(
  "${PYTHON_BIN}" "${I2V_INFER_ENTRY}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --wan_checkpoint_dir "${WAN_CHECKPOINT_DIR}"
  --qwen3vl_model_id "${QWEN3VL_MODEL_ID}"
  --mode i2v
  --prompt "${PROMPT}"
  --ref_image "${REF_IMAGE}"
  --negative_prompt "${NEGATIVE_PROMPT}"
  --frame_num "${FRAME_NUM}"
  --max_area "${MAX_AREA}"
  --sampling_steps "${SAMPLING_STEPS}"
  --guide_scale "${GUIDE_SCALE}"
  --shift "${SHIFT}"
  --sample_solver "${SAMPLE_SOLVER}"
  --seed "${SEED}"
  --num_metaqueries "${NUM_METAQUERIES}"
  --connector_num_hidden_layers "${CONNECTOR_LAYERS}"
  --device "${DEVICE}"
  --output_path "${OUTPUT_PATH}"
)

if infer_supports_arg "--i2v_method"; then
  if [[ "${FIRST_FRAME_STRONG_BIND}" == "1" ]]; then
    CMD+=(--i2v_method legacy_ref_lock)
  else
    CMD+=(--i2v_method animate_ref_slot)
  fi
fi
if infer_supports_arg "--i2v_ref_strategy" && [[ "${FIRST_FRAME_STRONG_BIND}" == "1" ]]; then
  CMD+=(--i2v_ref_strategy hard_lock)
fi
if infer_supports_arg "--cfg_uncond_mode"; then
  CMD+=(--cfg_uncond_mode "${CFG_UNCOND_MODE}")
fi
if infer_supports_arg "--mq_norm_warn_ratio_low"; then
  CMD+=(--mq_norm_warn_ratio_low "${MQ_NORM_WARN_RATIO_LOW}")
fi
if infer_supports_arg "--mq_norm_warn_ratio_high"; then
  CMD+=(--mq_norm_warn_ratio_high "${MQ_NORM_WARN_RATIO_HIGH}")
fi
if infer_supports_arg "--mq_norm_match_clip_min"; then
  CMD+=(--mq_norm_match_clip_min "${MQ_NORM_MATCH_CLIP_MIN}")
fi
if infer_supports_arg "--mq_norm_match_clip_max"; then
  CMD+=(--mq_norm_match_clip_max "${MQ_NORM_MATCH_CLIP_MAX}")
fi
if infer_supports_arg "--mq_norm_log_interval"; then
  CMD+=(--mq_norm_log_interval "${MQ_NORM_LOG_INTERVAL}")
fi
if infer_supports_arg "--verify_level"; then
  CMD+=(--verify_level "${VERIFY_LEVEL}")
fi
if infer_supports_arg "--verify_report_path"; then
  CMD+=(--verify_report_path "${VERIFY_REPORT_PATH}")
fi

if [[ "${OFFLOAD_MODEL}" == "1" ]]; then
  CMD+=(--offload_model)
fi
if infer_supports_arg "--mq_norm_probe_with_t5" && [[ "${MQ_NORM_PROBE_WITH_T5}" == "1" ]]; then
  CMD+=(--mq_norm_probe_with_t5)
elif infer_supports_arg "--disable_mq_norm_probe_with_t5"; then
  CMD+=(--disable_mq_norm_probe_with_t5)
fi
if infer_supports_arg "--mq_norm_match_t5" && [[ "${MQ_NORM_MATCH_T5}" == "1" ]]; then
  CMD+=(--mq_norm_match_t5)
elif infer_supports_arg "--disable_mq_norm_match_t5"; then
  CMD+=(--disable_mq_norm_match_t5)
fi
if infer_supports_arg "--mq_norm_match_every_step" && [[ "${MQ_NORM_MATCH_EVERY_STEP}" == "1" ]]; then
  CMD+=(--mq_norm_match_every_step)
fi
if infer_supports_arg "--mq_norm_log_each_step" && [[ "${MQ_NORM_LOG_EACH_STEP}" == "1" ]]; then
  CMD+=(--mq_norm_log_each_step)
fi
if infer_supports_arg "--verify_fail_on_warning" && [[ "${VERIFY_FAIL_ON_WARNING}" == "1" ]]; then
  CMD+=(--verify_fail_on_warning)
fi

echo "[I2V-WANCOND][GUIDE-ONLY] checkpoint=${CHECKPOINT_PATH}"
echo "[I2V-WANCOND][GUIDE-ONLY] first_frame_strong_bind=${FIRST_FRAME_STRONG_BIND}"
if [[ "${FIRST_FRAME_STRONG_BIND}" == "1" ]]; then
  echo "[I2V-WANCOND][GUIDE-ONLY] WAN first-frame condition=ENABLED (legacy_ref_lock + hard_lock)"
else
  echo "[I2V-WANCOND][GUIDE-ONLY] WAN first-frame condition=ENABLED (animate_ref_slot)"
fi
echo "[I2V-WANCOND][GUIDE-ONLY] WAN context mode=$( [[ "${WAN_CONTEXT_MQ_ONLY}" == "1" ]] && echo mq_only || echo mq_plus_t5 )"
echo "[I2V-WANCOND][GUIDE-ONLY] prompt=${PROMPT}"
echo "[I2V-WANCOND][GUIDE-ONLY] ref_image=${REF_IMAGE}"
echo "[I2V-WANCOND][GUIDE-ONLY] guide_scale=${GUIDE_SCALE} seed=${SEED}"
echo "[I2V-WANCOND][GUIDE-ONLY] cfg_uncond_mode=${CFG_UNCOND_MODE}"
echo "[I2V-WANCOND][GUIDE-ONLY] use_checkpoint_rms_hints=${USE_CHECKPOINT_RMS_HINTS}"
echo "[I2V-WANCOND][GUIDE-ONLY] mq_norm_probe_with_t5=${MQ_NORM_PROBE_WITH_T5} mq_norm_match_t5=${MQ_NORM_MATCH_T5} warn=[${MQ_NORM_WARN_RATIO_LOW},${MQ_NORM_WARN_RATIO_HIGH}] clip=[${MQ_NORM_MATCH_CLIP_MIN},${MQ_NORM_MATCH_CLIP_MAX}]"
echo "[I2V-WANCOND][GUIDE-ONLY] mq_norm_match_every_step=${MQ_NORM_MATCH_EVERY_STEP} mq_norm_log_each_step=${MQ_NORM_LOG_EACH_STEP} log_interval=${MQ_NORM_LOG_INTERVAL}"
echo "[I2V-WANCOND][GUIDE-ONLY] verify_level=${VERIFY_LEVEL} verify_report=${VERIFY_REPORT_PATH}"
echo "[I2V-WANCOND][GUIDE-ONLY] output=${OUTPUT_PATH}"
"${CMD[@]}"
