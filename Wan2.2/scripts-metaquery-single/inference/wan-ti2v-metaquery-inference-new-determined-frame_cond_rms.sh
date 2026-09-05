#!/bin/bash
set -euo pipefail

# Single-GPU TI2V inference launcher with explicit first-frame conditioning on BOTH:
# 1) MQ side (Qwen3-VL image input)
# 2) WAN side (TI2V animate_ref_slot: first-frame reference slots)
#
# Usage:
#   - Option A: set REF_IMAGE directly
#   - Option B: set SOURCE_VIDEO and auto-extract frame-0 as REF_IMAGE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_ti2v_firstframe_wan_cond/ti2v_firstframe_wancond_overfit10_steps800_nummq256_nullimg0.0_nullcap0.0/checkpoint-final}"
WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
QWEN3VL_MODEL_ID="${QWEN3VL_MODEL_ID:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"

PROMPT="${PROMPT:-A person is speaking to camera with natural expression and smooth motion.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

# Option A: provide REF_IMAGE directly.
REF_IMAGE="${REF_IMAGE:-}"

# Option B: provide SOURCE_VIDEO and let this script extract frame-0.
SOURCE_VIDEO="${SOURCE_VIDEO:-}"

FRAME_NUM="${FRAME_NUM:-49}"            # 4n+1
MAX_AREA="${MAX_AREA:-262144}"
SAMPLING_STEPS="${SAMPLING_STEPS:-40}"
GUIDE_SCALE_LOW="${GUIDE_SCALE_LOW:-5.0}"
GUIDE_SCALE_HIGH="${GUIDE_SCALE_HIGH:-5.0}"
GUIDE_SCALE="${GUIDE_SCALE:-${GUIDE_SCALE_LOW}}"
CFG_UNCOND_MODE="${CFG_UNCOND_MODE:-encode_negative}" # encode_negative | zero_mq
MQ_NORM_PROBE_WITH_T5="${MQ_NORM_PROBE_WITH_T5:-1}"   # 1=对比 MQ/T5 token RMS
MQ_NORM_MATCH_T5="${MQ_NORM_MATCH_T5:-1}"             # 1=按 T5 RMS 对齐 MQ RMS（默认开启，和训练侧一致）
MQ_NORM_WARN_RATIO_LOW="${MQ_NORM_WARN_RATIO_LOW:-0.25}"
MQ_NORM_WARN_RATIO_HIGH="${MQ_NORM_WARN_RATIO_HIGH:-4.0}"
MQ_NORM_MATCH_CLIP_MIN="${MQ_NORM_MATCH_CLIP_MIN:-0.03}"
MQ_NORM_MATCH_CLIP_MAX="${MQ_NORM_MATCH_CLIP_MAX:-4.0}"
SHIFT="${SHIFT:-5.0}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-unipc}" # unipc | dpm++
SEED="${SEED:--1}"

NUM_METAQUERIES="${NUM_METAQUERIES:-256}"
CONNECTOR_LAYERS="${CONNECTOR_LAYERS:-24}"
DEVICE="${DEVICE:-0}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-1}"     # 1=enable --offload_model
WAN_CONTEXT_MQ_ONLY="${WAN_CONTEXT_MQ_ONLY:-1}" # 1: WAN context仅保留MQ特征(不拼接T5)
SHOW_CHECKPOINT_TRAINING_HINTS="${SHOW_CHECKPOINT_TRAINING_HINTS:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-/home/liuzhirui/model/Wan2.2/inference_outputs}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/ti2v_firstframe_wancond_seed${SEED}_${RUN_TS}.mp4}"
I2V_INFER_ENTRY="${SCRIPT_DIR}/inference_metaquery_wan_animate_like_v2.py"

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

CMD=(
  "${PYTHON_BIN}" "${I2V_INFER_ENTRY}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --wan_checkpoint_dir "${WAN_CHECKPOINT_DIR}"
  --qwen3vl_model_id "${QWEN3VL_MODEL_ID}"
  --mode i2v
  --i2v_method animate_ref_slot
  --prompt "${PROMPT}"
  --ref_image "${REF_IMAGE}"
  --negative_prompt "${NEGATIVE_PROMPT}"
  --frame_num "${FRAME_NUM}"
  --max_area "${MAX_AREA}"
  --sampling_steps "${SAMPLING_STEPS}"
  --guide_scale "${GUIDE_SCALE}"
  --cfg_uncond_mode "${CFG_UNCOND_MODE}"
  --mq_norm_warn_ratio_low "${MQ_NORM_WARN_RATIO_LOW}"
  --mq_norm_warn_ratio_high "${MQ_NORM_WARN_RATIO_HIGH}"
  --mq_norm_match_clip_min "${MQ_NORM_MATCH_CLIP_MIN}"
  --mq_norm_match_clip_max "${MQ_NORM_MATCH_CLIP_MAX}"
  --shift "${SHIFT}"
  --sample_solver "${SAMPLE_SOLVER}"
  --seed "${SEED}"
  --num_metaqueries "${NUM_METAQUERIES}"
  --connector_num_hidden_layers "${CONNECTOR_LAYERS}"
  --device "${DEVICE}"
  --output_path "${OUTPUT_PATH}"
)

if [[ "${OFFLOAD_MODEL}" == "1" ]]; then
  CMD+=(--offload_model)
fi
if [[ "${MQ_NORM_PROBE_WITH_T5}" == "1" ]]; then
  CMD+=(--mq_norm_probe_with_t5)
else
  CMD+=(--disable_mq_norm_probe_with_t5)
fi
if [[ "${MQ_NORM_MATCH_T5}" == "1" ]]; then
  CMD+=(--mq_norm_match_t5)
fi

echo "[I2V-WANCOND][INFER] checkpoint=${CHECKPOINT_PATH}"
echo "[I2V-WANCOND][INFER] WAN first-frame condition=ENABLED (TI2V animate_ref_slot)"
echo "[I2V-WANCOND][INFER] WAN context mode=$( [[ "${WAN_CONTEXT_MQ_ONLY}" == "1" ]] && echo mq_only || echo mq_plus_t5 )"
echo "[I2V-WANCOND][INFER] prompt=${PROMPT}"
echo "[I2V-WANCOND][INFER] ref_image=${REF_IMAGE}"
echo "[I2V-WANCOND][INFER] guide_scale=${GUIDE_SCALE} seed=${SEED}"
echo "[I2V-WANCOND][INFER] cfg_uncond_mode=${CFG_UNCOND_MODE}"
echo "[I2V-WANCOND][INFER] mq_norm_probe_with_t5=${MQ_NORM_PROBE_WITH_T5} mq_norm_match_t5=${MQ_NORM_MATCH_T5} warn=[${MQ_NORM_WARN_RATIO_LOW},${MQ_NORM_WARN_RATIO_HIGH}]"
echo "[I2V-WANCOND][INFER] output=${OUTPUT_PATH}"
"${CMD[@]}"
