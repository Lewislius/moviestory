#!/bin/bash
set -euo pipefail

# Usage:
#   bash train_stage1_openvid_local_metaquery_overfit10_unfreeze_dit.sh
#
# Goal:
#   固定 10 条 OpenVid 视频进行 overfit，训练 DiT + MetaQuery + Connector，
#   仅冻结 MLLM backbone（除可选 MQ embedding）、T5、VAE。
#   尽量复用原 overfit10 脚本的结构与日志风格。

source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

export http_proxy="${http_proxy:-10.130.130.6:56830}"
export https_proxy="${https_proxy:-10.130.130.6:56830}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-20}"
LOG_STEPS="${LOG_STEPS:-1}"
LOG_EVERY_STEP="${LOG_EVERY_STEP:-1}"
SEED="${SEED:-42}"

OPENVID_VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-/home/liuzhirui/dataset/OpenVid-1M/video}"
OPENVID_CSV_PATH="${OPENVID_CSV_PATH:-/home/liuzhirui/dataset/OpenVid-1M/data/train/OpenVid-1M.csv}"

QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
WAN_TI2V_CKPT="${WAN_TI2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_unfreeze_dit}"
OVERFIT_ROOT="${OVERFIT_ROOT:-/home/liuzhirui/model/Wan2.2/tmp/metaquery_overfit10_unfreeze_dit}"
OVERFIT_ROOT="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${OVERFIT_ROOT}")"
mkdir -p "${OVERFIT_ROOT}"

VIDEO_LIST_FILE="${OVERFIT_ROOT}/video_list_overfit10.txt"
MINI_VIDEO_ROOT="${OVERFIT_ROOT}/videos"
MINI_CSV_PATH="${OVERFIT_ROOT}/openvid_overfit10.csv"
PAIR_REPORT_JSON="${OVERFIT_ROOT}/pair_report.json"

cat > "${VIDEO_LIST_FILE}" << 'EOF'
/home/liuzhirui/dataset/OpenVid-1M/video/celebv___f2KtcXAxI_1.mp4
/home/liuzhirui/dataset/OpenVid-1M/video/celebv___lRwnjxeCg_3.mp4
/home/liuzhirui/dataset/OpenVid-1M/video/celebv__0yBbZJZqG8_1.mp4
/home/liuzhirui/dataset/OpenVid-1M/video/celebv__1osQSmJ2-s_4.mp4
/home/liuzhirui/dataset/OpenVid-1M/video/celebv__1s0YQ8dL04_0.mp4
/home/liuzhirui/dataset/OpenVid-1M/video/celebv__4DRXSdPuCo_2.mp4
/home/liuzhirui/dataset/OpenVid-1M/video/celebv__4DRXSdPuCo_4.mp4
/home/liuzhirui/dataset/OpenVid-1M/video/celebv__4juqo20ABE_0.mp4
/home/liuzhirui/dataset/OpenVid-1M/video/celebv__5ukjsqqLg4_12.mp4
/home/liuzhirui/dataset/OpenVid-1M/video/celebv__6RI-8Ia4do_0.mp4
EOF

rm -rf "${MINI_VIDEO_ROOT}"
mkdir -p "${MINI_VIDEO_ROOT}"

export VIDEO_LIST_FILE MINI_VIDEO_ROOT MINI_CSV_PATH PAIR_REPORT_JSON OPENVID_VIDEO_ROOT OPENVID_CSV_PATH
python - << 'PY'
import csv
import json
import os
import shutil
from pathlib import Path

video_list_file = Path(os.environ["VIDEO_LIST_FILE"])
mini_video_root = Path(os.environ["MINI_VIDEO_ROOT"])
mini_csv_path = Path(os.environ["MINI_CSV_PATH"])
pair_report_json = Path(os.environ["PAIR_REPORT_JSON"])
openvid_root = Path(os.environ["OPENVID_VIDEO_ROOT"]).resolve()
openvid_csv = Path(os.environ["OPENVID_CSV_PATH"]).resolve()

def norm_key(s: str) -> str:
    s = str(s).strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s.lstrip("/").lower()

def find_cols(fieldnames):
    lowered = {str(k).strip().lower(): k for k in (fieldnames or [])}
    video_col = None
    caption_col = None
    for k in ("video", "video_path", "filename", "file", "path"):
        if k in lowered:
            video_col = lowered[k]
            break
    for k in ("caption", "text", "description", "prompt", "summary"):
        if k in lowered:
            caption_col = lowered[k]
            break
    return video_col, caption_col

if not openvid_csv.exists():
    raise FileNotFoundError(f"OpenVid csv not found: {openvid_csv}")

path_to_caption = {}
name_to_caption = {}
with open(openvid_csv, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    video_col, caption_col = find_cols(reader.fieldnames)
    if video_col is None or caption_col is None:
        raise RuntimeError(f"Cannot find video/caption columns from {openvid_csv}")
    for row in reader:
        v = str(row.get(video_col, "") or "").strip()
        c = str(row.get(caption_col, "") or "").strip()
        if not v or not c:
            continue
        vk = norm_key(v)
        nk = norm_key(Path(v).name)
        if vk and (vk not in path_to_caption or len(c) > len(path_to_caption[vk])):
            path_to_caption[vk] = c
        if nk and (nk not in name_to_caption or len(c) > len(name_to_caption[nk])):
            name_to_caption[nk] = c

kept = []
dropped = []

videos = [ln.strip() for ln in video_list_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
for idx, raw in enumerate(videos):
    src = Path(raw).resolve()
    if not src.exists():
        dropped.append({"video": str(src), "reason": "video_not_found"})
        print(f"[OVERFIT10-UNFREEZE][DROP] video={src} reason=video_not_found")
        continue

    rel_key = ""
    try:
        rel_key = norm_key(src.relative_to(openvid_root))
    except Exception:
        rel_key = norm_key(src.name)

    cap = path_to_caption.get(rel_key)
    if cap is None:
        cap = name_to_caption.get(norm_key(src.name))
    if not cap:
        dropped.append({"video": str(src), "reason": "caption_not_found"})
        print(f"[OVERFIT10-UNFREEZE][DROP] video={src} reason=caption_not_found")
        continue

    target_name = src.name
    if (mini_video_root / target_name).exists():
        target_name = f"{idx:02d}_{src.name}"
    dst = mini_video_root / target_name

    linked = False
    try:
        os.symlink(src, dst)
        linked = True
    except Exception:
        try:
            os.link(src, dst)
            linked = True
        except Exception:
            shutil.copy2(src, dst)
            linked = False

    kept.append({"video": target_name, "caption": cap, "src": str(src), "link_type": "symlink_or_hardlink" if linked else "copy"})
    print(f"[OVERFIT10-UNFREEZE][KEEP] video={target_name} src={src} caption={cap}")

with open(mini_csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["video", "caption"])
    writer.writeheader()
    for row in kept:
        writer.writerow({"video": row["video"], "caption": row["caption"]})

report = {
    "openvid_csv": str(openvid_csv),
    "openvid_root": str(openvid_root),
    "mini_video_root": str(mini_video_root.resolve()),
    "mini_csv": str(mini_csv_path.resolve()),
    "kept_count": len(kept),
    "drop_count": len(dropped),
    "kept": kept,
    "dropped": dropped,
}
pair_report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OVERFIT10-UNFREEZE] mini_csv={mini_csv_path.resolve()} kept={len(kept)} dropped={len(dropped)}")
print(f"[OVERFIT10-UNFREEZE] pair_report={pair_report_json.resolve()}")
PY

if [[ ! -f "${MINI_CSV_PATH}" ]]; then
  echo "[ERROR] mini csv not generated: ${MINI_CSV_PATH}"
  exit 1
fi

KEPT_COUNT="$(python - << 'PY'
import csv, os
from pathlib import Path
p = Path(os.environ["MINI_CSV_PATH"])
if not p.exists():
    print(0)
else:
    with open(p, "r", encoding="utf-8", newline="") as f:
        n = sum(1 for _ in csv.DictReader(f))
    print(n)
PY
)"

if [[ "${KEPT_COUNT}" -le 0 ]]; then
  echo "[ERROR] kept_count=${KEPT_COUNT}, nothing to train."
  exit 2
fi

echo "[OVERFIT10-UNFREEZE] final_kept_count=${KEPT_COUNT}"
echo "[OVERFIT10-UNFREEZE] mini_video_root=${MINI_VIDEO_ROOT}"
echo "[OVERFIT10-UNFREEZE] mini_csv=${MINI_CSV_PATH}"

# Dataset debug print
export WAN_LOCAL_OVERFIT_DEBUG=1
export WAN_LOCAL_MISSING_CAPTION_PRINT_MAX=1000
export WAN_DATA_PRECLEAN=1

WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-wan-metaquery-overfit10-unfreeze-dit}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-overfit10-unfreeze-dit-steps${NUM_TRAIN_STEPS}-$(date +%Y%m%d-%H%M%S)}"
WANDB_TAGS="${WANDB_TAGS:-overfit,debug,unfreeze_dit}"
WANDB_MODE="${WANDB_MODE:-offline}"

LOG_CUDA_MEMORY="${LOG_CUDA_MEMORY:-1}"
WANDB_LOG_EVERY_STEP="${WANDB_LOG_EVERY_STEP:-1}"
WANDB_LOG_CHECKPOINT="${WANDB_LOG_CHECKPOINT:-1}"

NUM_METAQUERIES="${NUM_METAQUERIES:-64}"
NULL_IMAGE_PROB="${NULL_IMAGE_PROB:-0.2}"
NULL_CAPTION_PROB="${NULL_CAPTION_PROB:-0.0}"

TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-49}"
TI2V_MAX_AREA="${TI2V_MAX_AREA:-262144}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-0.3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

# 这几个是 train_metaquery_wan_unfreeze_dit.py 读取的环境变量
export WAN_UNFREEZE_DIT_LR="${WAN_UNFREEZE_DIT_LR:-5e-6}"
export WAN_UNFREEZE_MQ_LR="${WAN_UNFREEZE_MQ_LR:-2e-5}"
export WAN_UNFREEZE_DIT_WEIGHT_DECAY="${WAN_UNFREEZE_DIT_WEIGHT_DECAY:-0.01}"
export WAN_UNFREEZE_MQ_WEIGHT_DECAY="${WAN_UNFREEZE_MQ_WEIGHT_DECAY:-0.1}"
export WAN_UNFREEZE_SAVE_OPTIMIZER="${WAN_UNFREEZE_SAVE_OPTIMIZER:-0}"
export WAN_UNFREEZE_SAVE_DIT_FULL="${WAN_UNFREEZE_SAVE_DIT_FULL:-0}"
export WAN_UNFREEZE_SAVE_DIT_FULL_EVERY="${WAN_UNFREEZE_SAVE_DIT_FULL_EVERY:-0}"

METRICS_JSONL_PATH="${METRICS_JSONL_PATH:-${OUTPUT_ROOT}/logs/ti2v_overfit10_unfreeze_dit_steps${NUM_TRAIN_STEPS}_$(date +%Y%m%d-%H%M%S).jsonl}"
METRICS_JSONL_PATH="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${METRICS_JSONL_PATH}")"

COMMON_ARGS=(
  --t5_cpu
  --mq_gradient_checkpointing
  --aggressive_empty_cache
  --local_openvid_video_root "${MINI_VIDEO_ROOT}"
  --local_openvid_csv_path "${MINI_CSV_PATH}"
  --local_openvid_limit "${KEPT_COUNT}"
  --local_openvid_hd_video_root ""
  --local_openvid_hd_csv_path ""
  --qwen3vl_model_id "${QWEN_MODEL}"
  --num_metaqueries "${NUM_METAQUERIES}"
  --null_image_prob "${NULL_IMAGE_PROB}"
  --null_caption_prob "${NULL_CAPTION_PROB}"
  --num_train_steps "${NUM_TRAIN_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --log_steps "${LOG_STEPS}"
  --seed "${SEED}"
  --learning_rate "${LEARNING_RATE}"
  --metrics_jsonl_path "${METRICS_JSONL_PATH}"
)

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

RUN_OUTPUT_DIR="${OUTPUT_ROOT}/ti2v_overfit10_unfreeze_dit_steps${NUM_TRAIN_STEPS}"
RUN_SCRIPT="/home/liuzhirui/model/Wan2.2/train_metaquery_wan_unfreeze_dit.py"

echo "[LAUNCH][OVERFIT10-UNFREEZE] steps=${NUM_TRAIN_STEPS} kept=${KEPT_COUNT}"
echo "[LAUNCH][OVERFIT10-UNFREEZE] mini_root=${MINI_VIDEO_ROOT}"
echo "[LAUNCH][OVERFIT10-UNFREEZE] mini_csv=${MINI_CSV_PATH}"
echo "[LAUNCH][OVERFIT10-UNFREEZE] metrics_jsonl=${METRICS_JSONL_PATH}"
echo "[LAUNCH][OVERFIT10-UNFREEZE] null_image_prob=${NULL_IMAGE_PROB} null_caption_prob=${NULL_CAPTION_PROB}"
echo "[LAUNCH][OVERFIT10-UNFREEZE] lr(mq)=${WAN_UNFREEZE_MQ_LR} lr(dit)=${WAN_UNFREEZE_DIT_LR} base_lr=${LEARNING_RATE}"
echo "[LAUNCH][OVERFIT10-UNFREEZE] save_optimizer=${WAN_UNFREEZE_SAVE_OPTIMIZER} save_dit_full=${WAN_UNFREEZE_SAVE_DIT_FULL}"

python "${RUN_SCRIPT}" \
  --wan_checkpoint_dir "${WAN_TI2V_CKPT}" \
  --output_dir "${RUN_OUTPUT_DIR}" \
  --frame_num "${TI2V_FRAME_NUM}" \
  --max_area "${TI2V_MAX_AREA}" \
  --min_duration_sec "${MIN_DURATION_SEC}" \
  "${COMMON_ARGS[@]}"

RUN_OUTPUT_DIR="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${RUN_OUTPUT_DIR}")"
echo "[OVERFIT10-UNFREEZE][VERIFY] before checkpoint: ${RUN_OUTPUT_DIR}/checkpoint-before-training"
echo "[OVERFIT10-UNFREEZE][VERIFY] final checkpoint : ${RUN_OUTPUT_DIR}/checkpoint-final"
echo "[OVERFIT10-UNFREEZE][VERIFY] chain manifest   : ${RUN_OUTPUT_DIR}/training_chain_manifest.json"
echo "[OVERFIT10-UNFREEZE][VERIFY] pair report      : ${PAIR_REPORT_JSON}"
echo "[OVERFIT10-UNFREEZE][VERIFY] audit cmd:"
echo "python /home/liuzhirui/model/Wan2.2/verify_metaquery_chain.py \\"
echo "  --before_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-before-training \\"
echo "  --after_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-final \\"
echo "  --output_json ${RUN_OUTPUT_DIR}/chain_audit_report.json --strict"

