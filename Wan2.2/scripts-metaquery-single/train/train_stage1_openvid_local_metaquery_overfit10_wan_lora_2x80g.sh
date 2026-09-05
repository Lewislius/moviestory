#!/bin/bash
set -euo pipefail

# Standalone 2x80G A100 launcher for Wan DiT LoRA overfit10 training.
#
# Topology:
#   - single process
#   - 2 GPUs per process
#   - auto_device_map => dit_device=0, encoder_device=1
#
# Notes:
#   - This is intentionally NOT a wrapper around another .sh file.
#   - Wan base weights stay frozen; only Wan LoRA params are trainable on the Wan side.
#   - MQ connector and MQ input embeddings stay trainable for MetaQuery conditioning.
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export http_proxy="${http_proxy:-10.130.130.6:56830}"
export https_proxy="${https_proxy:-10.130.130.6:56830}"



if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1"
fi

IFS=',' read -r -a _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#_GPU_ARR[@]} < 2 )); then
  echo "[ERROR] This launcher requires 2 visible GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  exit 2
fi

export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export WAN_DIST_PROCESS_DEVICE="${WAN_DIST_PROCESS_DEVICE:-encoder}"
export WANDB_ENABLED="${WANDB_ENABLED:-1}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_PROJECT="${WANDB_PROJECT:-wan-metaquery-overfit10}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"

TASK="${TASK:-ti2v}"  # ti2v | i2v | animate
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-900}"
SAVE_STEPS="${SAVE_STEPS:-600}"
LOG_STEPS="${LOG_STEPS:-1}"
LOG_EVERY_STEP="${LOG_EVERY_STEP:-1}"

# 2x80G topology: single-process dual-GPU split, not multi-process DDP.
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
GPUS_PER_PROCESS="${GPUS_PER_PROCESS:-2}"
if [[ "${NPROC_PER_NODE}" != "1" ]]; then
  echo "[WARN] 该 2x80g LoRA 脚本固定使用单进程双卡；自动将 NPROC_PER_NODE: ${NPROC_PER_NODE} -> 1"
  NPROC_PER_NODE="1"
fi
if [[ "${GPUS_PER_PROCESS}" != "2" ]]; then
  echo "[WARN] 该 2x80g LoRA 脚本固定使用每进程2卡；自动将 GPUS_PER_PROCESS: ${GPUS_PER_PROCESS} -> 2"
  GPUS_PER_PROCESS="2"
fi

SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-150}"
COOLDOWN_STEPS="${COOLDOWN_STEPS:-${WARMUP_STEPS}}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-warmup_hold_cooldown}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.01}"
ENABLE_LOSS_EARLY_STOP="${ENABLE_LOSS_EARLY_STOP:-1}"
LOSS_EARLY_STOP_MIN_STEP="${LOSS_EARLY_STOP_MIN_STEP:-860}"
LOSS_EARLY_STOP_THRESHOLD="${LOSS_EARLY_STOP_THRESHOLD:-0.25}"

OPENVID_VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video}"
OPENVID_CSV_PATH="${OPENVID_CSV_PATH:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/data/train/OpenVid-1M.csv}"

QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
WAN_TI2V_CKPT="${WAN_TI2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
WAN_I2V_CKPT="${WAN_I2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B}"
WAN_ANIMATE_CKPT="${WAN_ANIMATE_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-Animate-14B}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_wan_lora_2x80g}"
OVERFIT_ROOT="${OVERFIT_ROOT:-/home/liuzhirui/model/Wan2.2/tmp/metaquery_overfit10_wan_lora_2x80g}"
OVERFIT_ROOT="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${OVERFIT_ROOT}")"
mkdir -p "${OVERFIT_ROOT}"

VIDEO_LIST_FILE="${OVERFIT_ROOT}/video_list_overfit10.txt"
MINI_VIDEO_ROOT="${OVERFIT_ROOT}/videos"
MINI_CSV_PATH="${OVERFIT_ROOT}/openvid_overfit10.csv"
PAIR_REPORT_JSON="${OVERFIT_ROOT}/pair_report.json"

cat > "${VIDEO_LIST_FILE}" << 'EOF'
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv___f2KtcXAxI_1.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv___lRwnjxeCg_3.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__0yBbZJZqG8_1.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__1osQSmJ2-s_4.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__1s0YQ8dL04_0.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__4DRXSdPuCo_2.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__4DRXSdPuCo_4.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__4juqo20ABE_0.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__5ukjsqqLg4_12.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__6RI-8Ia4do_0.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__IBCLj2lq_U_0.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__GxarTw5BNM_1.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__DQAqotwgd4_18.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__d68kyNY0jI_0.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__BeHUjskbZo_1.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__BEb_ZjotP0_19_0.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__a-pB_5eRt0_6.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__5ukjsqqLg4_8.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__4DRXSdPuCo_3.mp4
/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__4DRXSdPuCo_0.mp4
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
        print(f"[OVERFIT10-LORA-2x80G][DROP] video={src} reason=video_not_found")
        continue

    try:
        rel_key = norm_key(src.relative_to(openvid_root))
    except Exception:
        rel_key = norm_key(src.name)

    cap = path_to_caption.get(rel_key)
    if cap is None:
        cap = name_to_caption.get(norm_key(src.name))
    if not cap:
        dropped.append({"video": str(src), "reason": "caption_not_found"})
        print(f"[OVERFIT10-LORA-2x80G][DROP] video={src} reason=caption_not_found")
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

    kept.append({
        "video": target_name,
        "caption": cap,
        "src": str(src),
        "link_type": "symlink_or_hardlink" if linked else "copy",
    })
    print(f"[OVERFIT10-LORA-2x80G][KEEP] video={target_name} src={src} caption={cap}")

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
print(f"[OVERFIT10-LORA-2x80G] mini_csv={mini_csv_path.resolve()} kept={len(kept)} dropped={len(dropped)}")
print(f"[OVERFIT10-LORA-2x80G] pair_report={pair_report_json.resolve()}")
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
        print(sum(1 for _ in csv.DictReader(f)))
PY
)"

if [[ "${KEPT_COUNT}" -le 0 ]]; then
  echo "[ERROR] kept_count=${KEPT_COUNT}, nothing to train."
  exit 2
fi

echo "[OVERFIT10-LORA-2x80G] final_kept_count=${KEPT_COUNT}"
echo "[OVERFIT10-LORA-2x80G] mini_video_root=${MINI_VIDEO_ROOT}"
echo "[OVERFIT10-LORA-2x80G] mini_csv=${MINI_CSV_PATH}"

export WAN_LOCAL_OVERFIT_DEBUG=1
export WAN_LOCAL_MISSING_CAPTION_PRINT_MAX=1000
export WAN_DATA_PRECLEAN=1
HF_NO_SUBSET_CACHE_FLAG="--hf_no_subset_cache"

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-overfit10-lora-2x80g-${TASK}-steps${NUM_TRAIN_STEPS}-$(date +%Y%m%d-%H%M%S)}"
WANDB_TAGS="${WANDB_TAGS:-overfit,debug,overfit10,wan-lora,2x80g}"
WANDB_MODE="${WANDB_MODE:-offline}"

LOG_CUDA_MEMORY="${LOG_CUDA_MEMORY:-1}"
WANDB_LOG_EVERY_STEP="${WANDB_LOG_EVERY_STEP:-1}"
WANDB_LOG_CHECKPOINT="${WANDB_LOG_CHECKPOINT:-1}"

NUM_METAQUERIES="${NUM_METAQUERIES:-256}"
NULL_IMAGE_PROB="${NULL_IMAGE_PROB:-0.1}"
NULL_CAPTION_PROB="${NULL_CAPTION_PROB:-0.1}"

# MQ side: these two are the components that should remain trainable.
TRAIN_MQ_INPUT_EMBEDDINGS="${TRAIN_MQ_INPUT_EMBEDDINGS:-1}"
REQUIRE_MQ_EMBEDDING_TRAIN="${REQUIRE_MQ_EMBEDDING_TRAIN:-1}"

# Auxiliary alignment / preserve objectives stay enabled by default.
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

# Wan side: keep base frozen and train LoRA only.
WAN_TRAIN_MODE="${WAN_TRAIN_MODE:-frozen}"
WAN_AUTO_FULL_MEM_GB="${WAN_AUTO_FULL_MEM_GB:-120}"
WAN_LR_RATIO="${WAN_LR_RATIO:-1.0}"
WAN_COND_NAME_PATTERN="${WAN_COND_NAME_PATTERN:-}"
ENABLE_WAN_LORA="${ENABLE_WAN_LORA:-1}"
WAN_LORA_RANK="${WAN_LORA_RANK:-16}"
WAN_LORA_ALPHA="${WAN_LORA_ALPHA:-16}"
WAN_LORA_DROPOUT="${WAN_LORA_DROPOUT:-0.0}"
WAN_LORA_TARGETS="${WAN_LORA_TARGETS:-self_attn,cross_attn,ffn}"

if [[ "${ENABLE_WAN_LORA}" != "1" ]]; then
  echo "[WARN] 该脚本默认用于 Wan LoRA 训练；自动将 ENABLE_WAN_LORA: ${ENABLE_WAN_LORA} -> 1"
  ENABLE_WAN_LORA="1"
fi
if [[ "${WAN_TRAIN_MODE}" != "frozen" ]]; then
  echo "[WARN] 该脚本默认冻结 base Wan 权重；自动将 WAN_TRAIN_MODE: ${WAN_TRAIN_MODE} -> frozen"
  WAN_TRAIN_MODE="frozen"
fi

TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-49}"
TI2V_MAX_AREA="${TI2V_MAX_AREA:-262144}"
I2V_FRAME_NUM="${I2V_FRAME_NUM:-49}"
I2V_MAX_AREA="${I2V_MAX_AREA:-262144}"
ANIMATE_FRAME_NUM="${ANIMATE_FRAME_NUM:-49}"
ANIMATE_MAX_AREA="${ANIMATE_MAX_AREA:-262144}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-0.3}"
DIT_CONDITION_MODE="${DIT_CONDITION_MODE:-mq_only}"
TRAIN_REF_ANCHOR_MODE="${TRAIN_REF_ANCHOR_MODE:-none}"

METRICS_JSONL_PATH="${METRICS_JSONL_PATH:-${OUTPUT_ROOT}/logs/${TASK}_overfit10_lora_2x80g_steps${NUM_TRAIN_STEPS}_$(date +%Y%m%d-%H%M%S).jsonl}"
METRICS_JSONL_PATH="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${METRICS_JSONL_PATH}")"

COMMON_ARGS=(
  --distributed
  --auto_device_map
  --gpus_per_process "${GPUS_PER_PROCESS}"
  --hf_stage stage1
  --hf_no_streaming
  "${HF_NO_SUBSET_CACHE_FLAG}"
  --t5_cpu
  --mq_gradient_checkpointing
  --aggressive_empty_cache
  --local_openvid_video_root "${MINI_VIDEO_ROOT}"
  --local_openvid_csv_path "${MINI_CSV_PATH}"
  --local_openvid_limit "${KEPT_COUNT}"
  --local_openvid_hd_video_root ""
  --local_openvid_hd_csv_path ""
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
  COMMON_ARGS+=(--enable_wan_lora)
  COMMON_ARGS+=(--wan_lora_rank "${WAN_LORA_RANK}")
  COMMON_ARGS+=(--wan_lora_alpha "${WAN_LORA_ALPHA}")
  COMMON_ARGS+=(--wan_lora_dropout "${WAN_LORA_DROPOUT}")
  COMMON_ARGS+=(--wan_lora_targets "${WAN_LORA_TARGETS}")
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
  echo "[FATAL] REQUIRE_MQ_EMBEDDING_TRAIN=1 但 TRAIN_MQ_INPUT_EMBEDDINGS=${TRAIN_MQ_INPUT_EMBEDDINGS}"
  echo "[FATAL] 这会导致 MetaQuery learnable token embedding 不参与训练。"
  echo "[FATAL] 若确实要做冻结对照实验，请显式设置 REQUIRE_MQ_EMBEDDING_TRAIN=0。"
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

echo "[LAUNCH][OVERFIT10-LORA-2x80G] TASK=${TASK} steps=${NUM_TRAIN_STEPS} kept=${KEPT_COUNT}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] topology=single-process dual-gpu split (dit->gpu0, encoder->gpu1)"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] mini_root=${MINI_VIDEO_ROOT}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] mini_csv=${MINI_CSV_PATH}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] metrics_jsonl=${METRICS_JSONL_PATH}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] warmup_steps=${WARMUP_STEPS} cooldown_steps=${COOLDOWN_STEPS} lr_scheduler_type=${LR_SCHEDULER_TYPE} lr_min_ratio=${LR_MIN_RATIO}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] early_stop enabled=${ENABLE_LOSS_EARLY_STOP} min_step=${LOSS_EARLY_STOP_MIN_STEP} threshold=${LOSS_EARLY_STOP_THRESHOLD}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] null_image_prob=${NULL_IMAGE_PROB} null_caption_prob=${NULL_CAPTION_PROB}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] t5_align=enable:${ENABLE_T5_ALIGNMENT} mode=${T5_ALIGN_MODE} anchor=${T5_ALIGN_ANCHOR_TOKENS} l2=${LAMBDA_T5_ALIGN_L2} cos=${LAMBDA_T5_ALIGN_COS} stats=${LAMBDA_T5_ALIGN_STATS} ot_eps=${T5_ALIGN_OT_EPSILON} ot_iters=${T5_ALIGN_OT_ITERS}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] mq_image_preserve=enable:${ENABLE_MQ_IMAGE_PRESERVE} lambda=${LAMBDA_MQ_IMAGE_PRESERVE} margin=${MQ_IMAGE_PRESERVE_MARGIN}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] wan_func_distill=enable:${ENABLE_WAN_FUNC_DISTILL} lambda=${LAMBDA_WAN_FUNC_DISTILL} teacher=${WAN_FUNC_TEACHER_MODE}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] wan_train_mode=${WAN_TRAIN_MODE} auto_full_mem_gb=${WAN_AUTO_FULL_MEM_GB} wan_lr_ratio=${WAN_LR_RATIO}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] wan_lora enabled=${ENABLE_WAN_LORA} rank=${WAN_LORA_RANK} alpha=${WAN_LORA_ALPHA} dropout=${WAN_LORA_DROPOUT} targets=${WAN_LORA_TARGETS}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] batch_size=${BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} lr=${LEARNING_RATE}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] train_mq_input_embeddings=${TRAIN_MQ_INPUT_EMBEDDINGS} require_mq_embedding_train=${REQUIRE_MQ_EMBEDDING_TRAIN}"
echo "[LAUNCH][OVERFIT10-LORA-2x80G] dit_condition_mode=${DIT_CONDITION_MODE} train_ref_anchor_mode=${TRAIN_REF_ANCHOR_MODE}"

case "${TASK}" in
  ti2v)
    RUN_OUTPUT_DIR="${OUTPUT_ROOT}/ti2v_overfit10_lora_2x80g_steps${NUM_TRAIN_STEPS}_nummq${NUM_METAQUERIES}_rank${WAN_LORA_RANK}_alpha${WAN_LORA_ALPHA}"
    torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_wan_new_lora.py \
      --wan_checkpoint_dir "${WAN_TI2V_CKPT}" \
      --output_dir "${RUN_OUTPUT_DIR}" \
      --frame_num "${TI2V_FRAME_NUM}" \
      --max_area "${TI2V_MAX_AREA}" \
      --min_duration_sec "${MIN_DURATION_SEC}" \
      --train_ref_anchor_mode "${TRAIN_REF_ANCHOR_MODE}" \
      "${COMMON_ARGS[@]}"
    ;;
  i2v)
    echo "[ERROR] overfit10_wan_lora_2x80g 当前仅支持 ti2v"
    exit 3
    ;;
  animate)
    echo "[ERROR] overfit10_wan_lora_2x80g 当前仅支持 ti2v"
    exit 3
    ;;
  *)
    echo "[ERROR] TASK must be one of: ti2v | i2v | animate"
    exit 3
    ;;
esac

RUN_OUTPUT_DIR="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${RUN_OUTPUT_DIR}")"
echo "[OVERFIT10-LORA-2x80G][VERIFY] before checkpoint: ${RUN_OUTPUT_DIR}/checkpoint-before-training"
echo "[OVERFIT10-LORA-2x80G][VERIFY] final checkpoint : ${RUN_OUTPUT_DIR}/checkpoint-final"
echo "[OVERFIT10-LORA-2x80G][VERIFY] chain manifest   : ${RUN_OUTPUT_DIR}/training_chain_manifest.json"
echo "[OVERFIT10-LORA-2x80G][VERIFY] pair report      : ${PAIR_REPORT_JSON}"
echo "[OVERFIT10-LORA-2x80G][VERIFY] audit cmd:"
echo "python /home/liuzhirui/model/Wan2.2/verify_metaquery_chain.py \\"
echo "  --before_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-before-training \\"
echo "  --after_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-final \\"
echo "  --output_json ${RUN_OUTPUT_DIR}/chain_audit_report.json --strict"
