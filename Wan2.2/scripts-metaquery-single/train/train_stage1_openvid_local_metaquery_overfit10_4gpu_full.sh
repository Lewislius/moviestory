#!/bin/bash

# 下面这个是DiT部分/全部训练，同时不添加其他额外损失函数的情况：
set -euo pipefail

# Usage:
#   bash train_stage1_openvid_local_metaquery_overfit10.sh
#
# Goal:
#   Overfit 10 fixed OpenVid videos for pipeline sanity-check (100 steps by default).
#   The script will:
#   1) Build a tiny local dataset (video symlink dir + mini csv with caption)
#   2) Print KEEP/DROP details for those 10 videos before training
#   3) Enable dataset debug print during preclean (KEEP/DROP + reason + caption)
#   4) Run train_metaquery_*_new.py with tiny dataset

source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

export http_proxy="${http_proxy:-10.130.130.6:56830}"
export https_proxy="${https_proxy:-10.130.130.6:56830}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=1
export WANDB_ENABLED=1
export WANDB_API_KEY="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"
export WANDB_PROJECT="wan-metaquery-overfit10"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export WAN_DIST_TIMEOUT_SEC="${WAN_DIST_TIMEOUT_SEC:-3600}"  # 默认1小时，覆盖分布式collective超时

TASK="${TASK:-ti2v}"  # ti2v | i2v | animate
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-861}"
SAVE_STEPS="${SAVE_STEPS:-600}"
LOG_STEPS="${LOG_STEPS:-1}"
LOG_EVERY_STEP="${LOG_EVERY_STEP:-1}"
# 为了让 DiT FSDP 在 4 卡上尽可能均分，默认使用 4 进程 x 每进程 1 卡。
# 这样 world_size=4，DiT 参数分片会跨 4 个 rank。
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GPUS_PER_PROCESS="${GPUS_PER_PROCESS:-1}"
FORCE_EVEN_FSDP_TOPOLOGY="${FORCE_EVEN_FSDP_TOPOLOGY:-1}"  # 1=强制回到 4x1 拓扑
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-150}"
COOLDOWN_STEPS="${COOLDOWN_STEPS:-${WARMUP_STEPS}}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-warmup_hold_cooldown}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.01}"
ENABLE_LOSS_EARLY_STOP="${ENABLE_LOSS_EARLY_STOP:-1}" # 这个设置为1的时候，如果当前训练步数超过 LOSS_EARLY_STOP_MIN_STEP 且当前损失低于 LOSS_EARLY_STOP_THRESHOLD，则提前停止训练
LOSS_EARLY_STOP_MIN_STEP="${LOSS_EARLY_STOP_MIN_STEP:-810}"
LOSS_EARLY_STOP_THRESHOLD="${LOSS_EARLY_STOP_THRESHOLD:-0.25}"

OPENVID_VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video}"
OPENVID_CSV_PATH="${OPENVID_CSV_PATH:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/data/train/OpenVid-1M.csv}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1"
fi
IFS=',' read -r -a _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#_GPU_ARR[@]} < 2 )); then
  echo "[ERROR] This launcher requires 2 visible GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  exit 10
fi

QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
WAN_TI2V_CKPT="${WAN_TI2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
WAN_I2V_CKPT="${WAN_I2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B}"
WAN_ANIMATE_CKPT="${WAN_ANIMATE_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-Animate-14B}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_full_4gpu}"
OVERFIT_ROOT="${OVERFIT_ROOT:-/home/liuzhirui/model/Wan2.2/tmp/metaquery_overfit10}"
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

# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video_HD/__5lGFYcmwk_2_0to149.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video_HD/_1Skgh5WQEo_13_130to315.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video_HD/_2icvUottyg_4_30to189.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video_HD/_2VzuDsnin8_16_0to105.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video_HD/__4c1JCHvaQ_147_35to148.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video_HD/__zSJ-Ha5r4_15_0to102.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video_HD/_-Gmm3wKKXs_37_0to118.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video_HD/_-uC9nkcd0w_17_115to253.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video_HD/_0wFGE2BtOQ_3_0to130.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video_HD/_1BYTrCWy_k_56_92to218.mp4


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
        print(f"[OVERFIT10][DROP] video={src} reason=video_not_found")
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
        print(f"[OVERFIT10][DROP] video={src} reason=caption_not_found")
        continue

    target_name = src.name
    # avoid basename collision
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
    print(f"[OVERFIT10][KEEP] video={target_name} src={src} caption={cap}")

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
print(f"[OVERFIT10] mini_csv={mini_csv_path.resolve()} kept={len(kept)} dropped={len(dropped)}")
print(f"[OVERFIT10] pair_report={pair_report_json.resolve()}")
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

echo "[OVERFIT10] final_kept_count=${KEPT_COUNT}"
echo "[OVERFIT10] mini_video_root=${MINI_VIDEO_ROOT}"
echo "[OVERFIT10] mini_csv=${MINI_CSV_PATH}"

# Dataset debug print (KEEP/DROP during preclean with reason+caption)
export WAN_LOCAL_OVERFIT_DEBUG=1
export WAN_LOCAL_MISSING_CAPTION_PRINT_MAX=1000
export WAN_DATA_PRECLEAN=1

# disable subset cache so every run prints full KEEP/DROP details
HF_NO_SUBSET_CACHE_FLAG="--hf_no_subset_cache"

WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-wan-metaquery-overfit10}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-overfit10-${TASK}-steps${NUM_TRAIN_STEPS}-$(date +%Y%m%d-%H%M%S)}"
WANDB_TAGS="${WANDB_TAGS:-overfit,debug,overfit10}"
WANDB_MODE="${WANDB_MODE:-offline}"

LOG_CUDA_MEMORY="${LOG_CUDA_MEMORY:-1}"
WANDB_LOG_EVERY_STEP="${WANDB_LOG_EVERY_STEP:-1}"
WANDB_LOG_CHECKPOINT="${WANDB_LOG_CHECKPOINT:-1}"

NUM_METAQUERIES="${NUM_METAQUERIES:-256}"
NULL_IMAGE_PROB="${NULL_IMAGE_PROB:-0.1}"
NULL_CAPTION_PROB="${NULL_CAPTION_PROB:-0.1}"
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
TRAIN_MQ_INPUT_EMBEDDINGS="${TRAIN_MQ_INPUT_EMBEDDINGS:-1}" # 1=默认训练MQ新增token embedding+connector；0=仅训connector
REQUIRE_MQ_EMBEDDING_TRAIN="${REQUIRE_MQ_EMBEDDING_TRAIN:-1}" # 1=强制要求训练MQ token embedding，防止误配
WAN_TRAIN_MODE="${WAN_TRAIN_MODE:-cond_only}"                       # auto/full/cond_only/frozen
WAN_AUTO_FULL_MEM_GB="${WAN_AUTO_FULL_MEM_GB:-120}"           # auto 决策阈值
WAN_LR_RATIO="${WAN_LR_RATIO:-0.4}"                           # Wan 参数 lr 倍率
WAN_COND_NAME_PATTERN="${WAN_COND_NAME_PATTERN:-}"            # cond_only 关键字覆盖（逗号分隔）
DIT_FSDP="${DIT_FSDP:-1}"                                     # 4卡full默认启用，保证 Wan 参数跨rank同步
T5_FSDP="${T5_FSDP:-0}"
USE_SP="${USE_SP:-0}"
T5_CPU="${T5_CPU:-1}"                                         # 仅在 T5_FSDP=0 时有效

if [[ "${FORCE_EVEN_FSDP_TOPOLOGY}" == "1" ]]; then
  if [[ "${NPROC_PER_NODE}" != "2" || "${GPUS_PER_PROCESS}" != "1" ]]; then
    echo "[WARN] FORCE_EVEN_FSDP_TOPOLOGY=1，自动将拓扑重置为 NPROC_PER_NODE=2, GPUS_PER_PROCESS=1"
    NPROC_PER_NODE="2"
    GPUS_PER_PROCESS="1"
  fi
fi

# cond_only 依赖参数名关键字筛选；在 DiT FSDP/SP 扁平化后参数名可能丢失语义，导致选参为空并触发审计失败。
# 这里自动切回单进程非 FSDP，以保持与 overfit10.sh 一致的 cond_only 行为。
if [[ "${TASK}" == "ti2v" && "${WAN_TRAIN_MODE}" == "cond_only" ]]; then
  echo "[INFO] WAN_TRAIN_MODE=cond_only: 自动切换到单进程非FSDP路径（NPROC_PER_NODE=1, GPUS_PER_PROCESS=1, DIT_FSDP=0, USE_SP=0）"
  NPROC_PER_NODE="1"
  GPUS_PER_PROCESS="1"
  DIT_FSDP="0"
  T5_FSDP="0"
  USE_SP="0"
  T5_CPU="1"
fi

if [[ "${T5_FSDP}" == "1" && "${T5_CPU}" == "1" ]]; then
  echo "[WARN] T5_FSDP=1 与 T5_CPU=1 互斥，自动将 T5_CPU: 1 -> 0"
  T5_CPU="0"
fi

if [[ "${WAN_TRAIN_MODE}" == "full" && "${DIT_FSDP}" != "1" ]]; then
  echo "[WARN] 当前脚本目标是 2 卡 full + 显存均分，建议保持 DIT_FSDP=1；自动设置 DIT_FSDP=1"
  DIT_FSDP="1"
fi

if [[ "${DIT_FSDP}" == "1" || "${USE_SP}" == "1" ]]; then
  export WAN_DIST_PROCESS_DEVICE="${WAN_DIST_PROCESS_DEVICE:-dit}"
else
  export WAN_DIST_PROCESS_DEVICE="${WAN_DIST_PROCESS_DEVICE:-encoder}"
fi

if [[ "${TASK}" == "ti2v" && "${WAN_TRAIN_MODE}" != "frozen" && "${NPROC_PER_NODE}" != "1" && "${DIT_FSDP}" != "1" && "${USE_SP}" != "1" ]]; then
  echo "[WARN] 多进程 full/cond_only 训练建议启用 DIT_FSDP 或 USE_SP，否则 Wan 参数无法可靠同步。"
  echo "[WARN] 自动将 NPROC_PER_NODE: ${NPROC_PER_NODE} -> 1（如需4卡并行请设置 DIT_FSDP=1）"
  NPROC_PER_NODE="1"
  GPUS_PER_PROCESS="1"
fi
if [[ "${TASK}" != "ti2v" && "${LR_SCHEDULER_TYPE}" == "warmup_hold_cooldown" ]]; then
  echo "[WARN] 当前 warmup_hold_cooldown 仅在 ti2v 主训练脚本中实现；${TASK} 任务自动回退为 constant_with_warmup"
  LR_SCHEDULER_TYPE="constant_with_warmup"
fi

# TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-13}"
TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-49}"
I2V_FRAME_NUM="${I2V_FRAME_NUM:-49}"
ANIMATE_FRAME_NUM="${ANIMATE_FRAME_NUM:-49}"
# TI2V_MAX_AREA="${TI2V_MAX_AREA:-102400}"
TI2V_MAX_AREA="${TI2V_MAX_AREA:-262144}"
I2V_MAX_AREA="${I2V_MAX_AREA:-262144}"
ANIMATE_MAX_AREA="${ANIMATE_MAX_AREA:-262144}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-0.3}"
DIT_CONDITION_MODE="${DIT_CONDITION_MODE:-mq_only}"
TRAIN_REF_ANCHOR_MODE="${TRAIN_REF_ANCHOR_MODE:-none}"  # 固定为 none: 不启用 animate-like/hard-lock 派生逻辑

METRICS_JSONL_PATH="${METRICS_JSONL_PATH:-${OUTPUT_ROOT}/logs/${TASK}_overfit10_steps${NUM_TRAIN_STEPS}_$(date +%Y%m%d-%H%M%S).jsonl}"
METRICS_JSONL_PATH="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${METRICS_JSONL_PATH}")"

COMMON_ARGS=(
  --distributed
  --auto_device_map
  --gpus_per_process "${GPUS_PER_PROCESS}"
  --hf_stage stage1
  --hf_no_streaming
  "${HF_NO_SUBSET_CACHE_FLAG}"
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

if [[ "${T5_CPU}" == "1" ]]; then
  COMMON_ARGS+=(--t5_cpu)
fi
if [[ "${DIT_FSDP}" == "1" ]]; then
  COMMON_ARGS+=(--dit_fsdp)
fi
if [[ "${T5_FSDP}" == "1" ]]; then
  COMMON_ARGS+=(--t5_fsdp)
fi
if [[ "${USE_SP}" == "1" ]]; then
  COMMON_ARGS+=(--use_sp)
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
  echo "[FATAL] REQUIRE_MQ_EMBEDDING_TRAIN=1 但 TRAIN_MQ_INPUT_EMBEDDINGS=${TRAIN_MQ_INPUT_EMBEDDINGS}"
  echo "[FATAL] 这会导致 256 个 MetaQuery learnable token embedding 不参与训练。"
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

echo "[LAUNCH][OVERFIT10] TASK=${TASK} steps=${NUM_TRAIN_STEPS} kept=${KEPT_COUNT}"
echo "[LAUNCH][OVERFIT10] mini_root=${MINI_VIDEO_ROOT}"
echo "[LAUNCH][OVERFIT10] mini_csv=${MINI_CSV_PATH}"
echo "[LAUNCH][OVERFIT10] metrics_jsonl=${METRICS_JSONL_PATH}"
echo "[LAUNCH][OVERFIT10] warmup_steps=${WARMUP_STEPS} cooldown_steps=${COOLDOWN_STEPS} lr_scheduler_type=${LR_SCHEDULER_TYPE} lr_min_ratio=${LR_MIN_RATIO}"
echo "[LAUNCH][OVERFIT10] early_stop enabled=${ENABLE_LOSS_EARLY_STOP} min_step=${LOSS_EARLY_STOP_MIN_STEP} threshold=${LOSS_EARLY_STOP_THRESHOLD}"
echo "[LAUNCH][OVERFIT10] null_image_prob=${NULL_IMAGE_PROB} null_caption_prob=${NULL_CAPTION_PROB}"
echo "[LAUNCH][OVERFIT10] t5_align=enable:${ENABLE_T5_ALIGNMENT} mode=${T5_ALIGN_MODE} anchor=${T5_ALIGN_ANCHOR_TOKENS} l2=${LAMBDA_T5_ALIGN_L2} cos=${LAMBDA_T5_ALIGN_COS} stats=${LAMBDA_T5_ALIGN_STATS} ot_eps=${T5_ALIGN_OT_EPSILON} ot_iters=${T5_ALIGN_OT_ITERS}"
echo "[LAUNCH][OVERFIT10] mq_image_preserve=enable:${ENABLE_MQ_IMAGE_PRESERVE} lambda=${LAMBDA_MQ_IMAGE_PRESERVE} margin=${MQ_IMAGE_PRESERVE_MARGIN}"
echo "[LAUNCH][OVERFIT10] wan_func_distill=enable:${ENABLE_WAN_FUNC_DISTILL} lambda=${LAMBDA_WAN_FUNC_DISTILL} teacher=${WAN_FUNC_TEACHER_MODE}"
echo "[LAUNCH][OVERFIT10] wan_train_mode=${WAN_TRAIN_MODE} auto_full_mem_gb=${WAN_AUTO_FULL_MEM_GB} wan_lr_ratio=${WAN_LR_RATIO} wan_cond_name_pattern=${WAN_COND_NAME_PATTERN:-<default>}"
echo "[LAUNCH][OVERFIT10] topology nproc_per_node=${NPROC_PER_NODE} gpus_per_process=${GPUS_PER_PROCESS} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[LAUNCH][OVERFIT10] fsdp dit=${DIT_FSDP} t5=${T5_FSDP} use_sp=${USE_SP} t5_cpu=${T5_CPU} process_device_pref=${WAN_DIST_PROCESS_DEVICE}"
echo "[LAUNCH][OVERFIT10] target=even_vram_sharding force_even_topology=${FORCE_EVEN_FSDP_TOPOLOGY}"
echo "[LAUNCH][OVERFIT10] batch_size=${BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} lr=${LEARNING_RATE} train_mq_input_embeddings=${TRAIN_MQ_INPUT_EMBEDDINGS}"
echo "[LAUNCH][OVERFIT10] dit_condition_mode=${DIT_CONDITION_MODE} train_ref_anchor_mode=${TRAIN_REF_ANCHOR_MODE}"
echo "[LAUNCH][OVERFIT10] dist_timeout_sec=${WAN_DIST_TIMEOUT_SEC}"

case "${TASK}" in
  ti2v)
    RUN_OUTPUT_DIR="${OUTPUT_ROOT}/ti2v_overfit30_steps${NUM_TRAIN_STEPS}_nummq${NUM_METAQUERIES}_nullimg${NULL_IMAGE_PROB}_nullcap${NULL_CAPTION_PROB}"
    torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_wan_new.py \
      --wan_checkpoint_dir "${WAN_TI2V_CKPT}" \
      --output_dir "${RUN_OUTPUT_DIR}" \
      --frame_num "${TI2V_FRAME_NUM}" \
      --max_area "${TI2V_MAX_AREA}" \
      --min_duration_sec "${MIN_DURATION_SEC}" \
      --train_ref_anchor_mode "${TRAIN_REF_ANCHOR_MODE}" \
      "${COMMON_ARGS[@]}"
    ;;
  i2v)
    RUN_OUTPUT_DIR="${OUTPUT_ROOT}/i2v_overfit30_steps${NUM_TRAIN_STEPS}_nummq${NUM_METAQUERIES}_nullimg${NULL_IMAGE_PROB}_nullcap${NULL_CAPTION_PROB}"
    torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_i2v_new.py \
      --wan_checkpoint_dir "${WAN_I2V_CKPT}" \
      --output_dir "${RUN_OUTPUT_DIR}" \
      --frame_num "${I2V_FRAME_NUM}" \
      --max_area "${I2V_MAX_AREA}" \
      --min_duration_sec "${MIN_DURATION_SEC}" \
      "${COMMON_ARGS[@]}"
    ;;
  animate)
    RUN_OUTPUT_DIR="${OUTPUT_ROOT}/animate_overfit30_steps${NUM_TRAIN_STEPS}"
    torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_animate_new.py \
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
echo "[OVERFIT10][VERIFY] before checkpoint: ${RUN_OUTPUT_DIR}/checkpoint-before-training"
echo "[OVERFIT10][VERIFY] final checkpoint : ${RUN_OUTPUT_DIR}/checkpoint-final"
echo "[OVERFIT10][VERIFY] chain manifest   : ${RUN_OUTPUT_DIR}/training_chain_manifest.json"
echo "[OVERFIT10][VERIFY] pair report      : ${PAIR_REPORT_JSON}"
echo "[OVERFIT10][VERIFY] audit cmd:"
echo "python /home/liuzhirui/model/Wan2.2/verify_metaquery_chain.py \\"
echo "  --before_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-before-training \\"
echo "  --after_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-final \\"
echo "  --output_json ${RUN_OUTPUT_DIR}/chain_audit_report.json --strict"
