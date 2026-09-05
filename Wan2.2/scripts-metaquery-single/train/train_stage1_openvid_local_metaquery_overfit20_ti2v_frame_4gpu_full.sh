#!/bin/bash
set -euo pipefail

# 目标:
# - 独立脚本（不调用其他 sh）
# - 2 GPU 并行 + torchrun + DiT FSDP
# - 参数组织与默认值尽量对齐 train_stage1_openvid_local_metaquery_overfit10_4gpu_full.sh
# - 适配 TI2V 首帧条件训练（WAN 侧也注入首帧条件）

source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export http_proxy="${http_proxy:-10.130.130.6:56830}"
export https_proxy="${https_proxy:-10.130.130.6:56830}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"
export WAN_DIST_TIMEOUT_SEC="${WAN_DIST_TIMEOUT_SEC:-3600}"

TASK="${TASK:-ti2v}"  # 固定面向 ti2v
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-800}"
SAVE_STEPS="${SAVE_STEPS:-600}"
LOG_STEPS="${LOG_STEPS:-1}"
LOG_EVERY_STEP="${LOG_EVERY_STEP:-1}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
COOLDOWN_STEPS="${COOLDOWN_STEPS:-${WARMUP_STEPS}}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-warmup_hold_cooldown}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.01}"
ENABLE_LOSS_EARLY_STOP="${ENABLE_LOSS_EARLY_STOP:-1}"
LOSS_EARLY_STOP_MIN_STEP="${LOSS_EARLY_STOP_MIN_STEP:-760}"
LOSS_EARLY_STOP_THRESHOLD="${LOSS_EARLY_STOP_THRESHOLD:-0.25}"

# 2 GPU + 2 进程 x 1 卡
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GPUS_PER_PROCESS="${GPUS_PER_PROCESS:-1}"
FORCE_EVEN_FSDP_TOPOLOGY="${FORCE_EVEN_FSDP_TOPOLOGY:-1}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1"
fi
IFS=',' read -r -a _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#_GPU_ARR[@]} < 2 )); then
  echo "[ERROR] This launcher requires 2 visible GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  exit 10
fi

if [[ "${FORCE_EVEN_FSDP_TOPOLOGY}" == "1" ]]; then
  if [[ "${NPROC_PER_NODE}" != "2" || "${GPUS_PER_PROCESS}" != "1" ]]; then
    echo "[WARN] FORCE_EVEN_FSDP_TOPOLOGY=1，自动重置为 NPROC_PER_NODE=2, GPUS_PER_PROCESS=1"
    NPROC_PER_NODE="2"
    GPUS_PER_PROCESS="1"
  fi
fi

OPENVID_VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video}"
OPENVID_CSV_PATH="${OPENVID_CSV_PATH:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/data/train/OpenVid-1M.csv}"
QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
WAN_TI2V_CKPT="${WAN_TI2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit20_ti2v_frame_2gpu_fsdp}"
OVERFIT_ROOT="${OVERFIT_ROOT:-/home/liuzhirui/model/Wan2.2/tmp/metaquery_overfit20_ti2v_frame_2gpu_fsdp}"
OVERFIT_ROOT="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${OVERFIT_ROOT}")"
mkdir -p "${OVERFIT_ROOT}"

VIDEO_LIST_FILE="${VIDEO_LIST_FILE:-${OVERFIT_ROOT}/video_list_overfit20.txt}"
MINI_VIDEO_ROOT="${OVERFIT_ROOT}/videos"
MINI_CSV_PATH="${OVERFIT_ROOT}/openvid_overfit20.csv"
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
"${PYTHON_BIN}" - << 'PY'
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
        print(f"[OVERFIT20-2GPU][DROP] video={src} reason=video_not_found")
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
        print(f"[OVERFIT20-2GPU][DROP] video={src} reason=caption_not_found")
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

    kept.append(
        {
            "video": target_name,
            "caption": cap,
            "src": str(src),
            "link_type": "symlink_or_hardlink" if linked else "copy",
        }
    )
    print(f"[OVERFIT20-2GPU][KEEP] video={target_name} src={src}")

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
print(f"[OVERFIT20-2GPU] mini_csv={mini_csv_path.resolve()} kept={len(kept)} dropped={len(dropped)}")
print(f"[OVERFIT20-2GPU] pair_report={pair_report_json.resolve()}")
PY

if [[ ! -f "${MINI_CSV_PATH}" ]]; then
  echo "[ERROR] mini csv not generated: ${MINI_CSV_PATH}"
  exit 1
fi

KEPT_COUNT="$("${PYTHON_BIN}" - << 'PY'
import csv
import os
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

echo "[OVERFIT20-2GPU] final_kept_count=${KEPT_COUNT}"
echo "[OVERFIT20-2GPU] mini_video_root=${MINI_VIDEO_ROOT}"
echo "[OVERFIT20-2GPU] mini_csv=${MINI_CSV_PATH}"

export WAN_LOCAL_OVERFIT_DEBUG=1
export WAN_LOCAL_MISSING_CAPTION_PRINT_MAX=1000
export WAN_DATA_PRECLEAN=1
HF_NO_SUBSET_CACHE_FLAG="--hf_no_subset_cache"

WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-wan-metaquery-overfit20-ti2v-frame-2gpu-fsdp}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-overfit20-ti2v-frame-2gpu-fsdp-steps${NUM_TRAIN_STEPS}-$(date +%Y%m%d-%H%M%S)}"
WANDB_TAGS="${WANDB_TAGS:-overfit20,ti2v,firstframe,2gpu,fsdp}"
WANDB_MODE="${WANDB_MODE:-offline}"
LOG_CUDA_MEMORY="${LOG_CUDA_MEMORY:-1}"
WANDB_LOG_EVERY_STEP="${WANDB_LOG_EVERY_STEP:-1}"
WANDB_LOG_CHECKPOINT="${WANDB_LOG_CHECKPOINT:-1}"

# 与 overfit10_4gpu_full 对齐的默认参数
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
TRAIN_MQ_INPUT_EMBEDDINGS="${TRAIN_MQ_INPUT_EMBEDDINGS:-1}"
REQUIRE_MQ_EMBEDDING_TRAIN="${REQUIRE_MQ_EMBEDDING_TRAIN:-1}"
WAN_TRAIN_MODE="${WAN_TRAIN_MODE:-full}"
WAN_AUTO_FULL_MEM_GB="${WAN_AUTO_FULL_MEM_GB:-120}"
WAN_LR_RATIO="${WAN_LR_RATIO:-1}"
WAN_COND_NAME_PATTERN="${WAN_COND_NAME_PATTERN:-}"
DIT_FSDP="${DIT_FSDP:-1}"
T5_FSDP="${T5_FSDP:-0}"
USE_SP="${USE_SP:-0}"
T5_CPU="${T5_CPU:-1}"

# TI2V 首帧条件相关（来自 overfit20_ti2v_frame）
TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-49}"        # 使用较长视频帧，更贴近 ti2v_frame 场景
TI2V_MAX_AREA="${TI2V_MAX_AREA:-262144}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-0.3}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
DIT_CONDITION_MODE="${DIT_CONDITION_MODE:-mq_only}"
ENABLE_TI2V_FIRST_FRAME_CONDITION="${ENABLE_TI2V_FIRST_FRAME_CONDITION:-1}"
TRAIN_VIDEO_CONDITIONING_MODE="${TRAIN_VIDEO_CONDITIONING_MODE:-legacy_t2v}"  # legacy_t2v | wan_animate_slot
TRAIN_ANIMATE_REF_FRAMES="${TRAIN_ANIMATE_REF_FRAMES:-1}"
TRAIN_ANIMATE_TEMPORAL_FRAMES="${TRAIN_ANIMATE_TEMPORAL_FRAMES:-0}"
TRAIN_ANIMATE_CONDITIONAL_FRAMES="${TRAIN_ANIMATE_CONDITIONAL_FRAMES:-0}"
TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO="${TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO:-1}"
TRAIN_ANIMATE_DROP_PREFIX_LOSS="${TRAIN_ANIMATE_DROP_PREFIX_LOSS:-1}"
TRAIN_REF_ANCHOR_MODE="${TRAIN_REF_ANCHOR_MODE:-animate_like}"
TRAIN_REF_ANCHOR_ALPHA0="${TRAIN_REF_ANCHOR_ALPHA0:-1.0}"
TRAIN_REF_ANCHOR_WARMUP_RATIO="${TRAIN_REF_ANCHOR_WARMUP_RATIO:-1.0}"

if [[ "${TASK}" != "ti2v" ]]; then
  echo "[ERROR] This launcher is TI2V-only. TASK=${TASK}"
  exit 3
fi

if [[ "${T5_FSDP}" == "1" && "${T5_CPU}" == "1" ]]; then
  echo "[WARN] T5_FSDP=1 与 T5_CPU=1 互斥，自动将 T5_CPU: 1 -> 0"
  T5_CPU="0"
fi

if [[ "${DIT_FSDP}" != "1" ]]; then
  echo "[WARN] 当前脚本目标是 2 卡 FSDP，自动设置 DIT_FSDP=1"
  DIT_FSDP="1"
fi

if [[ "${DIT_FSDP}" == "1" || "${USE_SP}" == "1" ]]; then
  export WAN_DIST_PROCESS_DEVICE="${WAN_DIST_PROCESS_DEVICE:-dit}"
else
  export WAN_DIST_PROCESS_DEVICE="${WAN_DIST_PROCESS_DEVICE:-encoder}"
fi

if [[ "${WAN_TRAIN_MODE}" != "frozen" && "${NPROC_PER_NODE}" != "1" && "${DIT_FSDP}" != "1" && "${USE_SP}" != "1" ]]; then
  echo "[WARN] 多进程 full/cond_only 建议启用 DIT_FSDP 或 USE_SP；自动降级单进程"
  NPROC_PER_NODE="1"
  GPUS_PER_PROCESS="1"
fi

METRICS_JSONL_PATH="${METRICS_JSONL_PATH:-${OUTPUT_ROOT}/logs/ti2v_overfit20_2gpu_fsdp_steps${NUM_TRAIN_STEPS}_$(date +%Y%m%d-%H%M%S).jsonl}"
METRICS_JSONL_PATH="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${METRICS_JSONL_PATH}")"

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
  --cooldown_steps "${COOLDOWN_STEPS}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --lr_min_ratio "${LR_MIN_RATIO}"
  --save_steps "${SAVE_STEPS}"
  --log_steps "${LOG_STEPS}"
  --loss_early_stop_min_step "${LOSS_EARLY_STOP_MIN_STEP}"
  --loss_early_stop_threshold "${LOSS_EARLY_STOP_THRESHOLD}"
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --frame_num "${TI2V_FRAME_NUM}"
  --max_area "${TI2V_MAX_AREA}"
  --min_duration_sec "${MIN_DURATION_SEC}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --wan_train_mode "${WAN_TRAIN_MODE}"
  --wan_auto_full_mem_gb "${WAN_AUTO_FULL_MEM_GB}"
  --wan_lr_ratio "${WAN_LR_RATIO}"
  --train_video_conditioning_mode "${TRAIN_VIDEO_CONDITIONING_MODE}"
  --train_animate_ref_frames "${TRAIN_ANIMATE_REF_FRAMES}"
  --train_animate_temporal_frames "${TRAIN_ANIMATE_TEMPORAL_FRAMES}"
  --train_animate_conditional_frames "${TRAIN_ANIMATE_CONDITIONAL_FRAMES}"
  --train_ref_anchor_mode "${TRAIN_REF_ANCHOR_MODE}"
  --train_ref_anchor_alpha0 "${TRAIN_REF_ANCHOR_ALPHA0}"
  --train_ref_anchor_warmup_ratio "${TRAIN_REF_ANCHOR_WARMUP_RATIO}"
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

if [[ "${ENABLE_TI2V_FIRST_FRAME_CONDITION}" == "1" ]]; then
  COMMON_ARGS+=(--enable_ti2v_first_frame_condition)
else
  COMMON_ARGS+=(--disable_ti2v_first_frame_condition)
fi

if [[ "${TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO}" == "1" ]]; then
  COMMON_ARGS+=(--train_animate_preserve_timestep_zero)
else
  COMMON_ARGS+=(--train_animate_no_preserve_timestep_zero)
fi

if [[ "${TRAIN_ANIMATE_DROP_PREFIX_LOSS}" == "1" ]]; then
  COMMON_ARGS+=(--train_animate_drop_prefix_loss)
else
  COMMON_ARGS+=(--train_animate_no_drop_prefix_loss)
fi

if [[ -n "${WAN_COND_NAME_PATTERN}" ]]; then
  COMMON_ARGS+=(--wan_cond_name_pattern "${WAN_COND_NAME_PATTERN}")
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
  COMMON_ARGS+=(--enable_wan_func_distill)
else
  COMMON_ARGS+=(--disable_wan_func_distill)
fi

if [[ "${REQUIRE_MQ_EMBEDDING_TRAIN}" == "1" && "${TRAIN_MQ_INPUT_EMBEDDINGS}" != "1" ]]; then
  echo "[FATAL] REQUIRE_MQ_EMBEDDING_TRAIN=1 但 TRAIN_MQ_INPUT_EMBEDDINGS=${TRAIN_MQ_INPUT_EMBEDDINGS}"
  exit 11
fi

if [[ "${LOG_EVERY_STEP}" == "1" ]]; then
  COMMON_ARGS+=(--log_every_step)
fi
if [[ "${LOG_CUDA_MEMORY}" == "1" ]]; then
  COMMON_ARGS+=(--log_cuda_memory)
fi
if [[ -n "${RESUME_MQ_ENCODER_PATH:-}" ]]; then
  COMMON_ARGS+=(--resume_mq_encoder_path "${RESUME_MQ_ENCODER_PATH}")
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

RUN_OUTPUT_DIR="${OUTPUT_ROOT}/ti2v_overfit20_2gpu_fsdp_steps${NUM_TRAIN_STEPS}_nummq${NUM_METAQUERIES}_nullimg${NULL_IMAGE_PROB}_nullcap${NULL_CAPTION_PROB}"

echo "[LAUNCH][OVERFIT20-2GPU] task=${TASK} steps=${NUM_TRAIN_STEPS} kept=${KEPT_COUNT}"
echo "[LAUNCH][OVERFIT20-2GPU] run_output_dir=${RUN_OUTPUT_DIR}"
echo "[LAUNCH][OVERFIT20-2GPU] topology nproc_per_node=${NPROC_PER_NODE} gpus_per_process=${GPUS_PER_PROCESS} cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[LAUNCH][OVERFIT20-2GPU] fsdp dit=${DIT_FSDP} t5=${T5_FSDP} use_sp=${USE_SP} t5_cpu=${T5_CPU} process_device_pref=${WAN_DIST_PROCESS_DEVICE}"
echo "[LAUNCH][OVERFIT20-2GPU] wan_train_mode=${WAN_TRAIN_MODE} auto_full_mem_gb=${WAN_AUTO_FULL_MEM_GB} wan_lr_ratio=${WAN_LR_RATIO}"
echo "[LAUNCH][OVERFIT20-2GPU] first_frame_condition=${ENABLE_TI2V_FIRST_FRAME_CONDITION} mode=${TRAIN_VIDEO_CONDITIONING_MODE} ref_frames=${TRAIN_ANIMATE_REF_FRAMES}"
echo "[LAUNCH][OVERFIT20-2GPU] dist_timeout_sec=${WAN_DIST_TIMEOUT_SEC}"
echo "[LAUNCH][OVERFIT20-2GPU] batch_size=${BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} lr=${LEARNING_RATE}"

torchrun --nproc_per_node "${NPROC_PER_NODE}" /home/liuzhirui/model/Wan2.2/train_metaquery_wan_new.py \
  --wan_checkpoint_dir "${WAN_TI2V_CKPT}" \
  --output_dir "${RUN_OUTPUT_DIR}" \
  "${COMMON_ARGS[@]}"

RUN_OUTPUT_DIR="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${RUN_OUTPUT_DIR}")"
echo "[OVERFIT20-2GPU][VERIFY] before checkpoint: ${RUN_OUTPUT_DIR}/checkpoint-before-training"
echo "[OVERFIT20-2GPU][VERIFY] final checkpoint : ${RUN_OUTPUT_DIR}/checkpoint-final"
echo "[OVERFIT20-2GPU][VERIFY] chain manifest   : ${RUN_OUTPUT_DIR}/training_chain_manifest.json"
echo "[OVERFIT20-2GPU][VERIFY] pair report      : ${PAIR_REPORT_JSON}"
echo "[OVERFIT20-2GPU][VERIFY] audit cmd:"
echo "python /home/liuzhirui/model/Wan2.2/verify_metaquery_chain.py \\"
echo "  --before_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-before-training \\"
echo "  --after_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-final \\"
echo "  --output_json ${RUN_OUTPUT_DIR}/chain_audit_report.json --strict"

