#!/bin/bash
# set -euo pipefail

# # Usage:
# #   bash train_stage1_openvid_local_metaquery_overfit10.sh
# #
# # Goal:
# #   Overfit 10 fixed OpenVid videos for pipeline sanity-check (100 steps by default).
# #   The script will:
# #   1) Build a tiny local dataset (video symlink dir + mini csv with caption)
# #   2) Print KEEP/DROP details for those 10 videos before training
# #   3) Enable dataset debug print during preclean (KEEP/DROP + reason + caption)
# #   4) Run train_metaquery_*_new.py with tiny dataset

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# WAN_ROOT="${WAN_ROOT:-${SCRIPT_DIR}}"

# source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
# conda activate /home/liuzhirui/miniconda3/envs/moviestory

# export http_proxy="${http_proxy:-10.130.130.6:56830}"
# export https_proxy="${https_proxy:-10.130.130.6:56830}"
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# export HF_TOKEN="${HF_TOKEN:-}"
# export PYTHONUNBUFFERED=1
# export TORCH_NCCL_BLOCKING_WAIT=1
# export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"

# TASK="${TASK:-ti2v}"  # ti2v | i2v | animate
# NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-52}"
# SAVE_STEPS="${SAVE_STEPS:-10}"
# LOG_STEPS="${LOG_STEPS:-1}"
# LOG_EVERY_STEP="${LOG_EVERY_STEP:-1}"
# NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
# GPUS_PER_PROCESS="${GPUS_PER_PROCESS:-1}"
# SEED="${SEED:-42}"

# OPENVID_VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video}"
# OPENVID_CSV_PATH="${OPENVID_CSV_PATH:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/data/train/OpenVid-1M.csv}"

# QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
# WAN_TI2V_CKPT="${WAN_TI2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
# WAN_I2V_CKPT="${WAN_I2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B}"
# WAN_ANIMATE_CKPT="${WAN_ANIMATE_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-Animate-14B}"

# OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_animate_like_v2}"
# OVERFIT_ROOT="${OVERFIT_ROOT:-/home/liuzhirui/model/Wan2.2/tmp/metaquery_overfit10}"
# OVERFIT_ROOT="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${OVERFIT_ROOT}")"
# mkdir -p "${OVERFIT_ROOT}"

# VIDEO_LIST_FILE="${OVERFIT_ROOT}/video_list_overfit10.txt"
# MINI_VIDEO_ROOT="${OVERFIT_ROOT}/videos"
# MINI_CSV_PATH="${OVERFIT_ROOT}/openvid_overfit10.csv"
# PAIR_REPORT_JSON="${OVERFIT_ROOT}/pair_report.json"

# cat > "${VIDEO_LIST_FILE}" << 'EOF'
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv___f2KtcXAxI_1.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv___lRwnjxeCg_3.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__0yBbZJZqG8_1.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__1osQSmJ2-s_4.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__1s0YQ8dL04_0.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__4DRXSdPuCo_2.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__4DRXSdPuCo_4.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__4juqo20ABE_0.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__5ukjsqqLg4_12.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__6RI-8Ia4do_0.mp4
# EOF

# rm -rf "${MINI_VIDEO_ROOT}"
# mkdir -p "${MINI_VIDEO_ROOT}"

# export VIDEO_LIST_FILE MINI_VIDEO_ROOT MINI_CSV_PATH PAIR_REPORT_JSON OPENVID_VIDEO_ROOT OPENVID_CSV_PATH
# python - << 'PY'
# import csv
# import json
# import os
# import shutil
# from pathlib import Path

# video_list_file = Path(os.environ["VIDEO_LIST_FILE"])
# mini_video_root = Path(os.environ["MINI_VIDEO_ROOT"])
# mini_csv_path = Path(os.environ["MINI_CSV_PATH"])
# pair_report_json = Path(os.environ["PAIR_REPORT_JSON"])
# openvid_root = Path(os.environ["OPENVID_VIDEO_ROOT"]).resolve()
# openvid_csv = Path(os.environ["OPENVID_CSV_PATH"]).resolve()

# def norm_key(s: str) -> str:
#     s = str(s).strip().replace("\\", "/")
#     while s.startswith("./"):
#         s = s[2:]
#     return s.lstrip("/").lower()

# def find_cols(fieldnames):
#     lowered = {str(k).strip().lower(): k for k in (fieldnames or [])}
#     video_col = None
#     caption_col = None
#     for k in ("video", "video_path", "filename", "file", "path"):
#         if k in lowered:
#             video_col = lowered[k]
#             break
#     for k in ("caption", "text", "description", "prompt", "summary"):
#         if k in lowered:
#             caption_col = lowered[k]
#             break
#     return video_col, caption_col

# if not openvid_csv.exists():
#     raise FileNotFoundError(f"OpenVid csv not found: {openvid_csv}")

# path_to_caption = {}
# name_to_caption = {}
# with open(openvid_csv, "r", encoding="utf-8", newline="") as f:
#     reader = csv.DictReader(f)
#     video_col, caption_col = find_cols(reader.fieldnames)
#     if video_col is None or caption_col is None:
#         raise RuntimeError(f"Cannot find video/caption columns from {openvid_csv}")
#     for row in reader:
#         v = str(row.get(video_col, "") or "").strip()
#         c = str(row.get(caption_col, "") or "").strip()
#         if not v or not c:
#             continue
#         vk = norm_key(v)
#         nk = norm_key(Path(v).name)
#         if vk and (vk not in path_to_caption or len(c) > len(path_to_caption[vk])):
#             path_to_caption[vk] = c
#         if nk and (nk not in name_to_caption or len(c) > len(name_to_caption[nk])):
#             name_to_caption[nk] = c

# kept = []
# dropped = []

# videos = [ln.strip() for ln in video_list_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
# for idx, raw in enumerate(videos):
#     src = Path(raw).resolve()
#     if not src.exists():
#         dropped.append({"video": str(src), "reason": "video_not_found"})
#         print(f"[OVERFIT10][DROP] video={src} reason=video_not_found")
#         continue

#     rel_key = ""
#     try:
#         rel_key = norm_key(src.relative_to(openvid_root))
#     except Exception:
#         rel_key = norm_key(src.name)

#     cap = path_to_caption.get(rel_key)
#     if cap is None:
#         cap = name_to_caption.get(norm_key(src.name))
#     if not cap:
#         dropped.append({"video": str(src), "reason": "caption_not_found"})
#         print(f"[OVERFIT10][DROP] video={src} reason=caption_not_found")
#         continue

#     target_name = src.name
#     # avoid basename collision
#     if (mini_video_root / target_name).exists():
#         target_name = f"{idx:02d}_{src.name}"
#     dst = mini_video_root / target_name

#     linked = False
#     try:
#         os.symlink(src, dst)
#         linked = True
#     except Exception:
#         try:
#             os.link(src, dst)
#             linked = True
#         except Exception:
#             shutil.copy2(src, dst)
#             linked = False

#     kept.append({"video": target_name, "caption": cap, "src": str(src), "link_type": "symlink_or_hardlink" if linked else "copy"})
#     print(f"[OVERFIT10][KEEP] video={target_name} src={src} caption={cap}")

# with open(mini_csv_path, "w", encoding="utf-8", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=["video", "caption"])
#     writer.writeheader()
#     for row in kept:
#         writer.writerow({"video": row["video"], "caption": row["caption"]})

# report = {
#     "openvid_csv": str(openvid_csv),
#     "openvid_root": str(openvid_root),
#     "mini_video_root": str(mini_video_root.resolve()),
#     "mini_csv": str(mini_csv_path.resolve()),
#     "kept_count": len(kept),
#     "drop_count": len(dropped),
#     "kept": kept,
#     "dropped": dropped,
# }
# pair_report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
# print(f"[OVERFIT10] mini_csv={mini_csv_path.resolve()} kept={len(kept)} dropped={len(dropped)}")
# print(f"[OVERFIT10] pair_report={pair_report_json.resolve()}")
# PY

# if [[ ! -f "${MINI_CSV_PATH}" ]]; then
#   echo "[ERROR] mini csv not generated: ${MINI_CSV_PATH}"
#   exit 1
# fi

# KEPT_COUNT="$(python - << 'PY'
# import csv, os
# from pathlib import Path
# p = Path(os.environ["MINI_CSV_PATH"])
# if not p.exists():
#     print(0)
# else:
#     with open(p, "r", encoding="utf-8", newline="") as f:
#         n = sum(1 for _ in csv.DictReader(f))
#     print(n)
# PY
# )"

# if [[ "${KEPT_COUNT}" -le 0 ]]; then
#   echo "[ERROR] kept_count=${KEPT_COUNT}, nothing to train."
#   exit 2
# fi

# echo "[OVERFIT10] final_kept_count=${KEPT_COUNT}"
# echo "[OVERFIT10] mini_video_root=${MINI_VIDEO_ROOT}"
# echo "[OVERFIT10] mini_csv=${MINI_CSV_PATH}"

# # Dataset debug print (KEEP/DROP during preclean with reason+caption)
# export WAN_LOCAL_OVERFIT_DEBUG=1
# export WAN_LOCAL_MISSING_CAPTION_PRINT_MAX=1000
# export WAN_DATA_PRECLEAN=1

# # disable subset cache so every run prints full KEEP/DROP details
# HF_NO_SUBSET_CACHE_FLAG="--hf_no_subset_cache"

# WANDB_ENABLED="${WANDB_ENABLED:-0}"
# WANDB_PROJECT="${WANDB_PROJECT:-wan-metaquery-overfit10}"
# WANDB_ENTITY="${WANDB_ENTITY:-}"
# WANDB_RUN_NAME="${WANDB_RUN_NAME:-overfit10-${TASK}-steps${NUM_TRAIN_STEPS}-$(date +%Y%m%d-%H%M%S)}"
# WANDB_TAGS="${WANDB_TAGS:-overfit,debug,overfit10}"
# WANDB_MODE="${WANDB_MODE:-offline}"

# LOG_CUDA_MEMORY="${LOG_CUDA_MEMORY:-1}"
# WANDB_LOG_EVERY_STEP="${WANDB_LOG_EVERY_STEP:-1}"
# WANDB_LOG_CHECKPOINT="${WANDB_LOG_CHECKPOINT:-1}"

# NUM_METAQUERIES="${NUM_METAQUERIES:-256}"
# NULL_IMAGE_PROB="${NULL_IMAGE_PROB:-0.2}"
# NULL_CAPTION_PROB="${NULL_CAPTION_PROB:-0.0}"

# # TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-29}"
# TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-49}"
# I2V_FRAME_NUM="${I2V_FRAME_NUM:-49}"
# ANIMATE_FRAME_NUM="${ANIMATE_FRAME_NUM:-49}"
# # TI2V_MAX_AREA="${TI2V_MAX_AREA:-147456}"
# TI2V_MAX_AREA="${TI2V_MAX_AREA:-262144}"
# I2V_MAX_AREA="${I2V_MAX_AREA:-262144}"
# ANIMATE_MAX_AREA="${ANIMATE_MAX_AREA:-262144}"
# MIN_DURATION_SEC="${MIN_DURATION_SEC:-0.3}"
# TRAIN_VIDEO_CONDITIONING_MODE="${TRAIN_VIDEO_CONDITIONING_MODE:-wan_animate_slot}" # legacy_t2v | wan_animate_slot
# TRAIN_ANIMATE_REF_FRAMES="${TRAIN_ANIMATE_REF_FRAMES:-1}"
# TRAIN_ANIMATE_TEMPORAL_FRAMES="${TRAIN_ANIMATE_TEMPORAL_FRAMES:-0}"
# TRAIN_ANIMATE_CONDITIONAL_FRAMES="${TRAIN_ANIMATE_CONDITIONAL_FRAMES:-0}"
# TRAIN_ANIMATE_DROP_PREFIX_LOSS="${TRAIN_ANIMATE_DROP_PREFIX_LOSS:-1}"
# TRAIN_ANIMATE_PRESERVE_T0="${TRAIN_ANIMATE_PRESERVE_T0:-1}"
# TRAIN_REF_ANCHOR_MODE="${TRAIN_REF_ANCHOR_MODE:-none}"   # none | animate_like | mixed50
# TRAIN_REF_ANCHOR_ALPHA0="${TRAIN_REF_ANCHOR_ALPHA0:-0.95}"
# TRAIN_REF_ANCHOR_WARMUP_RATIO="${TRAIN_REF_ANCHOR_WARMUP_RATIO:-0.35}"

# METRICS_JSONL_PATH="${METRICS_JSONL_PATH:-${OUTPUT_ROOT}/logs/${TASK}_overfit10_steps${NUM_TRAIN_STEPS}_$(date +%Y%m%d-%H%M%S).jsonl}"
# METRICS_JSONL_PATH="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${METRICS_JSONL_PATH}")"

# COMMON_ARGS=(
#   --distributed
#   --auto_device_map
#   --gpus_per_process "${GPUS_PER_PROCESS}"
#   --hf_stage stage1
#   --hf_no_streaming
#   "${HF_NO_SUBSET_CACHE_FLAG}"
#   --t5_cpu
#   --mq_gradient_checkpointing
#   --aggressive_empty_cache
#   --local_openvid_video_root "${MINI_VIDEO_ROOT}"
#   --local_openvid_csv_path "${MINI_CSV_PATH}"
#   --local_openvid_limit "${KEPT_COUNT}"
#   --local_openvid_hd_video_root ""
#   --local_openvid_hd_csv_path ""
#   --qwen3vl_model_id "${QWEN_MODEL}"
#   --num_metaqueries "${NUM_METAQUERIES}"
#   --null_image_prob "${NULL_IMAGE_PROB}"
#   --null_caption_prob "${NULL_CAPTION_PROB}"
#   --num_train_steps "${NUM_TRAIN_STEPS}"
#   --save_steps "${SAVE_STEPS}"
#   --log_steps "${LOG_STEPS}"
#   --seed "${SEED}"
#   --metrics_jsonl_path "${METRICS_JSONL_PATH}"
# )

# if [[ "${LOG_EVERY_STEP}" == "1" ]]; then
#   COMMON_ARGS+=(--log_every_step)
# fi
# if [[ "${LOG_CUDA_MEMORY}" == "1" ]]; then
#   COMMON_ARGS+=(--log_cuda_memory)
# fi
# if [[ "${WANDB_ENABLED}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
#   COMMON_ARGS+=(--wandb_enabled)
#   COMMON_ARGS+=(--wandb_project "${WANDB_PROJECT}")
#   if [[ -n "${WANDB_ENTITY}" ]]; then
#     COMMON_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
#   fi
#   COMMON_ARGS+=(--wandb_mode "${WANDB_MODE}")
#   COMMON_ARGS+=(--wandb_api_key "${WANDB_API_KEY}")
#   COMMON_ARGS+=(--wandb_tags "${WANDB_TAGS}")
#   COMMON_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
#   if [[ "${WANDB_LOG_EVERY_STEP}" == "1" ]]; then
#     COMMON_ARGS+=(--wandb_log_every_step)
#   fi
#   if [[ "${WANDB_LOG_CHECKPOINT}" == "1" ]]; then
#     COMMON_ARGS+=(--wandb_log_checkpoint)
#   fi
# fi

# echo "[LAUNCH][OVERFIT10] TASK=${TASK} steps=${NUM_TRAIN_STEPS} kept=${KEPT_COUNT}"
# echo "[LAUNCH][OVERFIT10] mini_root=${MINI_VIDEO_ROOT}"
# echo "[LAUNCH][OVERFIT10] mini_csv=${MINI_CSV_PATH}"
# echo "[LAUNCH][OVERFIT10] metrics_jsonl=${METRICS_JSONL_PATH}"
# echo "[LAUNCH][OVERFIT10] null_image_prob=${NULL_IMAGE_PROB} null_caption_prob=${NULL_CAPTION_PROB}"
# echo "[LAUNCH][OVERFIT10] train_video_conditioning_mode=${TRAIN_VIDEO_CONDITIONING_MODE} ref_frames=${TRAIN_ANIMATE_REF_FRAMES} temporal_frames=${TRAIN_ANIMATE_TEMPORAL_FRAMES} conditional_frames=${TRAIN_ANIMATE_CONDITIONAL_FRAMES}"
# echo "[LAUNCH][OVERFIT10] train_ref_anchor_mode=${TRAIN_REF_ANCHOR_MODE} alpha0=${TRAIN_REF_ANCHOR_ALPHA0} warmup_ratio=${TRAIN_REF_ANCHOR_WARMUP_RATIO}"

# TI2V_MODE_EXTRA_ARGS=()
# if [[ "${TRAIN_ANIMATE_DROP_PREFIX_LOSS}" == "1" ]]; then
#   TI2V_MODE_EXTRA_ARGS+=(--train_animate_drop_prefix_loss)
# else
#   TI2V_MODE_EXTRA_ARGS+=(--train_animate_no_drop_prefix_loss)
# fi
# if [[ "${TRAIN_ANIMATE_PRESERVE_T0}" == "1" ]]; then
#   TI2V_MODE_EXTRA_ARGS+=(--train_animate_preserve_timestep_zero)
# else
#   TI2V_MODE_EXTRA_ARGS+=(--train_animate_no_preserve_timestep_zero)
# fi

# case "${TASK}" in
#   ti2v)
#     RUN_OUTPUT_DIR="${OUTPUT_ROOT}/ti2v_overfit10_steps${NUM_TRAIN_STEPS}"
#     torchrun --nproc_per_node "${NPROC_PER_NODE}" "${WAN_ROOT}/train_metaquery_wan_new_animate_like_v2.py" \
#       --wan_checkpoint_dir "${WAN_TI2V_CKPT}" \
#       --output_dir "${RUN_OUTPUT_DIR}" \
#       --frame_num "${TI2V_FRAME_NUM}" \
#       --max_area "${TI2V_MAX_AREA}" \
#       --min_duration_sec "${MIN_DURATION_SEC}" \
#       --train_video_conditioning_mode "${TRAIN_VIDEO_CONDITIONING_MODE}" \
#       --train_animate_ref_frames "${TRAIN_ANIMATE_REF_FRAMES}" \
#       --train_animate_temporal_frames "${TRAIN_ANIMATE_TEMPORAL_FRAMES}" \
#       --train_animate_conditional_frames "${TRAIN_ANIMATE_CONDITIONAL_FRAMES}" \
#       --train_ref_anchor_mode "${TRAIN_REF_ANCHOR_MODE}" \
#       --train_ref_anchor_alpha0 "${TRAIN_REF_ANCHOR_ALPHA0}" \
#       --train_ref_anchor_warmup_ratio "${TRAIN_REF_ANCHOR_WARMUP_RATIO}" \
#       "${TI2V_MODE_EXTRA_ARGS[@]}" \
#       "${COMMON_ARGS[@]}"
#     ;;
#   i2v)
#     RUN_OUTPUT_DIR="${OUTPUT_ROOT}/i2v_overfit10_steps${NUM_TRAIN_STEPS}"
#     torchrun --nproc_per_node "${NPROC_PER_NODE}" "${WAN_ROOT}/train_metaquery_i2v_new.py" \
#       --wan_checkpoint_dir "${WAN_I2V_CKPT}" \
#       --output_dir "${RUN_OUTPUT_DIR}" \
#       --frame_num "${I2V_FRAME_NUM}" \
#       --max_area "${I2V_MAX_AREA}" \
#       --min_duration_sec "${MIN_DURATION_SEC}" \
#       "${COMMON_ARGS[@]}"
#     ;;
#   animate)
#     RUN_OUTPUT_DIR="${OUTPUT_ROOT}/animate_overfit10_steps${NUM_TRAIN_STEPS}"
#     torchrun --nproc_per_node "${NPROC_PER_NODE}" "${WAN_ROOT}/train_metaquery_animate_new.py" \
#       --wan_checkpoint_dir "${WAN_ANIMATE_CKPT}" \
#       --output_dir "${RUN_OUTPUT_DIR}" \
#       --frame_num "${ANIMATE_FRAME_NUM}" \
#       --max_area "${ANIMATE_MAX_AREA}" \
#       --min_duration_sec "${MIN_DURATION_SEC}" \
#       "${COMMON_ARGS[@]}"
#     ;;
#   *)
#     echo "[ERROR] TASK must be one of: ti2v | i2v | animate"
#     exit 3
#     ;;
# esac

# RUN_OUTPUT_DIR="$(python -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${RUN_OUTPUT_DIR}")"
# echo "[OVERFIT10][VERIFY] before checkpoint: ${RUN_OUTPUT_DIR}/checkpoint-before-training"
# echo "[OVERFIT10][VERIFY] final checkpoint : ${RUN_OUTPUT_DIR}/checkpoint-final"
# echo "[OVERFIT10][VERIFY] chain manifest   : ${RUN_OUTPUT_DIR}/training_chain_manifest.json"
# echo "[OVERFIT10][VERIFY] pair report      : ${PAIR_REPORT_JSON}"
# echo "[OVERFIT10][VERIFY] audit cmd:"
# echo "python ${WAN_ROOT}/verify_metaquery_chain.py \\"
# echo "  --before_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-before-training \\"
# echo "  --after_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-final \\"
# echo "  --output_json ${RUN_OUTPUT_DIR}/chain_audit_report.json --strict"

# 上面对应T5 + MQ 作为条件，下面这个是加上了MQ-ONLY的版本
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAN_ROOT="${WAN_ROOT:-${SCRIPT_DIR}}"

source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

export http_proxy="${http_proxy:-10.130.130.6:56830}"
export https_proxy="${https_proxy:-10.130.130.6:56830}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"

TASK="${TASK:-ti2v}"  # ti2v | i2v | animate
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOG_STEPS="${LOG_STEPS:-1}"
LOG_EVERY_STEP="${LOG_EVERY_STEP:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
GPUS_PER_PROCESS="${GPUS_PER_PROCESS:-1}"
SEED="${SEED:-42}"

OPENVID_VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video}"
OPENVID_CSV_PATH="${OPENVID_CSV_PATH:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/data/train/OpenVid-1M.csv}"

QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
WAN_TI2V_CKPT="${WAN_TI2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}"
WAN_I2V_CKPT="${WAN_I2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-I2V-A14B}"
WAN_ANIMATE_CKPT="${WAN_ANIMATE_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-Animate-14B}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_animate_like_v2}"
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
NULL_IMAGE_PROB="${NULL_IMAGE_PROB:-0.2}"
NULL_CAPTION_PROB="${NULL_CAPTION_PROB:-0.0}"

# TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-29}"
TI2V_FRAME_NUM="${TI2V_FRAME_NUM:-49}"
I2V_FRAME_NUM="${I2V_FRAME_NUM:-49}"
ANIMATE_FRAME_NUM="${ANIMATE_FRAME_NUM:-49}"
# TI2V_MAX_AREA="${TI2V_MAX_AREA:-147456}"
TI2V_MAX_AREA="${TI2V_MAX_AREA:-262144}"
I2V_MAX_AREA="${I2V_MAX_AREA:-262144}"
ANIMATE_MAX_AREA="${ANIMATE_MAX_AREA:-262144}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-0.3}"
TRAIN_VIDEO_CONDITIONING_MODE="${TRAIN_VIDEO_CONDITIONING_MODE:-wan_animate_slot}" # legacy_t2v | wan_animate_slot
TRAIN_ANIMATE_REF_FRAMES="${TRAIN_ANIMATE_REF_FRAMES:-1}"
TRAIN_ANIMATE_TEMPORAL_FRAMES="${TRAIN_ANIMATE_TEMPORAL_FRAMES:-0}"
TRAIN_ANIMATE_CONDITIONAL_FRAMES="${TRAIN_ANIMATE_CONDITIONAL_FRAMES:-0}"
TRAIN_ANIMATE_DROP_PREFIX_LOSS="${TRAIN_ANIMATE_DROP_PREFIX_LOSS:-1}"
TRAIN_ANIMATE_PRESERVE_T0="${TRAIN_ANIMATE_PRESERVE_T0:-1}"
TRAIN_REF_ANCHOR_MODE="${TRAIN_REF_ANCHOR_MODE:-none}"   # none | animate_like | mixed50
TRAIN_REF_ANCHOR_ALPHA0="${TRAIN_REF_ANCHOR_ALPHA0:-0.95}"
TRAIN_REF_ANCHOR_WARMUP_RATIO="${TRAIN_REF_ANCHOR_WARMUP_RATIO:-0.35}"
DIT_CONDITION_MODE="${DIT_CONDITION_MODE:-mq_only}"

METRICS_JSONL_PATH="${METRICS_JSONL_PATH:-${OUTPUT_ROOT}/logs/${TASK}_overfit10_steps${NUM_TRAIN_STEPS}_$(date +%Y%m%d-%H%M%S).jsonl}"
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
  --num_metaqueries "${NUM_METAQUERIES}"
  --null_image_prob "${NULL_IMAGE_PROB}"
  --null_caption_prob "${NULL_CAPTION_PROB}"
  --num_train_steps "${NUM_TRAIN_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --log_steps "${LOG_STEPS}"
  --seed "${SEED}"
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

echo "[LAUNCH][OVERFIT10] TASK=${TASK} steps=${NUM_TRAIN_STEPS} kept=${KEPT_COUNT}"
echo "[LAUNCH][OVERFIT10] mini_root=${MINI_VIDEO_ROOT}"
echo "[LAUNCH][OVERFIT10] mini_csv=${MINI_CSV_PATH}"
echo "[LAUNCH][OVERFIT10] metrics_jsonl=${METRICS_JSONL_PATH}"
echo "[LAUNCH][OVERFIT10] null_image_prob=${NULL_IMAGE_PROB} null_caption_prob=${NULL_CAPTION_PROB}"
echo "[LAUNCH][OVERFIT10] train_video_conditioning_mode=${TRAIN_VIDEO_CONDITIONING_MODE} ref_frames=${TRAIN_ANIMATE_REF_FRAMES} temporal_frames=${TRAIN_ANIMATE_TEMPORAL_FRAMES} conditional_frames=${TRAIN_ANIMATE_CONDITIONAL_FRAMES}"
echo "[LAUNCH][OVERFIT10] train_ref_anchor_mode=${TRAIN_REF_ANCHOR_MODE} alpha0=${TRAIN_REF_ANCHOR_ALPHA0} warmup_ratio=${TRAIN_REF_ANCHOR_WARMUP_RATIO}"

TI2V_MODE_EXTRA_ARGS=()
if [[ "${TRAIN_ANIMATE_DROP_PREFIX_LOSS}" == "1" ]]; then
  TI2V_MODE_EXTRA_ARGS+=(--train_animate_drop_prefix_loss)
else
  TI2V_MODE_EXTRA_ARGS+=(--train_animate_no_drop_prefix_loss)
fi
if [[ "${TRAIN_ANIMATE_PRESERVE_T0}" == "1" ]]; then
  TI2V_MODE_EXTRA_ARGS+=(--train_animate_preserve_timestep_zero)
else
  TI2V_MODE_EXTRA_ARGS+=(--train_animate_no_preserve_timestep_zero)
fi

case "${TASK}" in
  ti2v)
    RUN_OUTPUT_DIR="${OUTPUT_ROOT}/ti2v_overfit10_steps_mq_only${NUM_TRAIN_STEPS}"
    torchrun --nproc_per_node "${NPROC_PER_NODE}" "${WAN_ROOT}/train_metaquery_wan_new_animate_like_v2.py" \
      --wan_checkpoint_dir "${WAN_TI2V_CKPT}" \
      --output_dir "${RUN_OUTPUT_DIR}" \
      --frame_num "${TI2V_FRAME_NUM}" \
      --max_area "${TI2V_MAX_AREA}" \
      --min_duration_sec "${MIN_DURATION_SEC}" \
      --dit_condition_mode "${DIT_CONDITION_MODE}" \
      --train_video_conditioning_mode "${TRAIN_VIDEO_CONDITIONING_MODE}" \
      --train_animate_ref_frames "${TRAIN_ANIMATE_REF_FRAMES}" \
      --train_animate_temporal_frames "${TRAIN_ANIMATE_TEMPORAL_FRAMES}" \
      --train_animate_conditional_frames "${TRAIN_ANIMATE_CONDITIONAL_FRAMES}" \
      --train_ref_anchor_mode "${TRAIN_REF_ANCHOR_MODE}" \
      --train_ref_anchor_alpha0 "${TRAIN_REF_ANCHOR_ALPHA0}" \
      --train_ref_anchor_warmup_ratio "${TRAIN_REF_ANCHOR_WARMUP_RATIO}" \
      "${TI2V_MODE_EXTRA_ARGS[@]}" \
      "${COMMON_ARGS[@]}"
    ;;
  i2v)
    RUN_OUTPUT_DIR="${OUTPUT_ROOT}/i2v_overfit10_steps_mq_only${NUM_TRAIN_STEPS}"
    torchrun --nproc_per_node "${NPROC_PER_NODE}" "${WAN_ROOT}/train_metaquery_i2v_new.py" \
      --wan_checkpoint_dir "${WAN_I2V_CKPT}" \
      --output_dir "${RUN_OUTPUT_DIR}" \
      --frame_num "${I2V_FRAME_NUM}" \
      --max_area "${I2V_MAX_AREA}" \
      --min_duration_sec "${MIN_DURATION_SEC}" \
      "${COMMON_ARGS[@]}"
    ;;
  animate)
    RUN_OUTPUT_DIR="${OUTPUT_ROOT}/animate_overfit10_steps_mq_only${NUM_TRAIN_STEPS}"
    torchrun --nproc_per_node "${NPROC_PER_NODE}" "${WAN_ROOT}/train_metaquery_animate_new.py" \
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
echo "python ${WAN_ROOT}/verify_metaquery_chain.py \\"
echo "  --before_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-before-training \\"
echo "  --after_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-final \\"
echo "  --output_json ${RUN_OUTPUT_DIR}/chain_audit_report.json --strict"
