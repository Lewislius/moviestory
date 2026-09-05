#!/bin/bash
# set -euo pipefail

# # Overfit launcher (TI2V path) with explicit first-frame conditioning on BOTH:
# # 1) MQ side (qwen3-vl MetaQuery image input)
# # 2) WAN side (TI2V latent first-slot condition, no I2V-style y concat)
# #
# # This is the "WAN also receives first-frame condition" variant.

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PYTHON_BIN="${PYTHON_BIN:-python}"

# source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
# conda activate /home/liuzhirui/miniconda3/envs/moviestory

# export http_proxy="${http_proxy:-10.130.130.6:56830}"
# export https_proxy="${https_proxy:-10.130.130.6:56830}"
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# export HF_TOKEN="${HF_TOKEN:-}"
# export PYTHONUNBUFFERED=1
# export TORCH_NCCL_BLOCKING_WAIT=1
# export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"

# NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
# GPUS_PER_PROCESS="${GPUS_PER_PROCESS:-1}"
# ENABLE_DISTRIBUTED_RUNTIME="${ENABLE_DISTRIBUTED_RUNTIME:-0}"  # 单卡默认不初始化 torch.distributed / NCCL
# SEED="${SEED:-42}"

# NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-920}"
# SAVE_STEPS="${SAVE_STEPS:-720}"
# LOG_STEPS="${LOG_STEPS:-1}"
# LOG_EVERY_STEP="${LOG_EVERY_STEP:-1}"
# LOG_CUDA_MEMORY="${LOG_CUDA_MEMORY:-1}"

# LEARNING_RATE="${LEARNING_RATE:-1e-5}"
# WARMUP_STEPS="${WARMUP_STEPS:-200}"
# BATCH_SIZE="${BATCH_SIZE:-1}"
# GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
# MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
# COOLDOWN_STEPS="${COOLDOWN_STEPS:-${WARMUP_STEPS}}"
# LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-warmup_hold_cooldown}"
# LR_MIN_RATIO="${LR_MIN_RATIO:-0.01}"

# # 可选早停：step >= min_step 且 train/loss_denoise < threshold 时提前停止
# ENABLE_LOSS_EARLY_STOP="${ENABLE_LOSS_EARLY_STOP:-1}"
# LOSS_EARLY_STOP_MIN_STEP="${LOSS_EARLY_STOP_MIN_STEP:-800}"
# LOSS_EARLY_STOP_THRESHOLD="${LOSS_EARLY_STOP_THRESHOLD:-0.25}"

# # 训练可训练参数范围相关
# DIT_CONDITION_MODE="${DIT_CONDITION_MODE:-mq_only}"
# TRAIN_MQ_INPUT_EMBEDDINGS="${TRAIN_MQ_INPUT_EMBEDDINGS:-1}"  # 1=训练 MQ 新增 token embedding
# REQUIRE_MQ_EMBEDDING_TRAIN="${REQUIRE_MQ_EMBEDDING_TRAIN:-1}"
# WAN_TRAIN_MODE="${WAN_TRAIN_MODE:-auto}"                     # auto | full | cond_only | frozen
# WAN_AUTO_FULL_MEM_GB="${WAN_AUTO_FULL_MEM_GB:-120}"
# WAN_LR_RATIO="${WAN_LR_RATIO:-0.4}"
# WAN_COND_NAME_PATTERN="${WAN_COND_NAME_PATTERN:-}"           # cond_only 关键字覆盖（逗号分隔）

# NUM_METAQUERIES="${NUM_METAQUERIES:-256}"
# CONNECTOR_LAYERS="${CONNECTOR_LAYERS:-24}"

# # Keep defaults conservative for overfit stability.
# NULL_IMAGE_PROB="${NULL_IMAGE_PROB:-0.1}"
# NULL_CAPTION_PROB="${NULL_CAPTION_PROB:-0.1}"

# I2V_FRAME_NUM="${I2V_FRAME_NUM:-49}"            # 4n+1
# I2V_MAX_AREA="${I2V_MAX_AREA:-262144}"
# MIN_DURATION_SEC="${MIN_DURATION_SEC:-0.3}"
# DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

# OPENVID_VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video}"
# OPENVID_CSV_PATH="${OPENVID_CSV_PATH:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/data/train/OpenVid-1M.csv}"

# QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
# WAN_TI2V_CKPT="${WAN_TI2V_CKPT:-${WAN_I2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}}"

# OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_ti2v_firstframe_wan_cond}"
# OVERFIT_ROOT="${OVERFIT_ROOT:-/home/liuzhirui/model/Wan2.2/tmp/metaquery_overfit10_ti2v_firstframe_wan_cond}"
# OVERFIT_ROOT="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${OVERFIT_ROOT}")"
# mkdir -p "${OVERFIT_ROOT}"

# VIDEO_LIST_FILE="${VIDEO_LIST_FILE:-${OVERFIT_ROOT}/video_list_overfit10.txt}"
# MINI_VIDEO_ROOT="${OVERFIT_ROOT}/videos"
# MINI_CSV_PATH="${OVERFIT_ROOT}/openvid_overfit10.csv"
# PAIR_REPORT_JSON="${OVERFIT_ROOT}/pair_report.json"

# # Keep the same overfit list style as your original overfit10 script.
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
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__IBCLj2lq_U_0.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__GxarTw5BNM_1.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__DQAqotwgd4_18.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__d68kyNY0jI_0.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__BeHUjskbZo_1.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__BEb_ZjotP0_19_0.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__a-pB_5eRt0_6.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__5ukjsqqLg4_8.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__4DRXSdPuCo_3.mp4
# /run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video/celebv__4DRXSdPuCo_0.mp4
# EOF

# rm -rf "${MINI_VIDEO_ROOT}"
# mkdir -p "${MINI_VIDEO_ROOT}"

# export VIDEO_LIST_FILE MINI_VIDEO_ROOT MINI_CSV_PATH PAIR_REPORT_JSON OPENVID_VIDEO_ROOT OPENVID_CSV_PATH
# "${PYTHON_BIN}" - << 'PY'
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
#         print(f"[I2V-WANCOND][DROP] video={src} reason=video_not_found")
#         continue
#     try:
#         rel_key = norm_key(src.relative_to(openvid_root))
#     except Exception:
#         rel_key = norm_key(src.name)
#     cap = path_to_caption.get(rel_key)
#     if cap is None:
#         cap = name_to_caption.get(norm_key(src.name))
#     if not cap:
#         dropped.append({"video": str(src), "reason": "caption_not_found"})
#         print(f"[I2V-WANCOND][DROP] video={src} reason=caption_not_found")
#         continue
#     target_name = src.name
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
#     kept.append(
#         {
#             "video": target_name,
#             "caption": cap,
#             "src": str(src),
#             "link_type": "symlink_or_hardlink" if linked else "copy",
#         }
#     )
#     print(f"[I2V-WANCOND][KEEP] video={target_name} src={src}")

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
# print(f"[I2V-WANCOND] mini_csv={mini_csv_path.resolve()} kept={len(kept)} dropped={len(dropped)}")
# print(f"[I2V-WANCOND] pair_report={pair_report_json.resolve()}")
# PY

# if [[ ! -f "${MINI_CSV_PATH}" ]]; then
#   echo "[ERROR] mini csv not generated: ${MINI_CSV_PATH}"
#   exit 1
# fi

# KEPT_COUNT="$("${PYTHON_BIN}" - << 'PY'
# import csv
# import os
# from pathlib import Path
# p = Path(os.environ["MINI_CSV_PATH"])
# if not p.exists():
#     print(0)
# else:
#     with open(p, "r", encoding="utf-8", newline="") as f:
#         print(sum(1 for _ in csv.DictReader(f)))
# PY
# )"

# if [[ "${KEPT_COUNT}" -le 0 ]]; then
#   echo "[ERROR] kept_count=${KEPT_COUNT}, nothing to train."
#   exit 2
# fi

# echo "[I2V-WANCOND] final_kept_count=${KEPT_COUNT}"
# echo "[I2V-WANCOND] mini_video_root=${MINI_VIDEO_ROOT}"
# echo "[I2V-WANCOND] mini_csv=${MINI_CSV_PATH}"

# # Dataset debug prints during preclean.
# export WAN_LOCAL_OVERFIT_DEBUG=1
# export WAN_LOCAL_MISSING_CAPTION_PRINT_MAX=1000
# export WAN_DATA_PRECLEAN=1

# METRICS_JSONL_PATH="${METRICS_JSONL_PATH:-${OUTPUT_ROOT}/logs/i2v_firstframe_wancond_overfit10_steps${NUM_TRAIN_STEPS}_$(date +%Y%m%d-%H%M%S).jsonl}"
# METRICS_JSONL_PATH="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${METRICS_JSONL_PATH}")"

# WANDB_ENABLED="${WANDB_ENABLED:-0}"
# WANDB_PROJECT="${WANDB_PROJECT:-wan-metaquery-overfit10-ti2v-firstframe-wancond}"
# WANDB_ENTITY="${WANDB_ENTITY:-}"
# WANDB_RUN_NAME="${WANDB_RUN_NAME:-overfit10-ti2v-firstframe-wancond-steps${NUM_TRAIN_STEPS}-$(date +%Y%m%d-%H%M%S)}"
# WANDB_TAGS="${WANDB_TAGS:-overfit,firstframe,ti2v,mq,wan-condition}"
# WANDB_MODE="${WANDB_MODE:-offline}"
# WANDB_LOG_EVERY_STEP="${WANDB_LOG_EVERY_STEP:-1}"
# WANDB_LOG_CHECKPOINT="${WANDB_LOG_CHECKPOINT:-1}"
# WAN_CONTEXT_MQ_ONLY="${WAN_CONTEXT_MQ_ONLY:-1}"  # 1: WAN context仅保留MQ特征(不拼接T5)
# ENABLE_TI2V_FIRST_FRAME_CONDITION="${ENABLE_TI2V_FIRST_FRAME_CONDITION:-1}"  # 1: ref_image 作为 TI2V 首个 latent slot 条件
# TRAIN_VIDEO_CONDITIONING_MODE="${TRAIN_VIDEO_CONDITIONING_MODE:-wan_animate_slot}"  # legacy_t2v | wan_animate_slot
# TRAIN_ANIMATE_REF_FRAMES="${TRAIN_ANIMATE_REF_FRAMES:-1}"
# TRAIN_ANIMATE_TEMPORAL_FRAMES="${TRAIN_ANIMATE_TEMPORAL_FRAMES:-0}"
# TRAIN_ANIMATE_CONDITIONAL_FRAMES="${TRAIN_ANIMATE_CONDITIONAL_FRAMES:-0}"
# TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO="${TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO:-1}"
# TRAIN_ANIMATE_DROP_PREFIX_LOSS="${TRAIN_ANIMATE_DROP_PREFIX_LOSS:-1}"
# TRAIN_REF_ANCHOR_MODE="${TRAIN_REF_ANCHOR_MODE:-animate_like}"   # none | animate_like | mixed50
# TRAIN_REF_ANCHOR_ALPHA0="${TRAIN_REF_ANCHOR_ALPHA0:-0.95}"
# TRAIN_REF_ANCHOR_WARMUP_RATIO="${TRAIN_REF_ANCHOR_WARMUP_RATIO:-0.35}"

# RUN_OUTPUT_DIR="${OUTPUT_ROOT}/ti2v_firstframe_wancond_overfit10_steps${NUM_TRAIN_STEPS}_nummq${NUM_METAQUERIES}_nullimg${NULL_IMAGE_PROB}_nullcap${NULL_CAPTION_PROB}"

# if [[ "${WAN_TRAIN_MODE}" != "frozen" && "${NPROC_PER_NODE}" != "1" ]]; then
#   echo "[WARN] WAN 可训练模式(${WAN_TRAIN_MODE}) 当前建议单进程训练；自动将 NPROC_PER_NODE: ${NPROC_PER_NODE} -> 1"
#   NPROC_PER_NODE="1"
#   GPUS_PER_PROCESS="1"
# fi

# COMMON_ARGS=(
#   --auto_device_map
#   --gpus_per_process "${GPUS_PER_PROCESS}"
#   --hf_stage stage1
#   --hf_no_streaming
#   --hf_no_subset_cache
#   --t5_cpu
#   --mq_gradient_checkpointing
#   --aggressive_empty_cache
#   --wan_checkpoint_dir "${WAN_TI2V_CKPT}"
#   --output_dir "${RUN_OUTPUT_DIR}"
#   --qwen3vl_model_id "${QWEN_MODEL}"
#   --dit_condition_mode "${DIT_CONDITION_MODE}"
#   --local_openvid_video_root "${MINI_VIDEO_ROOT}"
#   --local_openvid_csv_path "${MINI_CSV_PATH}"
#   --local_openvid_limit "${KEPT_COUNT}"
#   --num_metaqueries "${NUM_METAQUERIES}"
#   --connector_num_hidden_layers "${CONNECTOR_LAYERS}"
#   --null_image_prob "${NULL_IMAGE_PROB}"
#   --null_caption_prob "${NULL_CAPTION_PROB}"
#   --num_train_steps "${NUM_TRAIN_STEPS}"
#   --warmup_steps "${WARMUP_STEPS}"
#   --cooldown_steps "${COOLDOWN_STEPS}"
#   --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
#   --lr_min_ratio "${LR_MIN_RATIO}"
#   --save_steps "${SAVE_STEPS}"
#   --log_steps "${LOG_STEPS}"
#   --loss_early_stop_min_step "${LOSS_EARLY_STOP_MIN_STEP}"
#   --loss_early_stop_threshold "${LOSS_EARLY_STOP_THRESHOLD}"
#   --batch_size "${BATCH_SIZE}"
#   --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
#   --learning_rate "${LEARNING_RATE}"
#   --max_grad_norm "${MAX_GRAD_NORM}"
#   --frame_num "${I2V_FRAME_NUM}"
#   --max_area "${I2V_MAX_AREA}"
#   --min_duration_sec "${MIN_DURATION_SEC}"
#   --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
#   --wan_train_mode "${WAN_TRAIN_MODE}"
#   --wan_auto_full_mem_gb "${WAN_AUTO_FULL_MEM_GB}"
#   --wan_lr_ratio "${WAN_LR_RATIO}"
#   --train_video_conditioning_mode "${TRAIN_VIDEO_CONDITIONING_MODE}"
#   --train_animate_ref_frames "${TRAIN_ANIMATE_REF_FRAMES}"
#   --train_animate_temporal_frames "${TRAIN_ANIMATE_TEMPORAL_FRAMES}"
#   --train_animate_conditional_frames "${TRAIN_ANIMATE_CONDITIONAL_FRAMES}"
#   --train_ref_anchor_mode "${TRAIN_REF_ANCHOR_MODE}"
#   --train_ref_anchor_alpha0 "${TRAIN_REF_ANCHOR_ALPHA0}"
#   --train_ref_anchor_warmup_ratio "${TRAIN_REF_ANCHOR_WARMUP_RATIO}"
#   --seed "${SEED}"
#   --metrics_jsonl_path "${METRICS_JSONL_PATH}"
# )

# if [[ "${ENABLE_DISTRIBUTED_RUNTIME}" == "1" || "${NPROC_PER_NODE}" != "1" ]]; then
#   COMMON_ARGS+=(--distributed)
# fi

# if [[ "${TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO}" == "1" ]]; then
#   COMMON_ARGS+=(--train_animate_preserve_timestep_zero)
# else
#   COMMON_ARGS+=(--train_animate_no_preserve_timestep_zero)
# fi

# if [[ "${TRAIN_ANIMATE_DROP_PREFIX_LOSS}" == "1" ]]; then
#   COMMON_ARGS+=(--train_animate_drop_prefix_loss)
# else
#   COMMON_ARGS+=(--train_animate_no_drop_prefix_loss)
# fi

# if [[ -n "${WAN_COND_NAME_PATTERN}" ]]; then
#   COMMON_ARGS+=(--wan_cond_name_pattern "${WAN_COND_NAME_PATTERN}")
# fi

# if [[ "${TRAIN_MQ_INPUT_EMBEDDINGS}" == "1" ]]; then
#   COMMON_ARGS+=(--train_mq_input_embeddings)
# else
#   COMMON_ARGS+=(--freeze_mq_input_embeddings)
# fi

# if [[ "${REQUIRE_MQ_EMBEDDING_TRAIN}" == "1" && "${TRAIN_MQ_INPUT_EMBEDDINGS}" != "1" ]]; then
#   echo "[FATAL] REQUIRE_MQ_EMBEDDING_TRAIN=1 但 TRAIN_MQ_INPUT_EMBEDDINGS=${TRAIN_MQ_INPUT_EMBEDDINGS}"
#   echo "[FATAL] 若要冻结 MQ embedding 做对照实验，请显式设置 REQUIRE_MQ_EMBEDDING_TRAIN=0。"
#   exit 11
# fi

# if [[ "${ENABLE_LOSS_EARLY_STOP}" == "1" ]]; then
#   COMMON_ARGS+=(--enable_loss_early_stop)
# else
#   COMMON_ARGS+=(--disable_loss_early_stop)
# fi

# if [[ "${LOG_EVERY_STEP}" == "1" ]]; then
#   COMMON_ARGS+=(--log_every_step)
# fi
# if [[ "${LOG_CUDA_MEMORY}" == "1" ]]; then
#   COMMON_ARGS+=(--log_cuda_memory)
# fi
# if [[ -n "${RESUME_MQ_ENCODER_PATH:-}" ]]; then
#   COMMON_ARGS+=(--resume_mq_encoder_path "${RESUME_MQ_ENCODER_PATH}")
# fi

# if [[ "${WANDB_ENABLED}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
#   COMMON_ARGS+=(--wandb_enabled --wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}")
#   COMMON_ARGS+=(--wandb_api_key "${WANDB_API_KEY}" --wandb_tags "${WANDB_TAGS}" --wandb_run_name "${WANDB_RUN_NAME}")
#   if [[ -n "${WANDB_ENTITY}" ]]; then
#     COMMON_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
#   fi
#   if [[ "${WANDB_LOG_EVERY_STEP}" == "1" ]]; then
#     COMMON_ARGS+=(--wandb_log_every_step)
#   fi
#   if [[ "${WANDB_LOG_CHECKPOINT}" == "1" ]]; then
#     COMMON_ARGS+=(--wandb_log_checkpoint)
#   fi
# fi

# TRAIN_ENTRY="${SCRIPT_DIR}/train_metaquery_wan_new.py"
# export WAN_BASE_TI2V_MODULE="${WAN_BASE_TI2V_MODULE:-train_metaquery_wan}"
# export WAN_CONNECTOR_FILE="${WAN_CONNECTOR_FILE:-${SCRIPT_DIR}/train_connector_for_wan.py}"

# # Only pass TI2V first-frame CLI flags if the selected base module supports them.
# BASE_TI2V_FILE_CANDIDATE="${WAN_BASE_TI2V_FILE:-${SCRIPT_DIR}/${WAN_BASE_TI2V_MODULE}.py}"
# TI2V_FIRSTFRAME_ARG_SUPPORTED=0
# if [[ -f "${BASE_TI2V_FILE_CANDIDATE}" ]]; then
#   if grep -q -- "--enable_ti2v_first_frame_condition" "${BASE_TI2V_FILE_CANDIDATE}"; then
#     TI2V_FIRSTFRAME_ARG_SUPPORTED=1
#   fi
# fi
# if [[ "${TI2V_FIRSTFRAME_ARG_SUPPORTED}" == "1" ]]; then
#   if [[ "${ENABLE_TI2V_FIRST_FRAME_CONDITION}" == "1" ]]; then
#     COMMON_ARGS+=(--enable_ti2v_first_frame_condition)
#   else
#     COMMON_ARGS+=(--disable_ti2v_first_frame_condition)
#   fi
# else
#   echo "[WARN] base module does not support --enable_ti2v_first_frame_condition, skip this flag."
#   echo "[WARN] checked_file=${BASE_TI2V_FILE_CANDIDATE}"
# fi

# echo "[LAUNCH][I2V-WANCOND] steps=${NUM_TRAIN_STEPS} kept=${KEPT_COUNT}"
# echo "[LAUNCH][I2V-WANCOND] run_output_dir=${RUN_OUTPUT_DIR}"
# echo "[LAUNCH][I2V-WANCOND] WAN first-frame condition=$( [[ "${ENABLE_TI2V_FIRST_FRAME_CONDITION}" == "1" ]] && echo ENABLED || echo DISABLED ) (TI2V latent first-slot, no y concat)"
# echo "[LAUNCH][I2V-WANCOND] video_conditioning_mode=${TRAIN_VIDEO_CONDITIONING_MODE} ref_frames=${TRAIN_ANIMATE_REF_FRAMES} temporal_frames=${TRAIN_ANIMATE_TEMPORAL_FRAMES} conditional_frames=${TRAIN_ANIMATE_CONDITIONAL_FRAMES}"
# echo "[LAUNCH][I2V-WANCOND] preserve_timestep_zero=${TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO} drop_prefix_loss=${TRAIN_ANIMATE_DROP_PREFIX_LOSS}"
# echo "[LAUNCH][I2V-WANCOND] ref_anchor_mode=${TRAIN_REF_ANCHOR_MODE} alpha0=${TRAIN_REF_ANCHOR_ALPHA0} warmup_ratio=${TRAIN_REF_ANCHOR_WARMUP_RATIO}"
# echo "[LAUNCH][I2V-WANCOND] MQ image condition default null_image_prob=${NULL_IMAGE_PROB}"
# echo "[LAUNCH][I2V-WANCOND] WAN context mode=$( [[ "${WAN_CONTEXT_MQ_ONLY}" == "1" ]] && echo mq_only || echo mq_plus_t5 )"
# echo "[LAUNCH][I2V-WANCOND] grad_accum=${GRADIENT_ACCUMULATION_STEPS} lr=${LEARNING_RATE} batch_size=${BATCH_SIZE}"
# echo "[LAUNCH][I2V-WANCOND] num_metaqueries=${NUM_METAQUERIES} connector_layers=${CONNECTOR_LAYERS}"
# echo "[LAUNCH][I2V-WANCOND] lr_scheduler=${LR_SCHEDULER_TYPE} warmup=${WARMUP_STEPS} cooldown=${COOLDOWN_STEPS} lr_min_ratio=${LR_MIN_RATIO}"
# echo "[LAUNCH][I2V-WANCOND] early_stop enabled=${ENABLE_LOSS_EARLY_STOP} min_step=${LOSS_EARLY_STOP_MIN_STEP} threshold=${LOSS_EARLY_STOP_THRESHOLD}"
# echo "[LAUNCH][I2V-WANCOND] wan_train_mode=${WAN_TRAIN_MODE} auto_full_mem_gb=${WAN_AUTO_FULL_MEM_GB} wan_lr_ratio=${WAN_LR_RATIO} wan_cond_name_pattern=${WAN_COND_NAME_PATTERN:-<default>}"
# echo "[LAUNCH][I2V-WANCOND] train_mq_input_embeddings=${TRAIN_MQ_INPUT_EMBEDDINGS} dit_condition_mode=${DIT_CONDITION_MODE}"

# if [[ "${ENABLE_DISTRIBUTED_RUNTIME}" == "1" || "${NPROC_PER_NODE}" != "1" ]]; then
#   TRAIN_LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}")
#   LAUNCH_MODE="torchrun+distributed"
# else
#   TRAIN_LAUNCHER=("${PYTHON_BIN}")
#   LAUNCH_MODE="python-single-process"
# fi
# echo "[LAUNCH][I2V-WANCOND] launch_mode=${LAUNCH_MODE} nproc_per_node=${NPROC_PER_NODE} gpus_per_process=${GPUS_PER_PROCESS} enable_distributed_runtime=${ENABLE_DISTRIBUTED_RUNTIME}"

# "${TRAIN_LAUNCHER[@]}" "${TRAIN_ENTRY}" "${COMMON_ARGS[@]}"

# RUN_OUTPUT_DIR="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${RUN_OUTPUT_DIR}")"
# echo "[I2V-WANCOND][VERIFY] before checkpoint: ${RUN_OUTPUT_DIR}/checkpoint-before-training"
# echo "[I2V-WANCOND][VERIFY] final checkpoint : ${RUN_OUTPUT_DIR}/checkpoint-final"
# echo "[I2V-WANCOND][VERIFY] chain manifest   : ${RUN_OUTPUT_DIR}/training_chain_manifest.json"
# echo "[I2V-WANCOND][VERIFY] pair report      : ${PAIR_REPORT_JSON}"
# echo "[I2V-WANCOND][VERIFY] audit cmd:"
# echo "python ${SCRIPT_DIR}/verify_metaquery_chain.py \\"
# echo "  --before_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-before-training \\"
# echo "  --after_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-final \\"
# echo "  --output_json ${RUN_OUTPUT_DIR}/chain_audit_report.json --strict"




# 这个是mq的rms适配t5
set -euo pipefail

# Overfit launcher (TI2V path) with explicit first-frame conditioning on BOTH:
# 1) MQ side (qwen3-vl MetaQuery image input)
# 2) WAN side (TI2V latent first-slot condition, no I2V-style y concat)
#
# This is the "WAN also receives first-frame condition" variant.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

export http_proxy=10.130.130.6:56830
export https_proxy=10.130.130.6:56830
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,expandable_segments:True}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
GPUS_PER_PROCESS="${GPUS_PER_PROCESS:-1}"
ENABLE_DISTRIBUTED_RUNTIME="${ENABLE_DISTRIBUTED_RUNTIME:-0}"  # 单卡默认不初始化 torch.distributed / NCCL
SEED="${SEED:-42}"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-890}"
SAVE_STEPS="${SAVE_STEPS:-680}"
LOG_STEPS="${LOG_STEPS:-1}"
LOG_EVERY_STEP="${LOG_EVERY_STEP:-1}"
LOG_CUDA_MEMORY="${LOG_CUDA_MEMORY:-1}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
COOLDOWN_STEPS="${COOLDOWN_STEPS:-${WARMUP_STEPS}}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-warmup_hold_cooldown}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.01}"

# 可选早停：step >= min_step 且 train/loss_denoise < threshold 时提前停止
ENABLE_LOSS_EARLY_STOP="${ENABLE_LOSS_EARLY_STOP:-1}"
LOSS_EARLY_STOP_MIN_STEP="${LOSS_EARLY_STOP_MIN_STEP:-860}"
LOSS_EARLY_STOP_THRESHOLD="${LOSS_EARLY_STOP_THRESHOLD:-0.25}"

# 训练可训练参数范围相关
DIT_CONDITION_MODE="${DIT_CONDITION_MODE:-mq_only}"
TRAIN_MQ_INPUT_EMBEDDINGS="${TRAIN_MQ_INPUT_EMBEDDINGS:-1}"  # 1=训练 MQ 新增 token embedding
REQUIRE_MQ_EMBEDDING_TRAIN="${REQUIRE_MQ_EMBEDDING_TRAIN:-1}"
MQ_CONNECTOR_NORM_INIT_SCALE="${MQ_CONNECTOR_NORM_INIT_SCALE:-1.0}"
MQ_NORM_PROBE_WITH_T5="${MQ_NORM_PROBE_WITH_T5:-1}"
MQ_NORM_PROBE_EVERY_N_STEPS="${MQ_NORM_PROBE_EVERY_N_STEPS:-20}"
MQ_NORM_WARN_RATIO_LOW="${MQ_NORM_WARN_RATIO_LOW:-0.25}"
MQ_NORM_WARN_RATIO_HIGH="${MQ_NORM_WARN_RATIO_HIGH:-4.0}"
MQ_NORM_MATCH_T5="${MQ_NORM_MATCH_T5:-1}"
MQ_NORM_MATCH_CLIP_MIN="${MQ_NORM_MATCH_CLIP_MIN:-0.03}"
MQ_NORM_MATCH_CLIP_MAX="${MQ_NORM_MATCH_CLIP_MAX:-4.0}"
WAN_TRAIN_MODE="${WAN_TRAIN_MODE:-cond_only}"                     # auto | full | cond_only | frozen
WAN_AUTO_FULL_MEM_GB="${WAN_AUTO_FULL_MEM_GB:-120}"
WAN_LR_RATIO="${WAN_LR_RATIO:-1.0}"
WAN_COND_NAME_PATTERN="${WAN_COND_NAME_PATTERN:-}"           # cond_only 关键字覆盖（逗号分隔）

NUM_METAQUERIES="${NUM_METAQUERIES:-256}"
CONNECTOR_LAYERS="${CONNECTOR_LAYERS:-24}"

# Keep defaults conservative for overfit stability.
NULL_IMAGE_PROB="${NULL_IMAGE_PROB:-0.1}"
NULL_CAPTION_PROB="${NULL_CAPTION_PROB:-0.1}"

I2V_FRAME_NUM="${I2V_FRAME_NUM:-49}"            # 4n+1
I2V_MAX_AREA="${I2V_MAX_AREA:-262144}"
MIN_DURATION_SEC="${MIN_DURATION_SEC:-0.3}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"

OPENVID_VIDEO_ROOT="${OPENVID_VIDEO_ROOT:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/video}"
OPENVID_CSV_PATH="${OPENVID_CSV_PATH:-/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M/data/train/OpenVid-1M.csv}"

QWEN_MODEL="${QWEN_MODEL:-/home/liuzhirui/model/Qwen3-VL-main/Qwen3-VL-2B-Thinking}"
WAN_TI2V_CKPT="${WAN_TI2V_CKPT:-${WAN_I2V_CKPT:-/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B}}"

# Tokenizer: remote-first loading from Hugging Face.
CAPTION_TOKENIZER_PATH="${CAPTION_TOKENIZER_PATH:-google/umt5-xxl}"
export WAN_TOKENIZER_LOCAL_ONLY=0
export TOKENIZER_LOCAL_ONLY=0
unset TRANSFORMERS_OFFLINE HF_HUB_OFFLINE

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/liuzhirui/model/Wan2.2/checkpoint/metaquery_overfit10_ti2v_firstframe_wan_cond}"
OVERFIT_ROOT="${OVERFIT_ROOT:-/home/liuzhirui/model/Wan2.2/tmp/metaquery_overfit10_ti2v_firstframe_wan_cond}"
OVERFIT_ROOT="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${OVERFIT_ROOT}")"
mkdir -p "${OVERFIT_ROOT}"

VIDEO_LIST_FILE="${VIDEO_LIST_FILE:-${OVERFIT_ROOT}/video_list_overfit10.txt}"
MINI_VIDEO_ROOT="${OVERFIT_ROOT}/videos"
MINI_CSV_PATH="${OVERFIT_ROOT}/openvid_overfit10.csv"
PAIR_REPORT_JSON="${OVERFIT_ROOT}/pair_report.json"

# Keep the same overfit list style as your original overfit10 script.
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
        print(f"[I2V-WANCOND][DROP] video={src} reason=video_not_found")
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
        print(f"[I2V-WANCOND][DROP] video={src} reason=caption_not_found")
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
    print(f"[I2V-WANCOND][KEEP] video={target_name} src={src}")

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
print(f"[I2V-WANCOND] mini_csv={mini_csv_path.resolve()} kept={len(kept)} dropped={len(dropped)}")
print(f"[I2V-WANCOND] pair_report={pair_report_json.resolve()}")
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

echo "[I2V-WANCOND] final_kept_count=${KEPT_COUNT}"
echo "[I2V-WANCOND] mini_video_root=${MINI_VIDEO_ROOT}"
echo "[I2V-WANCOND] mini_csv=${MINI_CSV_PATH}"

# Dataset debug prints during preclean.
export WAN_LOCAL_OVERFIT_DEBUG=1
export WAN_LOCAL_MISSING_CAPTION_PRINT_MAX=1000
export WAN_DATA_PRECLEAN=1

METRICS_JSONL_PATH="${METRICS_JSONL_PATH:-${OUTPUT_ROOT}/logs/i2v_firstframe_wancond_overfit10_steps${NUM_TRAIN_STEPS}_$(date +%Y%m%d-%H%M%S).jsonl}"
METRICS_JSONL_PATH="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${METRICS_JSONL_PATH}")"

WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-wan-metaquery-overfit10-ti2v-firstframe-wancond}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-overfit10-ti2v-firstframe-wancond-steps${NUM_TRAIN_STEPS}-$(date +%Y%m%d-%H%M%S)}"
WANDB_TAGS="${WANDB_TAGS:-overfit,firstframe,ti2v,mq,wan-condition}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_LOG_EVERY_STEP="${WANDB_LOG_EVERY_STEP:-1}"
WANDB_LOG_CHECKPOINT="${WANDB_LOG_CHECKPOINT:-1}"
WAN_CONTEXT_MQ_ONLY="${WAN_CONTEXT_MQ_ONLY:-1}"  # 1: WAN context仅保留MQ特征(不拼接T5)
ENABLE_TI2V_FIRST_FRAME_CONDITION="${ENABLE_TI2V_FIRST_FRAME_CONDITION:-1}"  # 1: ref_image 作为 TI2V 首个 latent slot 条件
TRAIN_VIDEO_CONDITIONING_MODE="${TRAIN_VIDEO_CONDITIONING_MODE:-legacy_t2v}"  # legacy_t2v | wan_animate_slot
TRAIN_ANIMATE_REF_FRAMES="${TRAIN_ANIMATE_REF_FRAMES:-1}"
TRAIN_ANIMATE_TEMPORAL_FRAMES="${TRAIN_ANIMATE_TEMPORAL_FRAMES:-0}"
TRAIN_ANIMATE_CONDITIONAL_FRAMES="${TRAIN_ANIMATE_CONDITIONAL_FRAMES:-0}"
TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO="${TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO:-1}"
TRAIN_ANIMATE_DROP_PREFIX_LOSS="${TRAIN_ANIMATE_DROP_PREFIX_LOSS:-1}"
TRAIN_REF_ANCHOR_MODE="${TRAIN_REF_ANCHOR_MODE:-animate_like}"   # none | animate_like | mixed50
TRAIN_REF_ANCHOR_ALPHA0="${TRAIN_REF_ANCHOR_ALPHA0:-1.0}"
TRAIN_REF_ANCHOR_WARMUP_RATIO="${TRAIN_REF_ANCHOR_WARMUP_RATIO:-1.0}"

RUN_OUTPUT_DIR="${OUTPUT_ROOT}/ti2v_firstframe_wancond_overfit10_steps${NUM_TRAIN_STEPS}_nummq${NUM_METAQUERIES}_nullimg${NULL_IMAGE_PROB}_nullcap${NULL_CAPTION_PROB}"

if [[ "${WAN_TRAIN_MODE}" != "frozen" && "${NPROC_PER_NODE}" != "1" ]]; then
  echo "[WARN] WAN 可训练模式(${WAN_TRAIN_MODE}) 当前建议单进程训练；自动将 NPROC_PER_NODE: ${NPROC_PER_NODE} -> 1"
  NPROC_PER_NODE="1"
  GPUS_PER_PROCESS="1"
fi

COMMON_ARGS=(
  --auto_device_map
  --gpus_per_process "${GPUS_PER_PROCESS}"
  --hf_stage stage1
  --hf_no_streaming
  --hf_no_subset_cache
  --t5_cpu
  --mq_gradient_checkpointing
  --aggressive_empty_cache
  --wan_checkpoint_dir "${WAN_TI2V_CKPT}"
  --output_dir "${RUN_OUTPUT_DIR}"
  --qwen3vl_model_id "${QWEN_MODEL}"
  --caption_tokenizer_path "${CAPTION_TOKENIZER_PATH}"
  --dit_condition_mode "${DIT_CONDITION_MODE}"
  --local_openvid_video_root "${MINI_VIDEO_ROOT}"
  --local_openvid_csv_path "${MINI_CSV_PATH}"
  --local_openvid_limit "${KEPT_COUNT}"
  --num_metaqueries "${NUM_METAQUERIES}"
  --connector_num_hidden_layers "${CONNECTOR_LAYERS}"
  --null_image_prob "${NULL_IMAGE_PROB}"
  --null_caption_prob "${NULL_CAPTION_PROB}"
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
  --frame_num "${I2V_FRAME_NUM}"
  --max_area "${I2V_MAX_AREA}"
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

if [[ "${ENABLE_DISTRIBUTED_RUNTIME}" == "1" || "${NPROC_PER_NODE}" != "1" ]]; then
  COMMON_ARGS+=(--distributed)
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

if [[ "${TRAIN_MQ_INPUT_EMBEDDINGS}" == "1" ]]; then
  COMMON_ARGS+=(--train_mq_input_embeddings)
else
  COMMON_ARGS+=(--freeze_mq_input_embeddings)
fi
COMMON_ARGS+=(--mq_connector_norm_init_scale "${MQ_CONNECTOR_NORM_INIT_SCALE}")
if [[ "${MQ_NORM_PROBE_WITH_T5}" == "1" ]]; then
  COMMON_ARGS+=(--mq_norm_probe_with_t5)
else
  COMMON_ARGS+=(--disable_mq_norm_probe_with_t5)
fi
COMMON_ARGS+=(--mq_norm_probe_every_n_steps "${MQ_NORM_PROBE_EVERY_N_STEPS}")
COMMON_ARGS+=(--mq_norm_warn_ratio_low "${MQ_NORM_WARN_RATIO_LOW}")
COMMON_ARGS+=(--mq_norm_warn_ratio_high "${MQ_NORM_WARN_RATIO_HIGH}")
COMMON_ARGS+=(--mq_norm_match_clip_min "${MQ_NORM_MATCH_CLIP_MIN}")
COMMON_ARGS+=(--mq_norm_match_clip_max "${MQ_NORM_MATCH_CLIP_MAX}")
if [[ "${MQ_NORM_MATCH_T5}" == "1" ]]; then
  COMMON_ARGS+=(--mq_norm_match_t5)
else
  COMMON_ARGS+=(--disable_mq_norm_match_t5)
fi

if [[ "${REQUIRE_MQ_EMBEDDING_TRAIN}" == "1" && "${TRAIN_MQ_INPUT_EMBEDDINGS}" != "1" ]]; then
  echo "[FATAL] REQUIRE_MQ_EMBEDDING_TRAIN=1 但 TRAIN_MQ_INPUT_EMBEDDINGS=${TRAIN_MQ_INPUT_EMBEDDINGS}"
  echo "[FATAL] 若要冻结 MQ embedding 做对照实验，请显式设置 REQUIRE_MQ_EMBEDDING_TRAIN=0。"
  exit 11
fi

if [[ "${ENABLE_LOSS_EARLY_STOP}" == "1" ]]; then
  COMMON_ARGS+=(--enable_loss_early_stop)
else
  COMMON_ARGS+=(--disable_loss_early_stop)
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
  COMMON_ARGS+=(--wandb_enabled --wandb_project "${WANDB_PROJECT}" --wandb_mode "${WANDB_MODE}")
  COMMON_ARGS+=(--wandb_api_key "${WANDB_API_KEY}" --wandb_tags "${WANDB_TAGS}" --wandb_run_name "${WANDB_RUN_NAME}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    COMMON_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  if [[ "${WANDB_LOG_EVERY_STEP}" == "1" ]]; then
    COMMON_ARGS+=(--wandb_log_every_step)
  fi
  if [[ "${WANDB_LOG_CHECKPOINT}" == "1" ]]; then
    COMMON_ARGS+=(--wandb_log_checkpoint)
  fi
fi

TRAIN_ENTRY="${SCRIPT_DIR}/train_metaquery_wan_new.py"
export WAN_BASE_TI2V_MODULE="${WAN_BASE_TI2V_MODULE:-train_metaquery_wan}"
export WAN_CONNECTOR_FILE="${WAN_CONNECTOR_FILE:-${SCRIPT_DIR}/train_connector_for_wan.py}"

# Only pass TI2V first-frame CLI flags if the selected base module supports them.
BASE_TI2V_FILE_CANDIDATE="${WAN_BASE_TI2V_FILE:-${SCRIPT_DIR}/${WAN_BASE_TI2V_MODULE}.py}"
TI2V_FIRSTFRAME_ARG_SUPPORTED=0
if [[ -f "${BASE_TI2V_FILE_CANDIDATE}" ]]; then
  if grep -q -- "--enable_ti2v_first_frame_condition" "${BASE_TI2V_FILE_CANDIDATE}"; then
    TI2V_FIRSTFRAME_ARG_SUPPORTED=1
  fi
fi
if [[ "${TI2V_FIRSTFRAME_ARG_SUPPORTED}" == "1" ]]; then
  if [[ "${ENABLE_TI2V_FIRST_FRAME_CONDITION}" == "1" ]]; then
    COMMON_ARGS+=(--enable_ti2v_first_frame_condition)
  else
    COMMON_ARGS+=(--disable_ti2v_first_frame_condition)
  fi
else
  echo "[WARN] base module does not support --enable_ti2v_first_frame_condition, skip this flag."
  echo "[WARN] checked_file=${BASE_TI2V_FILE_CANDIDATE}"
fi

echo "[LAUNCH][I2V-WANCOND] steps=${NUM_TRAIN_STEPS} kept=${KEPT_COUNT}"
echo "[LAUNCH][I2V-WANCOND] run_output_dir=${RUN_OUTPUT_DIR}"
echo "[LAUNCH][I2V-WANCOND] WAN first-frame condition=$( [[ "${ENABLE_TI2V_FIRST_FRAME_CONDITION}" == "1" ]] && echo ENABLED || echo DISABLED ) (TI2V latent first-slot, no y concat)"
echo "[LAUNCH][I2V-WANCOND] video_conditioning_mode=${TRAIN_VIDEO_CONDITIONING_MODE} ref_frames=${TRAIN_ANIMATE_REF_FRAMES} temporal_frames=${TRAIN_ANIMATE_TEMPORAL_FRAMES} conditional_frames=${TRAIN_ANIMATE_CONDITIONAL_FRAMES}"
echo "[LAUNCH][I2V-WANCOND] preserve_timestep_zero=${TRAIN_ANIMATE_PRESERVE_TIMESTEP_ZERO} drop_prefix_loss=${TRAIN_ANIMATE_DROP_PREFIX_LOSS}"
echo "[LAUNCH][I2V-WANCOND] ref_anchor_mode=${TRAIN_REF_ANCHOR_MODE} alpha0=${TRAIN_REF_ANCHOR_ALPHA0} warmup_ratio=${TRAIN_REF_ANCHOR_WARMUP_RATIO}"
echo "[LAUNCH][I2V-WANCOND] MQ image condition default null_image_prob=${NULL_IMAGE_PROB}"
echo "[LAUNCH][I2V-WANCOND] caption_tokenizer_path=${CAPTION_TOKENIZER_PATH} WAN_TOKENIZER_LOCAL_ONLY=${WAN_TOKENIZER_LOCAL_ONLY}"
echo "[LAUNCH][I2V-WANCOND] mq_norm_probe=${MQ_NORM_PROBE_WITH_T5} every=${MQ_NORM_PROBE_EVERY_N_STEPS} warn=[${MQ_NORM_WARN_RATIO_LOW},${MQ_NORM_WARN_RATIO_HIGH}] match=${MQ_NORM_MATCH_T5} clip=[${MQ_NORM_MATCH_CLIP_MIN},${MQ_NORM_MATCH_CLIP_MAX}] connector_norm_init=${MQ_CONNECTOR_NORM_INIT_SCALE}"
echo "[LAUNCH][I2V-WANCOND] WAN context mode=$( [[ "${WAN_CONTEXT_MQ_ONLY}" == "1" ]] && echo mq_only || echo mq_plus_t5 )"
echo "[LAUNCH][I2V-WANCOND] grad_accum=${GRADIENT_ACCUMULATION_STEPS} lr=${LEARNING_RATE} batch_size=${BATCH_SIZE}"
echo "[LAUNCH][I2V-WANCOND] num_metaqueries=${NUM_METAQUERIES} connector_layers=${CONNECTOR_LAYERS}"
echo "[LAUNCH][I2V-WANCOND] lr_scheduler=${LR_SCHEDULER_TYPE} warmup=${WARMUP_STEPS} cooldown=${COOLDOWN_STEPS} lr_min_ratio=${LR_MIN_RATIO}"
echo "[LAUNCH][I2V-WANCOND] early_stop enabled=${ENABLE_LOSS_EARLY_STOP} min_step=${LOSS_EARLY_STOP_MIN_STEP} threshold=${LOSS_EARLY_STOP_THRESHOLD}"
echo "[LAUNCH][I2V-WANCOND] wan_train_mode=${WAN_TRAIN_MODE} auto_full_mem_gb=${WAN_AUTO_FULL_MEM_GB} wan_lr_ratio=${WAN_LR_RATIO} wan_cond_name_pattern=${WAN_COND_NAME_PATTERN:-<default>}"
echo "[LAUNCH][I2V-WANCOND] train_mq_input_embeddings=${TRAIN_MQ_INPUT_EMBEDDINGS} dit_condition_mode=${DIT_CONDITION_MODE}"

if [[ "${ENABLE_DISTRIBUTED_RUNTIME}" == "1" || "${NPROC_PER_NODE}" != "1" ]]; then
  TRAIN_LAUNCHER=(torchrun --nproc_per_node "${NPROC_PER_NODE}")
  LAUNCH_MODE="torchrun+distributed"
else
  TRAIN_LAUNCHER=("${PYTHON_BIN}")
  LAUNCH_MODE="python-single-process"
fi
echo "[LAUNCH][I2V-WANCOND] launch_mode=${LAUNCH_MODE} nproc_per_node=${NPROC_PER_NODE} gpus_per_process=${GPUS_PER_PROCESS} enable_distributed_runtime=${ENABLE_DISTRIBUTED_RUNTIME}"

"${TRAIN_LAUNCHER[@]}" "${TRAIN_ENTRY}" "${COMMON_ARGS[@]}"

RUN_OUTPUT_DIR="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${RUN_OUTPUT_DIR}")"
echo "[I2V-WANCOND][VERIFY] before checkpoint: ${RUN_OUTPUT_DIR}/checkpoint-before-training"
echo "[I2V-WANCOND][VERIFY] final checkpoint : ${RUN_OUTPUT_DIR}/checkpoint-final"
echo "[I2V-WANCOND][VERIFY] chain manifest   : ${RUN_OUTPUT_DIR}/training_chain_manifest.json"
echo "[I2V-WANCOND][VERIFY] pair report      : ${PAIR_REPORT_JSON}"
echo "[I2V-WANCOND][VERIFY] audit cmd:"
echo "python ${SCRIPT_DIR}/verify_metaquery_chain.py \\"
echo "  --before_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-before-training \\"
echo "  --after_checkpoint ${RUN_OUTPUT_DIR}/checkpoint-final \\"
echo "  --output_json ${RUN_OUTPUT_DIR}/chain_audit_report.json --strict"
