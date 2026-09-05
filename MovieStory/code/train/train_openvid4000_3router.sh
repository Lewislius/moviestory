#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/moviestory

export http_proxy=10.130.130.6:56830
export https_proxy=10.130.130.6:56830
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="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"
export WANDB_PROJECT="moviestory-3router"
OPENVID_ROOT="/run/determined/NAS1/public/Dataset/Metaquery-Wan/OpenVid-1M"
OPENVID_LIMIT=4000
SUBSET_ROOT="${CODE_ROOT}/tmp/openvid_first${OPENVID_LIMIT}"
# batch_size=1 and gradient_accumulation_steps=8: 500 optimizer steps request
# 4000 microbatches (one pass when all records survive pre-cleaning).
NUM_TRAIN_STEPS=500
ROUTER_ENABLED="${ROUTER_ENABLED:-1}"
ROUTER_TOGGLE_ARGS=()
case "${ROUTER_ENABLED,,}" in
  1|true|yes) RUN_VARIANT="three_router" ;;
  0|false|no)
    RUN_VARIANT="baseline"
    ROUTER_TOGGLE_ARGS+=(--disable_3router)
    ;;
  *) echo "Invalid ROUTER_ENABLED=${ROUTER_ENABLED}; use 1 or 0" >&2; exit 2 ;;
esac
OUTPUT_DIR="${CODE_ROOT}/checkpoint/${RUN_VARIANT}_mq256_conn24_strongbind_openvid${OPENVID_LIMIT}_steps${NUM_TRAIN_STEPS}"
source "${CODE_ROOT}/scripts/wandb_env.sh"
configure_wandb_args \
  "${RUN_VARIANT}-openvid${OPENVID_LIMIT}-steps${NUM_TRAIN_STEPS}" \
  "moviestory,${RUN_VARIANT},openvid${OPENVID_LIMIT},strong-bind,cond-only"

python "${CODE_ROOT}/scripts/prepare_openvid_subset.py" \
  --video_root "${OPENVID_ROOT}/video" \
  --csv_path "${OPENVID_ROOT}/data/train/OpenVid-1M.csv" \
  --output_root "${SUBSET_ROOT}" \
  --limit "${OPENVID_LIMIT}"

python "${CODE_ROOT}/train_3router_planner_wan.py" \
  --output_dir "${OUTPUT_DIR}" \
  --local_openvid_video_root "${SUBSET_ROOT}/videos" \
  --local_openvid_csv_path "${SUBSET_ROOT}/openvid_first${OPENVID_LIMIT}.csv" \
  --local_openvid_limit "${OPENVID_LIMIT}" \
  --num_metaqueries 256 \
  --connector_num_hidden_layers 24 \
  --num_train_steps "${NUM_TRAIN_STEPS}" \
  --save_steps 240 \
  --learning_rate 1e-5 \
  --warmup_steps 50 \
  --gradient_accumulation_steps 8 \
  --router_log_steps 1 \
  --router_stale_update_patience 5 \
  "${ROUTER_TOGGLE_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  --frame_num 49 \
  --max_area 262144 \
  --wan_train_mode cond_only \
  --wan_first_frame_strong_bind \
  --enable_ti2v_first_frame_condition \
  --train_video_conditioning_mode wan_animate_slot \
  --train_ref_anchor_mode none \
  --joint_null_prob 0.1 \
  --freeze_mq_input_embeddings \
  --mq_gradient_checkpointing \
  --disable_t5_alignment \
  --disable_wan_func_distill \
  --mq_norm_probe_with_t5 \
  --mq_norm_match_t5 \
  --t5_cpu \
  --null_image_prob 0.1 \
  --null_caption_prob 0.1
