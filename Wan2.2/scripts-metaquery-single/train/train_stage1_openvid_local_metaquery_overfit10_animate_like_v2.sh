#!/bin/bash
set -euo pipefail

# Overfit10 专用：TI2V + Wan-Animate reference-slot 训练入口
#
# 用法：
#   bash train_stage1_openvid_local_metaquery_overfit10_wan_animate_slot.sh
#   TRAIN_ANIMATE_TEMPORAL_FRAMES=5 NUM_TRAIN_STEPS=100 bash train_stage1_openvid_local_metaquery_overfit10_wan_animate_slot.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WAN_ROOT="${WAN_ROOT:-${SCRIPT_DIR}}"
export TASK="${TASK:-ti2v}"

# train_metaquery_wan_new.py 将优先加载该基础脚本
export WAN_BASE_TI2V_MODULE="${WAN_BASE_TI2V_MODULE:-train_metaquery_wan_animate_like_v2}"

# Wan-Animate slot 训练参数
export TRAIN_VIDEO_CONDITIONING_MODE="${TRAIN_VIDEO_CONDITIONING_MODE:-wan_animate_slot}"
export TRAIN_ANIMATE_REF_FRAMES="${TRAIN_ANIMATE_REF_FRAMES:-1}"
export TRAIN_ANIMATE_TEMPORAL_FRAMES="${TRAIN_ANIMATE_TEMPORAL_FRAMES:-1}"
export TRAIN_ANIMATE_CONDITIONAL_FRAMES="${TRAIN_ANIMATE_CONDITIONAL_FRAMES:-0}"
export TRAIN_ANIMATE_DROP_PREFIX_LOSS="${TRAIN_ANIMATE_DROP_PREFIX_LOSS:-1}"
export TRAIN_ANIMATE_PRESERVE_T0="${TRAIN_ANIMATE_PRESERVE_T0:-1}"

# 该模式下关闭 legacy 首帧软锚定
export TRAIN_REF_ANCHOR_MODE="${TRAIN_REF_ANCHOR_MODE:-none}"

BASE_SH="${SCRIPT_DIR}/train_stage1_openvid_local_metaquery_overfit10_animate_like_base_v2.sh"
if [[ ! -f "${BASE_SH}" ]]; then
  echo "[ERROR] base script not found: ${BASE_SH}"
  exit 2
fi

bash "${BASE_SH}"

