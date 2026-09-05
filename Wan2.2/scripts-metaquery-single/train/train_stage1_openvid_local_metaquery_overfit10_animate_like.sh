#!/bin/bash
set -euo pipefail

# Overfit10 专用：TI2V + animate-like 训练入口
# - 兼容基础脚本重命名：train_metaquery_wan_animate_like.py
# - 默认启用混合配方：50% none + 50% animate_like
#
# 用法：
#   bash train_stage1_openvid_local_metaquery_overfit10_animate_like.sh
#   NUM_TRAIN_STEPS=100 TRAIN_REF_ANCHOR_MODE=mixed50 bash train_stage1_openvid_local_metaquery_overfit10_animate_like.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WAN_ROOT="${WAN_ROOT:-${SCRIPT_DIR}}"

# 固定走 ti2v overfit10 流程
export TASK="${TASK:-ti2v}"

# train_metaquery_wan_new.py 会优先加载该基础模块名
export WAN_BASE_TI2V_MODULE="${WAN_BASE_TI2V_MODULE:-train_metaquery_wan_animate_like}"

# animate-like 混合训练默认参数
export TRAIN_REF_ANCHOR_MODE="${TRAIN_REF_ANCHOR_MODE:-mixed50}"
export TRAIN_REF_ANCHOR_ALPHA0="${TRAIN_REF_ANCHOR_ALPHA0:-0.95}"
export TRAIN_REF_ANCHOR_WARMUP_RATIO="${TRAIN_REF_ANCHOR_WARMUP_RATIO:-0.35}"

bash "${SCRIPT_DIR}/train_stage1_openvid_local_metaquery_overfit10_animate_like_base.sh"

