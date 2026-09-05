#!/bin/bash
set -euo pipefail

# Determined single-GPU launcher variant:
# - ref image is used by neither MQ encoder nor Wan y-condition

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NO_REF_CONDITION="${NO_REF_CONDITION:-1}"
export MQ_REF_ONLY="${MQ_REF_ONLY:-0}"

"/home/liuzhirui/model/Wan2.2/wan-animate-metaquery-inference-new-determined-single.sh"

