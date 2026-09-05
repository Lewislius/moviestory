#!/bin/bash
set -euo pipefail

# Determined single-GPU launcher variant:
# - ref_image is used by MetaQuery encoder
# - ref_image is NOT passed as Wan y-condition

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MQ_REF_ONLY="${MQ_REF_ONLY:-1}"

"/home/liuzhirui/model/Wan2.2/wan-animate-metaquery-inference-new-determined-single.sh"

