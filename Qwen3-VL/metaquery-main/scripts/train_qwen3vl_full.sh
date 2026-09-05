#!/bin/bash
eval "$(conda shell.bash hook)"
conda env list
conda activate /home/liuzhirui/miniconda3/envs/moviestory
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://10.39.23.15:808
export https_proxy=http://10.39.23.15:808
# export CUDA_VISIBLE_DEVICES=6
work_dir=/home/liuzhirui

# =============================================================================
# MetaQuery + Qwen3-VL 完整两阶段训练脚本
# =============================================================================
#
# 【base_dir 说明】
#   base_dir 是训练流程的 "工作根目录"，所有数据和产出都以它为根：
#
#     base_dir/
#     ├── .cache/              ← 数据缓存 (HuggingFace datasets 自动下载到此)
#     │   ├── pixparse___cc12m-wds/            ← Stage 1 CC12M 数据集
#     │   └── xcpan___MetaQuery_Instruct_2.4M/ ← Stage 2 指令微调数据集
#     ├── output/              ← checkpoint 输出
#     │   ├── qwen3vl2b_t2i/                   ← Stage 1 checkpoint
#     │   │   ├── checkpoint-1000/
#     │   │   ├── checkpoint-2000/
#     │   │   └── ...
#     │   └── qwen3vl2b_inst/                  ← Stage 2 checkpoint（最终可用的）
#     │       ├── checkpoint-500/
#     │       └── ...
#     └── logs/                ← TensorBoard / WandB 日志
#
#   代码中 trainer_utils.py 的 get_full_dirs() 函数会自动拼接：
#     output_dir  = base_dir + "/output"
#     data_dir    = base_dir + "/.cache"
#     logging_dir = base_dir + "/logs"
#
#   因此你只需选一个磁盘空间充足的目录作为 base_dir 即可。
#   CC12M 数据集约 ~300GB，MetaQuery-Instruct-2.4M 约 ~200GB，checkpoint 每个 ~10-30GB。
#   建议准备至少 **800GB** 可用空间。
#
# 【使用方式】
#   一键跑完两阶段：
#     bash scripts/train_metaquery_full.sh
#
#   只跑 Stage 1：
#     bash scripts/train_metaquery_full.sh --stage 1
#
#   只跑 Stage 2（需要 Stage 1 已完成）：
#     bash scripts/train_metaquery_full.sh --stage 2
#
#   自定义 base_dir：
#     bash scripts/train_metaquery_full.sh --base-dir /data/metaquery
#
#   自定义 GPU 数量：
#     bash scripts/train_metaquery_full.sh --gpus 4
#
#   自定义模型规模 (2b / 4b / 8b)：
#     bash scripts/train_metaquery_full.sh --model-size 8b
#
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 默认配置（可通过命令行参数覆盖）
# ─────────────────────────────────────────────────────────────────────────────

# base_dir: 训练工作根目录，所有数据/checkpoint/日志都在这里
BASE_DIR="/home/liuzhirui/model/Qwen3-VL-main/metaquery-main/checkpoints"

# GPU 数量
NUM_GPUS=4

# 模型规模: 2b / 4b / 8b
MODEL_SIZE="4b"

# 训练阶段: "all" / "1" / "2"
STAGE="all"

# Stage 2 是否从 Stage 1 checkpoint 恢复（默认 yes）
RESUME_FROM_STAGE1="yes"

# 手动指定 Stage 2 恢复的 checkpoint 路径（留空则自动查找 Stage 1 最新 checkpoint）
STAGE2_RESUME_CKPT=""

# OMP 线程数（影响数据下载和预处理并行度）
OMP_THREADS=12

# 是否禁用 wandb（设置为 yes 则使用 tensorboard）
DISABLE_WANDB="no"

# ─────────────────────────────────────────────────────────────────────────────
# 解析命令行参数
# ─────────────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --resume-ckpt)
            STAGE2_RESUME_CKPT="$2"
            shift 2
            ;;
        --no-wandb)
            DISABLE_WANDB="yes"
            shift
            ;;
        --omp-threads)
            OMP_THREADS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --base-dir DIR       训练工作根目录 (数据+checkpoint+日志)"
            echo "  --gpus N             GPU 数量 (default: 8)"
            echo "  --model-size SIZE    模型规模: 2b, 4b, 8b (default: 2b)"
            echo "  --stage STAGE        训练阶段: all, 1, 2 (default: all)"
            echo "  --resume-ckpt PATH   Stage 2 恢复的 checkpoint 路径"
            echo "  --no-wandb           禁用 wandb, 改用 tensorboard"
            echo "  --omp-threads N      OMP_NUM_THREADS (default: 12)"
            echo "  -h, --help           显示帮助"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# 根据模型规模确定配置文件和 run_name
# ─────────────────────────────────────────────────────────────────────────────

case ${MODEL_SIZE} in
    2b)
        STAGE1_CONFIG="qwen3vl2b_sana.yaml"
        STAGE2_CONFIG="qwen3vl2b_sana_inst.yaml"
        STAGE1_RUN_NAME="qwen3vl2b_t2i"
        STAGE2_RUN_NAME="qwen3vl2b_inst"
        ;;
    4b)
        STAGE1_CONFIG="qwen3vl4b_sana.yaml"
        STAGE2_CONFIG="qwen3vl4b_sana_inst.yaml"
        STAGE1_RUN_NAME="qwen3vl4b_t2i"
        STAGE2_RUN_NAME="qwen3vl4b_inst"
        ;;
    8b)
        STAGE1_CONFIG="qwen3vl8b_sana.yaml"
        STAGE2_CONFIG="qwen3vl8b_sana_inst.yaml"
        STAGE1_RUN_NAME="qwen3vl8b_t2i"
        STAGE2_RUN_NAME="qwen3vl8b_inst"
        ;;
    *)
        echo "错误: 不支持的模型规模 '${MODEL_SIZE}'，请使用 2b / 4b / 8b"
        exit 1
        ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# 环境变量
# ─────────────────────────────────────────────────────────────────────────────

export OMP_NUM_THREADS=${OMP_THREADS}
export TOKENIZERS_PARALLELISM=false
export WANDB__SERVICE_WAIT=300

if [[ "${DISABLE_WANDB}" == "yes" ]]; then
    export WANDB_MODE=disabled
fi

# ─────────────────────────────────────────────────────────────────────────────
# 确保 base_dir 已设置
# ─────────────────────────────────────────────────────────────────────────────

if [[ "${BASE_DIR}" == "" ]]; then
    echo "=========================================================="
    echo "  错误: 请设置 --base-dir 参数!"
    echo ""
    echo "  base_dir 是训练的工作根目录，用于存放："
    echo "    - 数据集缓存（~500GB+）"
    echo "    - 模型 checkpoint（~10-30GB 每个）"
    echo "    - 训练日志"
    echo ""
    echo "  示例:"
    echo "    bash $0 --base-dir /data/metaquery_training"
    echo "=========================================================="
    exit 1
fi

# 创建 base_dir（如果不存在）
mkdir -p "${BASE_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# 打印训练配置
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      MetaQuery + Qwen3-VL 训练                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  模型规模:    Qwen3-VL-${MODEL_SIZE^^}                               ║"
echo "║  训练阶段:    ${STAGE}                                            ║"
echo "║  GPU 数量:    ${NUM_GPUS}                                              ║"
echo "║  base_dir:    ${BASE_DIR}"
echo "║  OMP_THREADS: ${OMP_THREADS}                                            ║"
echo "║  WandB:       $([ "${DISABLE_WANDB}" == "yes" ] && echo "禁用" || echo "启用")                                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  目录结构:                                                    ║"
echo "║    ${BASE_DIR}/"
echo "║    ├── .cache/     ← HuggingFace 数据集缓存"
echo "║    ├── output/     ← checkpoint 输出"
echo "║    └── logs/       ← 训练日志"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 确认磁盘空间
# ─────────────────────────────────────────────────────────────────────────────

# AVAIL_GB=$(df -BG "${BASE_DIR}" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' || echo "unknown")
# if [[ "${AVAIL_GB}" != "unknown" ]] && [[ ${AVAIL_GB} -lt 500 ]]; then
#     echo "⚠️  警告: ${BASE_DIR} 可用空间仅 ${AVAIL_GB}GB，建议至少 800GB"
#     echo "   CC12M 数据约 300GB，MetaQuery-Instruct-2.4M 约 200GB"
#     read -p "   是否继续? [y/N] " -n 1 -r
#     echo
#     if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#         exit 1
#     fi
# fi

# ─────────────────────────────────────────────────────────────────────────────
# 切换到 metaquery-main 代码目录
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METAQUERY_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${METAQUERY_DIR}"
echo "[INFO] 工作目录: $(pwd)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: T2I 预训练 (在 CC12M 上)
# ─────────────────────────────────────────────────────────────────────────────

run_stage1() {
    echo "============================================================"
    echo "  Stage 1: 文本到图像预训练 (CC12M)"
    echo "  Config:   ${STAGE1_CONFIG}"
    echo "  Run Name: ${STAGE1_RUN_NAME}"
    echo "  数据集:   pixparse/cc12m-wds (HuggingFace 自动下载)"
    echo "============================================================"
    echo ""

    # 验证配置文件
    if [[ ! -f "configs/${STAGE1_CONFIG}" ]]; then
        echo "错误: 配置文件 configs/${STAGE1_CONFIG} 不存在!"
        exit 1
    fi

    echo "[Stage 1] 开始训练..."
    echo "[Stage 1] 数据集将自动下载到: ${BASE_DIR}/.cache/"
    echo "[Stage 1] Checkpoint 将保存到: ${BASE_DIR}/output/${STAGE1_RUN_NAME}/"
    echo ""

    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        /home/liuzhirui/model/Qwen3-VL-main/metaquery-main/train.py \
        --run_name "${STAGE1_RUN_NAME}" \
        --config_file "${STAGE1_CONFIG}" \
        --base_dir "${BASE_DIR}"

    echo ""
    echo "[Stage 1] ✅ 训练完成!"
    echo "[Stage 1] Checkpoint 位于: ${BASE_DIR}/output/${STAGE1_RUN_NAME}/"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: 指令微调 (在 MetaQuery-Instruct-2.4M 上)
# ─────────────────────────────────────────────────────────────────────────────

run_stage2() {
    echo "============================================================"
    echo "  Stage 2: 指令微调 (MetaQuery-Instruct-2.4M)"
    echo "  Config:   ${STAGE2_CONFIG}"
    echo "  Run Name: ${STAGE2_RUN_NAME}"
    echo "  数据集:   xcpan/MetaQuery_Instruct_2.4M_512res"
    echo "============================================================"
    echo ""

    # 验证配置文件
    if [[ ! -f "configs/${STAGE2_CONFIG}" ]]; then
        echo "错误: 配置文件 configs/${STAGE2_CONFIG} 不存在!"
        exit 1
    fi

    # 构建恢复参数
    RESUME_ARG=""
    if [[ -n "${STAGE2_RESUME_CKPT}" ]]; then
        # 用户手动指定了 checkpoint
        RESUME_ARG="--resume_from_checkpoint ${STAGE2_RESUME_CKPT}"
        echo "[Stage 2] 从指定 checkpoint 恢复: ${STAGE2_RESUME_CKPT}"
    elif [[ "${RESUME_FROM_STAGE1}" == "yes" ]]; then
        # 自动查找 Stage 1 最新 checkpoint
        STAGE1_OUTPUT="${BASE_DIR}/output/${STAGE1_RUN_NAME}"
        if [[ -d "${STAGE1_OUTPUT}" ]]; then
            # 查找最新的 checkpoint-XXXXX 目录
            LATEST_CKPT=$(ls -d "${STAGE1_OUTPUT}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
            if [[ -n "${LATEST_CKPT}" ]]; then
                RESUME_ARG="--resume_from_checkpoint ${LATEST_CKPT}"
                echo "[Stage 2] 自动从 Stage 1 checkpoint 恢复: ${LATEST_CKPT}"
            else
                echo "[Stage 2] 警告: Stage 1 输出目录存在但未找到 checkpoint，将从头训练"
            fi
        else
            echo "[Stage 2] 警告: Stage 1 输出目录 ${STAGE1_OUTPUT} 不存在"
            echo "[Stage 2] 将从头开始训练（没有 Stage 1 预训练权重）"
        fi
    fi

    echo "[Stage 2] 开始训练..."
    echo "[Stage 2] 数据集将自动下载到: ${BASE_DIR}/.cache/"
    echo "[Stage 2] Checkpoint 将保存到: ${BASE_DIR}/output/${STAGE2_RUN_NAME}/"
    echo ""

    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        /home/liuzhirui/model/Qwen3-VL-main/metaquery-main/train.py \
        --run_name "${STAGE2_RUN_NAME}" \
        --config_file "${STAGE2_CONFIG}" \
        --base_dir "${BASE_DIR}" \
        ${RESUME_ARG}

    echo ""
    echo "[Stage 2] ✅ 训练完成!"
    echo "[Stage 2] Checkpoint 位于: ${BASE_DIR}/output/${STAGE2_RUN_NAME}/"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# 执行训练
# ─────────────────────────────────────────────────────────────────────────────

START_TIME=$(date +%s)

case ${STAGE} in
    1)
        run_stage1
        ;;
    2)
        run_stage2
        ;;
    all)
        run_stage1
        run_stage2
        ;;
    *)
        echo "错误: 不支持的训练阶段 '${STAGE}'，请使用 all / 1 / 2"
        exit 1
        ;;
esac

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🎉 训练全部完成!                                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  总耗时: ${HOURS}h ${MINUTES}m"
echo "║"
echo "║  接下来你需要将最终 checkpoint 路径设置为 METAQUERY_CKPT:"
echo "║"

if [[ "${STAGE}" == "1" ]]; then
    echo "║  METAQUERY_CKPT = ${BASE_DIR}/output/${STAGE1_RUN_NAME}"
elif [[ "${STAGE}" == "2" ]] || [[ "${STAGE}" == "all" ]]; then
    echo "║  METAQUERY_CKPT = ${BASE_DIR}/output/${STAGE2_RUN_NAME}"
fi

echo "║"
echo "║  然后在 demo_metaquery_i2v.py / demo_metaquery_animate.py"
echo "║  中设置该路径即可运行推理。"
echo "╚══════════════════════════════════════════════════════════════╝"
