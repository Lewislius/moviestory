#!/bin/bash
# =============================================================================
# MetaQuery + Qwen3-VL 训练入口脚本（适用于 Determined AI）
# =============================================================================
# 流程：
#   1. 单进程预下载所有需要的 HuggingFace 数据集（避免多 GPU 同时下载导致 NCCL 超时）
#   2. 启动 torchrun 多 GPU 分布式训练
# =============================================================================
# eval "$(conda shell.bash hook)"
# conda env list
# conda activate /home/liuzhirui/miniconda3/envs/moviestory
# export HF_ENDPOINT=https://hf-mirror.com
# export http_proxy=http://10.39.23.15:808
# export https_proxy=http://10.39.23.15:808
# Set HF_TOKEN in the environment if authentication is required.
echo $HF_TOKEN
# export CUDA_VISIBLE_DEVICES=6
# work_dir=/home/liuzhirui
PROJECT_DIR="/home/liuzhirui/model/Qwen3-VL-main/metaquery-main"
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR="/home/liuzhirui/model/Qwen3-VL-main/metaquery-main/checkpoints"
NUM_GPUS=4            # 与 YAML 中 slots_per_trial 一致
MODEL_SIZE="2b-small"
STAGE="all"           # "all" / "1" / "2"
USE_SMALL="yes"        # 是否使用 1/100 小数据量模式

# 数据集缓存目录（持久化路径，确保在 bind_mounts 中挂载的目录下）
DATA_CACHE_DIR="${BASE_DIR}/.cache"

# HuggingFace 镜像（国内加速，如不需要可注释掉）
# export HF_ENDPOINT=https://hf-mirror.com

# ─────────────────────────────────────────────────────────────────────────────
# 环境变量
# ─────────────────────────────────────────────────────────────────────────────

export OMP_NUM_THREADS=12
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
# export WANDB__SERVICE_WAIT=300
# export WANDB_API_KEY="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"
# export TRITON_CACHE_DIR="/tmp/triton_cache_metaquery"

# ─────────────────────────────────────────────────────────────────────────────
# 根据模型规模确定配置文件
# ─────────────────────────────────────────────────────────────────────────────

# 如果用户使用 --small 参数，自动拼接为 MODEL_SIZE-small
# 也可以直接设置 MODEL_SIZE="4b-small"
if [ "${USE_SMALL}" = "yes" ]; then
    case "${MODEL_SIZE}" in
        *-small) ;; # 已经是 small 模式，不需要拼接
        *) MODEL_SIZE="${MODEL_SIZE}-small" ;;
    esac
fi

case ${MODEL_SIZE} in
    2b)
        STAGE1_CONFIG="qwen3vl2b_sana.yaml"
        STAGE2_CONFIG="qwen3vl2b_sana_inst.yaml"
        STAGE1_RUN_NAME="qwen3vl2b_t2i"
        STAGE2_RUN_NAME="qwen3vl2b_inst"
        ;;
    2b-small)
        STAGE1_CONFIG="qwen3vl2b_sana_small.yaml"
        STAGE2_CONFIG="qwen3vl2b_sana_inst_small.yaml"
        STAGE1_RUN_NAME="qwen3vl2b_t2i_small"
        STAGE2_RUN_NAME="qwen3vl2b_inst_small"
        ;;
    4b)
        STAGE1_CONFIG="qwen3vl4b_sana.yaml"
        STAGE2_CONFIG="qwen3vl4b_sana_inst.yaml"
        STAGE1_RUN_NAME="qwen3vl4b_t2i"
        STAGE2_RUN_NAME="qwen3vl4b_inst"
        ;;
    4b-small)
        STAGE1_CONFIG="qwen3vl4b_sana_small.yaml"
        STAGE2_CONFIG="qwen3vl4b_sana_inst_small.yaml"
        STAGE1_RUN_NAME="qwen3vl4b_t2i_small"
        STAGE2_RUN_NAME="qwen3vl4b_inst_small"
        ;;
    8b)
        STAGE1_CONFIG="qwen3vl8b_sana.yaml"
        STAGE2_CONFIG="qwen3vl8b_sana_inst.yaml"
        STAGE1_RUN_NAME="qwen3vl8b_t2i"
        STAGE2_RUN_NAME="qwen3vl8b_inst"
        ;;
    *)
        echo "错误: 不支持的模型规模 '${MODEL_SIZE}'"
        echo "支持: 2b / 2b-small / 4b / 4b-small / 8b"
        exit 1
        ;;
esac

# ─────────────────────────────────────────────────────────────────────────────
# 切换到 metaquery-main 代码目录
# ─────────────────────────────────────────────────────────────────────────────

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# METAQUERY_DIR="$(dirname "${SCRIPT_DIR}")"
# cd "${METAQUERY_DIR}"
# echo "[INFO] 工作目录: $(pwd)"

# ─────────────────────────────────────────────────────────────────────────────
# 第一步：单进程预下载数据集（核心！避免 NCCL 超时）
# ─────────────────────────────────────────────────────────────────────────────
# 为什么需要这一步：
#   torchrun 启动后，所有 GPU 进程同时执行代码。如果数据集没有预先下载，
#   Rank 0 会卡在网络下载上（可能耗时数小时），而 Rank 1/2/3 已经准备好，
#   在 NCCL allreduce 同步点等待 Rank 0，等待超过 30 分钟后触发 NCCL timeout。
#
#   解决方案：在 torchrun 之前，用单独的 Python 进程把数据集下载好。
#   下载完成后，torchrun 中的所有进程都能直接从本地缓存加载，不会卡住。
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${DATA_CACHE_DIR}"

echo ""
echo "============================================================"
echo "  [预下载] 检查并下载所需数据集到: ${DATA_CACHE_DIR}"
echo "============================================================"
echo ""

# 根据 STAGE 决定下载哪些数据集
# 小数据量模式下使用 streaming 模式，只下载所需的数据分片，不会下载完整数据集
if [ "${STAGE}" = "1" ] || [ "${STAGE}" = "all" ]; then
    echo "[预下载] Stage 1 需要: pixparse/cc12m-wds"
    # 判断是否为 small 模式
    case "${MODEL_SIZE}" in *-small)
        NUM_SAMPLES=12000
        echo "[预下载] 小数据量模式: streaming 下载前 ${NUM_SAMPLES} 条 (1/1000)"
        ;;
    *)
        NUM_SAMPLES=0
        ;;
    esac
    python -c "
from datasets import load_dataset, Dataset, load_from_disk
import os, sys

cache_dir = '${DATA_CACHE_DIR}'
num_samples = ${NUM_SAMPLES}
dataset_name = 'pixparse/cc12m-wds'

if num_samples > 0:
    # 使用 streaming 模式只下载所需分片
    safe_name = dataset_name.replace('/', '__')
    subset_dir = os.path.join(cache_dir, f'{safe_name}_subset_{num_samples}')
    if os.path.exists(subset_dir):
        ds = load_from_disk(subset_dir)
        print(f'[预下载] ✅ {dataset_name} 子集已缓存: {len(ds)} 条')
    else:
        print(f'[预下载] 使用 streaming 模式下载 {dataset_name} 的前 {num_samples} 条...')
        print(f'[预下载] (只下载所需的数据分片，不会下载完整数据集)')
        sys.stdout.flush()
        stream = load_dataset(dataset_name, split='train', streaming=True)
        count = [0]
        def take_gen():
            for item in stream:
                if count[0] >= num_samples:
                    break
                count[0] += 1
                if count[0] % 10000 == 0:
                    print(f'[预下载]   已下载 {count[0]}/{num_samples} 条...')
                    sys.stdout.flush()
                yield item
        ds = Dataset.from_generator(take_gen)
        os.makedirs(subset_dir, exist_ok=True)
        ds.save_to_disk(subset_dir)
        print(f'[预下载] ✅ {dataset_name} 子集就绪: {len(ds)} 条 (已缓存到 {subset_dir})')
else:
    print(f'[预下载] 正在加载 {dataset_name} (全量)...')
    sys.stdout.flush()
    try:
        ds = load_dataset(dataset_name, cache_dir=cache_dir, split='train')
        print(f'[预下载] ✅ {dataset_name} 就绪，共 {len(ds)} 条数据')
    except Exception as e:
        print(f'[预下载] ❌ {dataset_name} 下载失败: {e}')
        sys.exit(1)
"
    echo ""
fi

if [ "${STAGE}" = "2" ] || [ "${STAGE}" = "all" ]; then
    echo "[预下载] Stage 2 需要: xcpan/MetaQuery_Instruct_2.4M_512res"
    # 判断是否为 small 模式
    case "${MODEL_SIZE}" in *-small)
        NUM_SAMPLES=2400
        echo "[预下载] 小数据量模式: streaming 下载前 ${NUM_SAMPLES} 条 (1/1000)"
        ;;
    *)
        NUM_SAMPLES=0
        ;;
    esac
    python -c "
from datasets import load_dataset, Dataset, load_from_disk
import os, sys

cache_dir = '${DATA_CACHE_DIR}'
num_samples = ${NUM_SAMPLES}
dataset_name = 'xcpan/MetaQuery_Instruct_2.4M_512res'

if num_samples > 0:
    safe_name = dataset_name.replace('/', '__')
    subset_dir = os.path.join(cache_dir, f'{safe_name}_subset_{num_samples}')
    if os.path.exists(subset_dir):
        ds = load_from_disk(subset_dir)
        print(f'[预下载] ✅ {dataset_name} 子集已缓存: {len(ds)} 条')
    else:
        print(f'[预下载] 使用 streaming 模式下载 {dataset_name} 的前 {num_samples} 条...')
        print(f'[预下载] (只下载所需的数据分片，不会下载完整数据集)')
        sys.stdout.flush()
        stream = load_dataset(dataset_name, split='train', streaming=True)
        count = [0]
        def take_gen():
            for item in stream:
                if count[0] >= num_samples:
                    break
                count[0] += 1
                if count[0] % 5000 == 0:
                    print(f'[预下载]   已下载 {count[0]}/{num_samples} 条...')
                    sys.stdout.flush()
                yield item
        ds = Dataset.from_generator(take_gen)
        os.makedirs(subset_dir, exist_ok=True)
        ds.save_to_disk(subset_dir)
        print(f'[预下载] ✅ {dataset_name} 子集就绪: {len(ds)} 条 (已缓存到 {subset_dir})')
else:
    print(f'[预下载] 正在加载 {dataset_name} (全量)...')
    sys.stdout.flush()
    try:
        ds = load_dataset(dataset_name, cache_dir=cache_dir, split='train')
        print(f'[预下载] ✅ {dataset_name} 就绪，共 {len(ds)} 条数据')
    except Exception as e:
        print(f'[预下载] ❌ {dataset_name} 下载失败: {e}')
        sys.exit(1)
"
    echo ""
fi

echo "[预下载] ✅ 所有数据集已就绪"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 第二步：启动分布式训练
# ─────────────────────────────────────────────────────────────────────────────

run_stage1() {
    echo "============================================================"
    echo "  Stage 1: 文本到图像预训练 (CC12M)"
    echo "  Config:   ${STAGE1_CONFIG}"
    echo "  Run Name: ${STAGE1_RUN_NAME}"
    echo "============================================================"
    echo ""

    # if [ ! -f "/home/liuzhirui/model/Qwen3-VL-main/metaquery-main/metaquery-main/configs/${STAGE1_CONFIG}" ]; then
    #     echo "错误: 配置文件 configs/${STAGE1_CONFIG} 不存在!"
    #     exit 1
    # fi
    STAGE1_CONFIG_PATH="${PROJECT_DIR}/configs/${STAGE1_CONFIG}"
    if [ ! -f "${STAGE1_CONFIG_PATH}" ]; then
        echo "错误: 配置文件 ${STAGE1_CONFIG_PATH} 不存在!"
        echo "提示: PROJECT_DIR=${PROJECT_DIR}"
        echo "提示: 请确认 configs/ 目录下包含 ${STAGE1_CONFIG}"
        ls -la "${PROJECT_DIR}/configs/" 2>/dev/null || echo "  configs/ 目录不存在"
        exit 1
    fi

    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        ${PROJECT_DIR}/train.py \
        --run_name "${STAGE1_RUN_NAME}" \
        --config_file "${STAGE1_CONFIG_PATH}" \
        --base_dir "${BASE_DIR}"

    echo ""
    echo "[Stage 1] ✅ 训练完成!"
    echo ""
}

run_stage2() {
    echo "============================================================"
    echo "  Stage 2: 指令微调 (MetaQuery-Instruct-2.4M)"
    echo "  Config:   ${STAGE2_CONFIG}"
    echo "  Run Name: ${STAGE2_RUN_NAME}"
    echo "============================================================"
    echo ""

    # if [ ! -f "/home/liuzhirui/model/Qwen3-VL-main/metaquery-main/metaquery-main/configs/${STAGE2_CONFIG}" ]; then
    #     echo "错误: 配置文件 configs/${STAGE2_CONFIG} 不存在!"
    #     exit 1
    # fi
    STAGE2_CONFIG_PATH="${PROJECT_DIR}/configs/${STAGE2_CONFIG}"
    if [ ! -f "${STAGE2_CONFIG_PATH}" ]; then
        echo "错误: 配置文件 ${STAGE2_CONFIG_PATH} 不存在!"
        echo "提示: PROJECT_DIR=${PROJECT_DIR}"
        echo "提示: 请确认 configs/ 目录下包含 ${STAGE2_CONFIG}"
        ls -la "${PROJECT_DIR}/configs/" 2>/dev/null || echo "  configs/ 目录不存在"
        exit 1
    fi

    # 自动查找 Stage 1 最新 checkpoint
    RESUME_ARG=""
    STAGE1_OUTPUT="${BASE_DIR}/output/${STAGE1_RUN_NAME}"
    if [ -d "${STAGE1_OUTPUT}" ]; then
        LATEST_CKPT=$(ls -d "${STAGE1_OUTPUT}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        if [ -n "${LATEST_CKPT}" ]; then
            RESUME_ARG="--resume_from_checkpoint ${LATEST_CKPT}"
            echo "[Stage 2] 从 Stage 1 checkpoint 恢复: ${LATEST_CKPT}"
        fi
    fi

    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        ${PROJECT_DIR}/train.py \
        --run_name "${STAGE2_RUN_NAME}" \
        --config_file "${STAGE2_CONFIG_PATH}" \
        --base_dir "${BASE_DIR}" \
        ${RESUME_ARG}

    echo ""
    echo "[Stage 2] ✅ 训练完成!"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# 执行训练
# ─────────────────────────────────────────────────────────────────────────────

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
        echo "错误: 不支持的训练阶段 '${STAGE}'"
        exit 1
        ;;
esac

echo "🎉 训练全部完成!"
