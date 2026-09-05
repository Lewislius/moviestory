#!/bin/bash
###############################################################################
# batch_caption.sh
# 批量视频字幕生成 —— 遍历数据集目录，逐条调用 Python 生成字幕
#
# 目录结构：
#   DATASET_BASE/
#     S1940/  S1950/  S1960/ ...
#       output_<剧集名>/
#         segments/       <- segment_XXXX.mp4
#         references/     <- segment_XXXX_characterY_crop.jpg
#                            segment_XXXX_characterY_metadata.json
#
# 用法：
#   bash batch_caption.sh [DATASET_BASE] [MODEL_NAME]
#
# 示例：
#   bash batch_caption.sh ./dataset/output_Tom_and_Jerry
#   bash batch_caption.sh ./dataset/output_Tom_and_Jerry Qwen/Qwen3-VL-7B-Instruct
###############################################################################
#!/bin/bash
###############################################################################
# batch_caption.sh
# 批量视频字幕生成 —— 遍历数据集目录，逐条调用 Python 生成字幕
#
# 目录结构：
#   DATASET_BASE/
#     S1940/  S1950/  S1960/ ...
#       output_<剧集名>/
#         segments/       <- segment_XXXX.mp4
#         references/     <- segment_XXXX_characterY_crop.jpg
#                            segment_XXXX_characterY_metadata.json
#
# 用法：
#   bash batch_caption.sh [DATASET_BASE] [MODEL_NAME] [OUTPUT_BASE]
#
# 示例：
#   bash batch_caption.sh ./dataset/output_Tom_and_Jerry
#   bash batch_caption.sh ./dataset/output_Tom_and_Jerry Qwen/Qwen3-VL-7B-Instruct
#   bash batch_caption.sh ./dataset/output_Tom_and_Jerry "" ./my_captions
###############################################################################

set -euo pipefail

# ──────────────── 参数 ────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/video-caption-demo.py"

DATASET_BASE="${1:-/home/liuzhirui/model/SCAIL/dataset/output_Hayao_Movie_1280x720/output_kiki_delivery}"
MODEL_NAME="${2:-}"
OUTPUT_BASE="${3:-/home/liuzhirui/model/Qwen3-VL/dataset/Hayao_Movie_1280x720/output_kiki_delivery}"  # 结果输出根目录

# ──────────────── 统计 ────────────────
TOTAL_TASKS=0
DONE_TASKS=0
SKIPPED_EPISODES=0

echo "============================================================"
echo " 批量视频字幕生成"
echo " 数据集目录: ${DATASET_BASE}"
echo " 输出目录  : ${OUTPUT_BASE}"
[ -n "${MODEL_NAME}" ] && echo " 模型      : ${MODEL_NAME}"
echo "============================================================"

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "[错误] 找不到 Python 脚本: ${PYTHON_SCRIPT}"
    exit 1
fi

# ──────────────── 遍历 ────────────────
for season_dir in "${DATASET_BASE}"/S*/; do
    [ ! -d "${season_dir}" ] && continue
    echo ""
    echo ">>> 季度: $(basename "${season_dir}")"

    for episode_dir in "${season_dir}"*/; do
        [ ! -d "${episode_dir}" ] && continue
        episode_name="$(basename "${episode_dir}")"
        segments_dir="${episode_dir}segments"
        references_dir="${episode_dir}references"

        # ① segments 目录不存在 → 跳过
        if [ ! -d "${segments_dir}" ]; then
            echo "  [跳过] ${episode_name}: 无 segments 目录"
            SKIPPED_EPISODES=$((SKIPPED_EPISODES + 1))
            continue
        fi

        # ② segments 下无 mp4 → 跳过
        mp4_count=$(find "${segments_dir}" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)
        if [ "${mp4_count}" -eq 0 ]; then
            echo "  [跳过] ${episode_name}: segments/ 下无 mp4"
            SKIPPED_EPISODES=$((SKIPPED_EPISODES + 1))
            continue
        fi

        # ③ references 目录不存在 → 跳过
        if [ ! -d "${references_dir}" ]; then
            echo "  [跳过] ${episode_name}: 无 references 目录"
            SKIPPED_EPISODES=$((SKIPPED_EPISODES + 1))
            continue
        fi

        # # 从剧集文件夹名提取 S1940E01 式编号，若无则截断文件夹名兜底
        # if [[ "${episode_name}" =~ (S[0-9]{4}E[0-9]+) ]]; then
        #     episode_code="${BASH_REMATCH[1]}"
        # else
        #     episode_code="$(echo "${episode_name}" | tr ' /()' '_' | cut -c1-40)"
        # fi
        episode_code="${episode_name}"

        # 结果写入新目录 OUTPUT_BASE/S1940E01/captions_output.json
        output_json="${OUTPUT_BASE}/${episode_code}/captions_output.json"
        mkdir -p "${OUTPUT_BASE}/${episode_code}"

        echo "  ─── ${episode_name} ───"
        echo "      → ${output_json}"

        crop_found=0
        for crop_path in "${references_dir}"/*_crop.jpg; do
            [ ! -f "${crop_path}" ] && continue
            crop_filename="$(basename "${crop_path}")"

            # 解析 segment_XXXX_characterY_crop.jpg
            if [[ "${crop_filename}" =~ ^(segment_[0-9]+)_character([0-9]+)_crop\.jpg$ ]]; then
                segment_id="${BASH_REMATCH[1]}"
                character_idx="${BASH_REMATCH[2]}"
            else
                echo "    [警告] 无法解析文件名: ${crop_filename}"
                continue
            fi

            # ④ 确认对应 mp4 存在
            segment_mp4="${segments_dir}/${segment_id}.mp4"
            if [ ! -f "${segment_mp4}" ]; then
                echo "    [警告] ${segment_id}.mp4 不存在，跳过"
                continue
            fi

            # ⑤ metadata（可选）
            metadata_path="${references_dir}/${segment_id}_character${character_idx}_metadata.json"

            crop_found=$((crop_found + 1))
            TOTAL_TASKS=$((TOTAL_TASKS + 1))
            echo "    [任务 ${TOTAL_TASKS}] ${segment_id} | character${character_idx}"

            # ⑥ 用数组构建参数，确保含空格的路径被正确引用
            python_args=(
                --mode single_ref
                --video "${segment_mp4}"
                --ref_image "${crop_path}"
                --output_json "${output_json}"
            )
            [ -f "${metadata_path}" ] && python_args+=(--metadata "${metadata_path}")
            [ -n "${MODEL_NAME}" ]    && python_args+=(--model "${MODEL_NAME}")

            python "${PYTHON_SCRIPT}" "${python_args[@]}" \
                && DONE_TASKS=$((DONE_TASKS + 1)) \
                || echo "    [错误] 处理失败: ${segment_id}_character${character_idx}"
        done

        [ "${crop_found}" -eq 0 ] && echo "    [信息] 无有效 crop.jpg，跳过此集"
    done
done

echo ""
echo "============================================================"
echo " 完成！总任务: ${TOTAL_TASKS}  完成: ${DONE_TASKS}  跳过剧集: ${SKIPPED_EPISODES}"
echo "============================================================"
