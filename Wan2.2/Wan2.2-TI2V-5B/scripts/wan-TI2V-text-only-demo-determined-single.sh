#!/usr/bin/env bash
# Wan2.2-TI2V-5B selected-text-file-to-video demo (single GPU).
# Only the exact .txt files listed in PROMPT_FILES (or passed as arguments) run.

set -euo pipefail

source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/wan

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1

WAN_ROOT="/home/liuzhirui/model/Wan2.2"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WAN_ROOT}/Wan2.2-TI2V-5B}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${CHECKPOINT_DIR}/outputs_text_only}"
SIZE="${SIZE:-1280*704}"
BASE_SEED="${BASE_SEED:-42}"

# Explicitly list only the prompt files you want to process.
# Add/remove absolute .txt paths here; files not listed here are ignored.
PROMPT_FILES=(
    "/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B/examples/physics_text_prompts/001-yolo-plus-v2.txt"
    # "/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B/examples/physics_text_prompts/002-bowler.txt"
)

# Optional: paths passed on the command line replace the list above.
if [[ $# -gt 0 ]]; then
    PROMPT_FILES=("$@")
fi

if [[ ! -f "${WAN_ROOT}/generate.py" ]]; then
    echo "ERROR: generate.py not found: ${WAN_ROOT}/generate.py" >&2
    exit 1
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "ERROR: checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

if [[ ${#PROMPT_FILES[@]} -eq 0 ]]; then
    echo "ERROR: no prompt files selected. Add paths to PROMPT_FILES or pass .txt paths as arguments." >&2
    exit 1
fi

if [[ ! "${BASE_SEED}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: BASE_SEED must be a non-negative integer: ${BASE_SEED}" >&2
    exit 1
fi

mkdir -p "${OUTPUTS_DIR}"

# Validate and resolve every selected path before inference starts.
prompt_files=()
for prompt_file in "${PROMPT_FILES[@]}"; do
    if [[ ! -f "${prompt_file}" ]]; then
        echo "ERROR: selected prompt file not found: ${prompt_file}" >&2
        exit 1
    fi
    if [[ "${prompt_file,,}" != *.txt ]]; then
        echo "ERROR: selected prompt file must end in .txt: ${prompt_file}" >&2
        exit 1
    fi
    prompt_files+=("$(realpath "${prompt_file}")")
done

cd "${WAN_ROOT}"

total_count=${#prompt_files[@]}
success_count=0
fail_count=0

echo "========================================"
echo "Wan2.2-TI2V-5B text-only batch generation"
echo "Selected prompt files: ${total_count}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Outputs: ${OUTPUTS_DIR}"
echo "Size: ${SIZE}"
echo "Base seed: ${BASE_SEED}"
echo "========================================"

for prompt_index in "${!prompt_files[@]}"; do
    prompt_file="${prompt_files[$prompt_index]}"
    prompt_filename="$(basename "${prompt_file}")"
    prompt_name="${prompt_filename%.*}"
    prompt="$(tr -d '\r' < "${prompt_file}")"
    item_number=$((prompt_index + 1))
    item_id="$(printf '%03d' "${item_number}")"
    seed=$((BASE_SEED + prompt_index))
    generation_time="$(date +%Y%m%d_%H%M%S)"
    output_file="${OUTPUTS_DIR}/${prompt_name}_${generation_time}.mp4"

    if [[ -z "${prompt//[[:space:]]/}" ]]; then
        echo "[${item_id}/${total_count}] ERROR: prompt file is empty; skipping: ${prompt_file}"
        fail_count=$((fail_count + 1))
        continue
    fi

    echo
    echo "----------------------------------------"
    echo "[${item_id}/${total_count}] Prompt file: ${prompt_file}"
    echo "[${item_id}/${total_count}] Prompt used for inference:"
    printf '%s\n' "${prompt}"
    echo "[${item_id}/${total_count}] Seed: ${seed}"
    echo "[${item_id}/${total_count}] Output: ${output_file}"

    start_time=$(date +%s)

    # No --image argument is supplied: ti2v-5B therefore runs in pure T2V mode.
    if python "${WAN_ROOT}/generate.py" \
        --task ti2v-5B \
        --size "${SIZE}" \
        --ckpt_dir "${CHECKPOINT_DIR}" \
        --offload_model True \
        --convert_model_dtype \
        --t5_cpu \
        --base_seed "${seed}" \
        --prompt "${prompt}" \
        --save_file "${output_file}"; then
        duration=$(($(date +%s) - start_time))
        if [[ -f "${output_file}" ]]; then
            echo "[${item_id}/${total_count}] SUCCESS (${duration}s)"
            success_count=$((success_count + 1))
        else
            echo "[${item_id}/${total_count}] FAILED: command succeeded but MP4 was not created."
            fail_count=$((fail_count + 1))
        fi
    else
        exit_code=$?
        duration=$(($(date +%s) - start_time))
        echo "[${item_id}/${total_count}] FAILED after ${duration}s (exit code ${exit_code})."
        fail_count=$((fail_count + 1))
    fi
done

echo
echo "========================================"
echo "Batch generation finished"
echo "Total: ${total_count}"
echo "Succeeded: ${success_count}"
echo "Failed: ${fail_count}"
echo "========================================"

if [[ ${fail_count} -ne 0 ]]; then
    exit 1
fi
