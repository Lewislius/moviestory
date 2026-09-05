#!/bin/bash
# Batch generation script for TI2V
# This script processes all example folders (001-010) and generates videos
set -euo pipefail
source /home/liuzhirui/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuzhirui/miniconda3/envs/wan
export HF_ENDPOINT=https://hf-mirror.com
# export http_proxy=http://10.39.23.15:808
# export https_proxy=http://10.39.23.15:808
work_dir=/home/liuzhirui
# Default parameters
START_FOLDER="001"
END_FOLDER="005"
CHECKPOINT_DIR="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B"
EXAMPLES_DIR="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B/examples"
OUTPUTS_DIR="/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B/outputs"
SIZE="1280*704"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --start)
            START_FOLDER="$2"
            shift 2
            ;;
        --end)
            END_FOLDER="$2"
            shift 2
            ;;
        --ckpt_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --examples_dir)
            EXAMPLES_DIR="$2"
            shift 2
            ;;
        --outputs_dir)
            OUTPUTS_DIR="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create outputs directory if it doesn't exist
if [ ! -d "$OUTPUTS_DIR" ]; then
    mkdir -p "$OUTPUTS_DIR"
    echo "Created outputs directory: $OUTPUTS_DIR"
fi

echo "========================================"
echo "Starting batch TI2V generation"
echo "Processing folders: $START_FOLDER to $END_FOLDER"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Examples directory: $EXAMPLES_DIR"
echo "Outputs directory: $OUTPUTS_DIR"
echo "Size: $SIZE"
echo "========================================"
echo ""

# Convert folder names to integers
start_num=$(echo "$START_FOLDER" | sed 's/^0*//')
end_num=$(echo "$END_FOLDER" | sed 's/^0*//')

# Counter for successful and failed generations
success_count=0
fail_count=0
total_count=0

# Loop through each folder
for i in $(seq $start_num $end_num); do
    # Format folder name with leading zeros
    folder_name=$(printf "%03d" $i)
    folder_path="$EXAMPLES_DIR/$folder_name"
    
    total_count=$((total_count + 1))
    
    echo "========================================"
    echo "Processing folder: $folder_name"
    echo "========================================"
    
    # Check if folder exists
    if [ ! -d "$folder_path" ]; then
        echo "[$folder_name] WARNING: Folder not found: $folder_path"
        echo "[$folder_name] Skipping..."
        echo ""
        fail_count=$((fail_count + 1))
        continue
    fi
    
    # Define paths for input files
    input_text="$folder_path/input.txt"
    
    # Find any jpg/jpeg file in the folder
    ref_image=$(find "$folder_path" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) | head -n 1)
    
    # Check if any jpg/jpeg file exists
    if [ -z "$ref_image" ]; then
        echo "[$folder_name] ERROR: No .jpg or .jpeg file found in $folder_path"
        echo "[$folder_name] Skipping..."
        echo ""
        fail_count=$((fail_count + 1))
        continue
    fi
    
    echo "[$folder_name] Found image: $(basename "$ref_image")"
    
    # Check if input.txt exists
    if [ ! -f "$input_text" ]; then
        echo "[$folder_name] ERROR: input.txt not found in $folder_path"
        echo "[$folder_name] Skipping..."
        echo ""
        fail_count=$((fail_count + 1))
        continue
    fi
    
    # Read the prompt from input.txt
    prompt=$(cat "$input_text" | tr -d '\r' | tr -d '\n')
    
    if [ -z "$prompt" ]; then
        echo "[$folder_name] ERROR: input.txt is empty"
        echo "[$folder_name] Skipping..."
        echo ""
        fail_count=$((fail_count + 1))
        continue
    fi
    
    # Create output filename
    timestamp=$(date +"%Y%m%d_%H%M%S")
    output_file="$OUTPUTS_DIR/${folder_name}_${timestamp}.mp4"
    
    echo "[$folder_name] Reference image: $ref_image"
    echo "[$folder_name] Prompt: $prompt"
    echo "[$folder_name] Output file: $output_file"
    echo ""
    
    # Run the generation command
    echo "[$folder_name] Starting generation..."
    start_time=$(date +%s)
    
    python ${work_dir}/model/Wan2.2/generate.py \
        --task ti2v-5B \
        --size "$SIZE" \
        --ckpt_dir "$CHECKPOINT_DIR" \
        --offload_model True \
        --convert_model_dtype \
        --t5_cpu \
        --prompt "$prompt" \
        --image "$ref_image" \
        --save_file "$output_file"
    
    exit_code=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    
    # Check if generation was successful
    if [ $exit_code -eq 0 ] && [ -f "$output_file" ]; then
        echo "[$folder_name] SUCCESS: Video generated in ${duration} seconds"
        echo "[$folder_name] Output saved to: $output_file"
        success_count=$((success_count + 1))
    else
        echo "[$folder_name] FAILED: Generation failed with exit code $exit_code"
        fail_count=$((fail_count + 1))
    fi
    
    echo ""
    echo "----------------------------------------"
    echo ""
done

# Print summary
echo ""
echo "========================================"
echo "Batch generation completed!"
echo "========================================"
echo "Total folders processed: $total_count"
echo "Successful generations: $success_count"
echo "Failed generations: $fail_count"
echo "========================================"

# Exit with appropriate code
if [ $fail_count -eq 0 ]; then
    exit 0
else
    exit 1
fi
