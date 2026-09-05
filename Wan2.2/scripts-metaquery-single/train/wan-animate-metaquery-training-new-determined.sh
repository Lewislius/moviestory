#!/bin/bash
eval "$(conda shell.bash hook)"
conda env list
conda activate /home/liuzhirui/miniconda3/envs/moviestory
export http_proxy=10.130.130.6:56830
export https_proxy=10.130.130.6:56830
export HF_ENDPOINT=https://hf-mirror.com
: "${HF_TOKEN:?Set HF_TOKEN in the environment before running this script}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="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"
# 关键：让代码自动从HF datasets仓库拉取OpenVid归档
export OPENVID_SNAPSHOT_DOWNLOAD=1
export OPENVID_SNAPSHOT_DIR=/home/liuzhirui/hf_cache/openvid_repo
# export OPENVID_SNAPSHOT_PATTERNS="OpenVid_part*.zip,OpenVid_part*.part*,OpenVidHD.csv,data/*"
export OPENVID_SNAPSHOT_PATTERNS="OpenVid_part0.zip,OpenVid_part1.zip,OpenVidHD.csv,data/*"
export OPENVID_AUTO_JOIN_PARTS=1
export OPENVID_ALLOW_HTTP_GUESS=0
export OPENVID_RECORD_STREAMING=1
export OPENVID_SNAPSHOT_DOWNLOAD=0

export WAN_DATA_PRECLEAN_SCAN_CAP=30000
export WAN_DATA_PRECLEAN_ZERO_ACCEPT_ABORT_SCAN=5000
# torchrun --standalone --nproc_per_node=2 /home/liuzhirui/model/Wan2.2/train_metaquery_animate_new.py \
#   --run_two_stage \
#   --distributed \
#   --auto_device_map \
#   --gpus_per_process 2 \
#   --output_root /home/liuzhirui/model/Wan2.2/metaquery_animate_two_stage \
#   --stage1_steps 5000 \
#   --stage2_steps 3000
# torchrun --standalone --nproc_per_node=2 /home/liuzhirui/model/Wan2.2/train_metaquery_animate_new.py \
#   --run_two_stage \
#   --distributed \
#   --auto_device_map \
#   --gpus_per_process 2 \
#   --hf_subset_ratio 0.001 \
#   --hf_scan_factor 500 \
#   --hf_no_streaming \
#   --hf_no_subset_cache \
#   --dataloader_num_workers 0 \
#   --output_root /home/liuzhirui/model/Wan2.2/metaquery_animate_two_stage \
#   --stage1_steps 5000 \
#   --stage2_steps 3000 \
#   --wandb_enabled \
#   --wandb_project wan-metaquery \
#   --wandb_run_name wan-animate-metaquery-training-new-determined \
#   --wandb_tags determined,animate,mq \
#   --wandb_mode "${WANDB_MODE}" \
#   --wandb_log_checkpoint

# 下面是进行逐条下载
torchrun --standalone --nproc_per_node=2 /home/liuzhirui/model/Wan2.2/train_metaquery_animate_new.py \
  --run_two_stage \
  --distributed \
  --auto_device_map \
  --gpus_per_process 1 \
  --hf_subset_ratio 0.01 \
  --hf_scan_factor 500 \
  --hf_no_streaming \
  --hf_no_subset_cache \
  --dataloader_num_workers 0 \
  --output_root /home/liuzhirui/model/Wan2.2/metaquery_animate_two_stage \
  --stage1_steps 5000 \
  --stage2_steps 3000 \
  --wandb_enabled \
  --wandb_project wan-metaquery \
  --wandb_run_name wan-animate-metaquery-training-new-determined \
  --wandb_tags determined,animate,mq \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_log_checkpoint
