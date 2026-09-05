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

torchrun --standalone --nproc_per_node=2 /home/liuzhirui/model/Wan2.2/train_metaquery_i2v_new.py \
  --run_two_stage \
  --distributed \
  --auto_device_map \
  --gpus_per_process 2 \
  --hf_subset_ratio 0.01 \
  --hf_scan_factor 500 \
  --hf_no_streaming \
  --hf_no_subset_cache \
  --dataloader_num_workers 0 \
  --output_root /home/liuzhirui/model/Wan2.2/metaquery_i2v_two_stage \
  --stage1_steps 5000 \
  --stage2_steps 3000 \
  --wandb_enabled \
  --wandb_project wan-metaquery \
  --wandb_run_name wan-i2v-metaquery-training-new-determined \
  --wandb_tags determined,i2v,mq \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_log_checkpoint
