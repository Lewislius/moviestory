#! /bin/bash
eval "$(conda shell.bash hook)"
conda env list
conda activate /home/liuzhirui/miniconda3/envs/wan
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://10.39.23.15:808
export https_proxy=http://10.39.23.15:808
# export CUDA_VISIBLE_DEVICES=6
work_dir=/home/liuzhirui

python ${work_dir}/model/Wan2.2/generate.py  \
    --task t2v-A14B \
    --size 1280*720 \
    --ckpt_dir .${work_dir}/model/Wan2.2/Wan2.2-T2V-A14B \
    --offload_model True \
    --convert_model_dtype \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."