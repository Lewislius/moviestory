#! /bin/bash
eval "$(conda shell.bash hook)"
conda env list
conda activate /home/liuzhirui/miniconda3/envs/wan
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://172.17.0.38:9090
export https_proxy=http://172.17.0.38:9090
work_dir=/home/liuzhirui

python ${work_dir}/model/Wan2.2/generate.py --task animate-14B --ckpt_dir ${work_dir}/model/Wan2.2/Wan2.2-Animate-14B/ --src_root_path ${work_dir}/model/Wan2.2/examples/wan_animate/animate/process_results/ --refert_num 1 