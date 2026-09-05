#! /bin/bash
eval "$(conda shell.bash hook)"
conda env list
conda activate /home/liuzhirui/miniconda3/envs/wan
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://10.39.23.15:808
export https_proxy=http://10.39.23.15:808
work_dir=/home/liuzhirui
# export CUDA_VISIBLE_DEVICES=0,2

# python ${work_dir}/model/Wan2.2/generate.py --task animate-14B --ckpt_dir ${work_dir}/model/Wan2.2/Wan2.2-Animate-14B/ --src_root_path ${work_dir}/model/Wan2.2/examples/wan_animate/animate/process_results/ --refert_num 1 
python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 ${work_dir}/model/Wan2.2/generate.py --task animate-14B --ckpt_dir ${work_dir}/model/Wan2.2/Wan2.2-Animate-14B/ --src_root_path ${work_dir}/model/Wan2.2/examples/wan_animate/replace/process_results/ --refert_num 1 --dit_fsdp --t5_fsdp --ulysses_size 4