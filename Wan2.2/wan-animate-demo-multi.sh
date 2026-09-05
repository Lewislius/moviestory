#! /bin/bash
eval "$(conda shell.bash hook)"
conda env list
conda activate /home/liuzhirui/miniconda3/envs/wan

export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://10.39.23.15:808
export https_proxy=http://10.39.23.15:808
work_dir=/home/liuzhirui
# 设置完整的 CUDA 环境
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_VISIBLE_DEVICES=0,2,3,4

# 验证 CUDA 环境
echo "=== CUDA Environment Check ==="
which nvcc
nvidia-smi
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "PyTorch device count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# python ${work_dir}/model/Wan2.2/generate.py --task animate-14B --ckpt_dir ${work_dir}/model/Wan2.2/Wan2.2-Animate-14B/ --src_root_path ${work_dir}/model/Wan2.2/examples/wan_animate/animate/process_results/ --refert_num 1 
python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 ${work_dir}/model/Wan2.2/generate.py --task animate-14B --ckpt_dir ${work_dir}/model/Wan2.2/Wan2.2-Animate-14B/ --src_root_path ${work_dir}/model/Wan2.2/examples/wan_animate/animate/process_results/ --refert_num 1 --dit_fsdp --t5_fsdp --ulysses_size 4