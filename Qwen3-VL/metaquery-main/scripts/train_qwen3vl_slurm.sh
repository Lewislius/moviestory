#!/bin/bash
# =============================================================================
# MetaQuery + Qwen3-VL: SLURM Multi-Node Training Script
# =============================================================================
# Usage:
#   bash scripts/train_qwen3vl_slurm.sh <run_name> <config_file> [nodes]
#
# Example:
#   bash scripts/train_qwen3vl_slurm.sh qwen3vl8b_t2i qwen3vl8b_sana.yaml 2
# =============================================================================

run_name=$1
config_file=$2
nodes=${3:-1}
base_dir=${4:-"/path/to/base_dir"}
user_name=$(whoami)

if [ -z "$run_name" ] || [ -z "$config_file" ]; then
  echo "Usage: $0 <run_name> <config_file> [nodes] [base_dir]"
  echo ""
  echo "Examples:"
  echo "  $0 qwen3vl2b_t2i qwen3vl2b_sana.yaml 1 /data/metaquery"
  echo "  $0 qwen3vl8b_inst qwen3vl8b_sana_inst.yaml 2 /data/metaquery"
  exit 1
fi

# Create experiment directory
exp_dir="/home/$user_name/exp"
mkdir -p ${exp_dir}
rsync -av --exclude='__pycache__' --exclude='*/__pycache__' \
    --exclude='.git' --exclude='wandb' \
    "$(dirname "$0")/../" "${exp_dir}/${run_name}"
cd "${exp_dir}/${run_name}"

cat <<EOT > ${run_name}.sh
#!/bin/bash

#SBATCH --job-name=mq-${run_name}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH -t 3-00:00:00
#SBATCH --output=logs/${run_name}_%j.out
#SBATCH --error=logs/${run_name}_%j.err
#SBATCH --signal=USR1@140

mkdir -p logs

conda activate metaquery

export OMP_NUM_THREADS=12

if [ \$SLURM_NNODES -eq 1 ]; then
  srun torchrun --nproc-per-node=8 \
      train.py \
      --run_name ${run_name} \
      --config_file ${config_file} \
      --base_dir ${base_dir}
else
  srun torchrun \
      --nnodes=\$SLURM_NNODES \
      --nproc_per_node=8 \
      --rdzv_id=\$SLURM_JOB_ID \
      --rdzv_backend=c10d \
      --rdzv_endpoint=\$HOSTNAME:29500 \
      train.py \
      --run_name ${run_name} \
      --config_file ${config_file} \
      --base_dir ${base_dir}
fi
EOT

echo "Submitting SLURM job: ${run_name}"
sbatch ${run_name}.sh
echo "Job submitted. Check logs/ directory for output."
