# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

run_name=$1
config_file=$2
nodes=${3:-1}
user_name=$(whoami)

if [ -z "$run_name" ] || [ -z "$config_file" ] || [ -z "$nodes" ]; then
  echo "Usage: $0 <run_name> <config_file> <nodes>"
  exit 1
fi

mkdir -p /home/$user_name/exp
rsync -av --exclude='__pycache__' --exclude='*/__pycache__' --exclude='.git' --exclude='wandb' /home/$user_name/metaquery/ /home/$user_name/exp/$run_name
cd /home/$user_name/exp/$run_name

cat <<EOT > $run_name.sh
#!/bin/bash

#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH -t 3-00:00:00
#SBATCH --output=out
#SBATCH --error=err
#SBATCH --signal=USR1@140

conda activate metaquery

export OMP_NUM_THREADS=12

if [ \$SLURM_NNODES -eq 1 ]; then
  srun torchrun --nproc-per-node=8 train.py --run_name $run_name --config_file $config_file --base_dir /path/to/base_dir
else
  srun torchrun --nnodes=\$SLURM_NNODES --nproc_per_node=8 \\
      --rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=\$HOSTNAME:29500 train.py --run_name $run_name --config_file $config_file --base_dir /path/to/base_dir
fi
EOT

sbatch $run_name.sh