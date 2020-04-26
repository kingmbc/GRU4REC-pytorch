#!/bin/bash

# 실행 파라미터 출력

###############################################
# Batch job information
###############################################
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
##SBATCH --mem-per-cpu=64GB
#SBATCH --time=24:00:00  # Default: one-hour 01:00:00

# GPU resources to request
# - Format: "gpu:{NUM_GPUS}" or "gpu:{GPU_TYPE}:{NUM_GPUS}" (if you want to specify a gpu type)
# - Available GPU types: p1080, k80, p40, p100, v100
# - e.g., #SBATCH --gres=gpu:p100:2
#SBATCH --gres=gpu:1

#SBATCH --job-name=SR-GNN-batch  # Specify the name of this job.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tk2637@nyu.edu
#SBATCH --output=%x.%j.out

###############################################
# Module information to use
###############################################
module purge
module load cuda/10.1.105 cudnn/10.1v7.6.5.32

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

##########################################################
# From here you can write your own commands to run.
# e.g.
#   python train.py
##########################################################
source activate pytorch
cd /scratch/tk2637/SR-GNN/pytorch_code

bash train_script.sh $dataset