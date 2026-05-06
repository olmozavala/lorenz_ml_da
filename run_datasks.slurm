#!/bin/bash
#SBATCH --job-name=l63_DA_exp
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

CUDA_VISIBLE_DEVICES=0
# Path to my conda manager
source /home/jv24b/miniconda3/etc/profile.d/conda.sh

# Activate lorenzo env
conda activate lorenzo

# Launch training
echo "Job started at: $(date +"%Y-%m-%d %H:%M:%S")"
echo "Data Assimilation scripts started at: $(date +"%Y-%m-%d %H:%M:%S")"

python -u D_EnKF.py


echo "Training completed at: $(date +"%Y-%m-%d %H:%M:%S")"
echo "Job completed at: $(date +"%Y-%m-%d %H:%M:%S")"

echo ""
echo "=============================="
echo "All done : $(date)"
echo "=============================="



