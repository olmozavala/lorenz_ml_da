#!/bin/bash
#SBATCH --job-name=l63_training
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jvelasco@coaps.fsu.edu


# Path to my conda manager
source /home/jv24b/miniconda3/etc/profile.d/conda.sh

# Activate lorenzo env
conda activate lorenzo

# Ensure conda's newer libstdc++ is used (avoids GLIBCXX_* import errors)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

ML_MODEL_CONFIG_FILE="${1:-config.yml}"
echo "ML_MODEL_CONFIG_FILE: $ML_MODEL_CONFIG_FILE"

# Launch training
echo "Job started at: $(date +"%Y-%m-%d %H:%M:%S")"
echo "Training started at: $(date +"%Y-%m-%d %H:%M:%S")"

python Main_ML.py "$ML_MODEL_CONFIG_FILE"
TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "Training completed at: $(date +"%Y-%m-%d %H:%M:%S")"
echo "Job completed at: $(date +"%Y-%m-%d %H:%M:%S")"

echo ""
echo "=============================="
echo "All done : $(date)"
echo "=============================="
