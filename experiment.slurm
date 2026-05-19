#!/bin/bash
#SBATCH --job-name=l63_experiment
#SBATCH --output=/home/jv24b/github/lorenz_ml_da/logs/train_%j.out
#SBATCH --error=/home/jv24b/github/lorenz_ml_da/logs/train_%j.err
#SBATCH --nodes=10
#SBATCH --ntasks=16
#SBATCH --mem=16000M
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jvelasco@coaps.fsu.edu
#SBATCH --partition=class

# Path to my conda manager
source /home/jv24b/miniconda3/etc/profile.d/conda.sh

WORKING_DIR="/home/jv24b/github/lorenz_ml_da"
cd $WORKING_DIR

# Activate lorenzo env
conda activate lorenzo

# Ensure conda's newer libstdc++ is used (avoids GLIBCXX_* import errors)
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# Launch training
    echo "Job started at: $(date +"%Y-%m-%d %H:%M:%S")"
    echo "Training started at: $(date +"%Y-%m-%d %H:%M:%S")"


python -u $WORKING_DIR/ms_denkf.py
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
