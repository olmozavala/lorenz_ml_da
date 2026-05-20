#!/bin/bash
#SBATCH --job-name=l63_experiment
#SBATCH --output=/home/jv24b/github/lorenz_ml_da/logs/experiment_%j.out
#SBATCH --error=/home/jv24b/github/lorenz_ml_da/logs/experiment_%j.err
#SBATCH --nodes=11
#SBATCH --ntasks=11              # One task per seed (11 seeds total)
#SBATCH --ntasks-per-node=5     # Ceiling of 9/2 — SLURM fills nodes greedily
#SBATCH --cpus-per-task=1       # Adjust if ms_denkf.py uses threading/multiprocessing
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jvelasco@coaps.fsu.edu
#SBATCH --partition=class

# ── Environment ──────────────────────────────────────────────────────────────
source /home/jv24b/miniconda3/etc/profile.d/conda.sh
conda activate lorenzo
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# ── Paths & config ───────────────────────────────────────────────────────────
WORKING_DIR="/home/jv24b/github/lorenz_ml_da"
LOG_DIR="${WORKING_DIR}/logs"
cd "$WORKING_DIR"


SEEDS=(31 132 233 234 435 536 637 738 839 940 999)

echo "=============================="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Nodes    : ${SLURM_JOB_NODELIST}"
echo "Started  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Seeds    : ${SEEDS[*]}"
echo "=============================="

# ── Build node list from SLURM allocation ─────────────────────────────────────

# Debug: print what SLURM gave us
echo "SLURM_JOB_NODELIST = '${SLURM_JOB_NODELIST}'"
echo "scontrol output    = '$(scontrol show hostnames "$SLURM_JOB_NODELIST")'"

# Parse node list
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NUM_NODES=${#NODELIST[@]}

# Guard: abort early with a clear message instead of cryptic division by zero
if [ "$NUM_NODES" -eq 0 ]; then
    echo "ERROR: Could not parse node list from SLURM_JOB_NODELIST='${SLURM_JOB_NODELIST}'"
    echo "       Falling back to single-node mode using \$(hostname)"
    NODELIST=("$(hostname)")
    NUM_NODES=1
fi

echo "Parsed nodes (${NUM_NODES}): ${NODELIST[*]}"


# ── Launch one task per seed, pinned to a node via round-robin ────────────────
declare -a PIDS        # track background PIDs
declare -A PID_SEED    # map PID → seed for error reporting
declare -A PID_NODE    # map PID → node for logging

for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    target_node=${NODELIST[$((i % NUM_NODES))]}

    LOG_OUT="${LOG_DIR}/seed_${seed}_${SLURM_JOB_ID}.out"
    LOG_ERR="${LOG_DIR}/seed_${seed}_${SLURM_JOB_ID}.err"

    srun --ntasks=1 --nodes=1 --nodelist="$target_node" --exclusive \
        bash -c "
            source /home/jv24b/miniconda3/etc/profile.d/conda.sh
            conda activate lorenzo
            export LD_LIBRARY_PATH=\"\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}\"
            cd ${WORKING_DIR}
            echo \"Running on: \$(hostname)\"
            echo \"Python: \$(which python)\"
            python -u ${WORKING_DIR}/ms_denkf.py --seed ${seed} --ic 2
        " >"$LOG_OUT" 2>"$LOG_ERR" &

    pid=$!
    PIDS+=("$pid")
    PID_SEED[$pid]=$seed
    PID_NODE[$pid]=$target_node

    echo "[$(date '+%H:%M:%S')] Launched seed=${seed} → ${target_node}  PID=${pid}"
done

echo ""
echo "All ${#SEEDS[@]} tasks launched. Waiting for completion..."
echo ""

# ── Wait and collect exit codes ───────────────────────────────────────────────
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid"
    exit_code=$?
    seed=${PID_SEED[$pid]}
    node=${PID_NODE[$pid]}

    if [ "$exit_code" -ne 0 ]; then
        echo "ERROR: seed=${seed} on ${node} (PID=${pid}) failed with exit code ${exit_code}"
        FAILED=$((FAILED + 1))
    else
        echo "OK:    seed=${seed} on ${node} (PID=${pid}) completed successfully"
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "Finished : $(date '+%Y-%m-%d %H:%M:%S')"
if [ "$FAILED" -gt 0 ]; then
    echo "RESULT   : ${FAILED}/${#SEEDS[@]} seeds FAILED"
    echo "           Check logs in: ${LOG_DIR}/"
    echo "=============================="
    exit 1
else
    echo "RESULT   : All ${#SEEDS[@]} seeds completed successfully"
    echo "           Logs in: ${LOG_DIR}/"
    echo "=============================="
fi
