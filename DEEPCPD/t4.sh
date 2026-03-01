#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=RQ2_p4
#SBATCH --output=logs/rq2_p4_%A_%a.out
#SBATCH --error=logs/rq2_p4_%A_%a.err
#SBATCH --array=0-1   # only two tasks: 30,31

set -euo pipefail

mkdir -p logs RQ2_RESULTS

source ~/.bashrc
conda activate myenv

cd ~/Research/DEEPCPD

TASK_OFFSET=30
GLOBAL_ID=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))

echo "Running RQ2 task ${GLOBAL_ID} (part 4, local index ${SLURM_ARRAY_TASK_ID})..."

python rq2.py \
  --task_id "${GLOBAL_ID}" \
  --outputdir RQ2_RESULTS
