#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=RQ2_p3
#SBATCH --output=logs/rq2_p3_%A_%a.out
#SBATCH --error=logs/rq2_p3_%A_%a.err
#SBATCH --array=0-9

set -euo pipefail

mkdir -p logs RQ2_RESULTS

source ~/.bashrc
conda activate myenv

cd ~/Research/DEEPCPD

TASK_OFFSET=20
GLOBAL_ID=$((TASK_OFFSET + SLURM_ARRAY_TASK_ID))

echo "Running RQ2 task ${GLOBAL_ID} (part 3, local index ${SLURM_ARRAY_TASK_ID})..."

python rq2.py \
  --task_id "${GLOBAL_ID}" \
  --outputdir RQ2_RESULTS
