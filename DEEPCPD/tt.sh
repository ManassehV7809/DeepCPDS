#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=RQ2_part1
#SBATCH --output=logs/rq2_p1_%A_%a.out
#SBATCH --error=logs/rq2_p1_%A_%a.err
#SBATCH --array=0-9

set -euo pipefail

mkdir -p logs RQ2_RESULTS

source ~/.bashrc
conda activate myenv

cd ~/Research/DEEPCPD

echo "Running RQ2 task ${SLURM_ARRAY_TASK_ID} (part 1)..."

python rq2.py \
  --task_id "${SLURM_ARRAY_TASK_ID}" \
  --outputdir RQ2_RESULTS
