#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=RQ2_rq1style
#SBATCH --output=logs/rq2_%A_%a.out
#SBATCH --error=logs/rq2_%A_%a.err
#SBATCH --array=0-19%10   # 20 tasks total, max 10 running at once

# Create directories
mkdir -p logs RQ2_RESULTS

source ~/.bashrc
conda activate myenv

echo "Running task $SLURM_ARRAY_TASK_ID..."

python rq2_rq1style_full.py \
  --task_id $SLURM_ARRAY_TASK_ID \
  --output_dir RQ2_RESULTS

