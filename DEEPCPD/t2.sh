#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=RQ2_part2
#SBATCH --output=logs/rq2_p2_%A_%a.out
#SBATCH --error=logs/rq2_p2_%A_%a.err
#SBATCH --array=10-19      # remaining 10 tasks

mkdir -p logs RQ2_RESULTS

source ~/.bashrc
conda activate myenv

echo "Running RQ2 task $SLURM_ARRAY_TASK_ID (part 2)..."

python rq2_rq1style_full.py \
  --task_id $SLURM_ARRAY_TASK_ID \
  --output_dir RQ2_RESULTS

