#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=RQ2_pilot
#SBATCH --output=logs/rq2_%A_%a.out
#SBATCH --error=logs/rq2_%A_%a.err
#SBATCH --array=10-11

python rq2.py --task_id $SLURM_ARRAY_TASK_ID

