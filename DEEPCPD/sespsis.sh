#!/bin/bash
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=build_sepsis
#SBATCH --output=logs/process_rq3_%j.out
#SBATCH --error=logs/process_rq3_%j.err

set -euo pipefail

source ~/.bashrc
conda activate myenv
cd ~/Research/DEEPCPD

python3 preprocess_rq3_cohort.py
