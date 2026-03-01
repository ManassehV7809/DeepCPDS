#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=RQ3_MIMIC
#SBATCH --output=logs/rq3_mimic_%j.out
#SBATCH --error=logs/rq3_mimic_%j.err

set -euo pipefail

source ~/.bashrc
conda activate myenv
cd ~/Research/DEEPCPD

mkdir -p logs
mkdir -p RQ3_RESULTS

python3 rq3.py \
  --output_dir RQ3_RESULTS
