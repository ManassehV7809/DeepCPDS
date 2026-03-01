#!/bin/bash
#SBATCH --partition=biggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=RQ3_SCARCE
#SBATCH --output=logs/rq3_scarce_%j.out
#SBATCH --error=logs/rq3_scarce_%j.err

set -euo pipefail

source ~/.bashrc
conda activate myenv
cd ~/Research/DEEPCPD

mkdir -p logs
mkdir -p RQ3_SCARCE_RESULTS

python3 rq3_scarce.py \
  --output_dir RQ3_SCARCE_RESULTS
