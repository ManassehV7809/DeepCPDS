#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=RQ2_all
#SBATCH --output=logs/rq2_all_%j.out
#SBATCH --error=logs/rq2_all_%j.err
#SBATCH --time=12:00:00   # adjust if needed

set -euo pipefail

mkdir -p logs RQ2_RESULTS

source ~/.bashrc
conda activate myenv

cd ~/Research/DEEPCPD

echo "Running full RQ2 grid sequentially..."

python rq2.py --run_all --outputdir RQ2_RESULTS
