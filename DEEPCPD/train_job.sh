#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=DeepCPDs_RQ1
#SBATCH --output=logs/rq1_%A_%a.out
#SBATCH --error=logs/rq1_%A_%a.err

set -euo pipefail

# Change to the working directory explicitly
cd /home-mscluster/$USER/Research/DEEPCPD/

python rq1.py --mode merge --output_dir RQ1_RESULTS_COMBINED

