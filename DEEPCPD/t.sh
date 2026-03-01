#!/bin/bash
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=RQ2_pilot
#SBATCH --output=logs/rq2_%A_%a.out
#SBATCH --error=logs/rq2_%A_%a.err
#SBATCH --array=0-9

# Create directories
mkdir -p logs RQ2_PILOT_RESULTS

source ~/.bashrc
conda activate myenv

echo "Running task $SLURM_ARRAY_TASK_ID..."
python rq2_pilot_complete.py --task_id $SLURM_ARRAY_TASK_ID

# If this is the last array task (9), also run tasks 10-11 and merge
if [ $SLURM_ARRAY_TASK_ID -eq 9 ]; then
    echo ""
    echo "=========================================="
    echo "Running remaining tasks 10-11..."
    echo "=========================================="
    
    python rq2_pilot_complete.py --task_id 10
    python rq2_pilot_complete.py --task_id 11
    
    echo ""
    echo "All 12 tasks complete. Merging results..."
    python rq2_pilot_complete.py --merge
    
    echo ""
    echo "DONE! Results in RQ2_PILOT_RESULTS/"
fi

