#!/bin/bash
#SBATCH -A ACD114153
#SBATCH -J MD_GPU_Serial
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1     
#SBATCH --ntasks-per-node=1   
#SBATCH --cpus-per-task=4     
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# ==============================================================================
# 1. Environment Setup
# ==============================================================================
module purge
module load miniconda3/24.11.1

# IMPORTANT: Initialize conda properly for a non-interactive batch script
eval "$(conda shell.bash hook)"
conda activate /home/$USER/.conda/envs/fairchem_env

# ==============================================================================
# 2. Parse Arguments (to determine folder name dynamically)
# ==============================================================================
python interface_plots.py

echo "======================================================================"
echo "All done!"
echo "======================================================================"
