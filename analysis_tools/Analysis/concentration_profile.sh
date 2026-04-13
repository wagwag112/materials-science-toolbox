#!/bin/bash
#SBATCH -A ACD114153
#SBATCH -J Density_Analysis
#SBATCH -p dev
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1      
#SBATCH --ntasks-per-node=1   
#SBATCH --cpus-per-task=4      
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# ==============================================================================
# 1. Environment Setup (Gi? nguyên nhu cu)
# ==============================================================================
module purge
module load miniconda3/24.11.1

eval "$(conda shell.bash hook)"
conda activate /home/$USER/.conda/envs/fairchem_env

# ==============================================================================
# 2. Parse Arguments
# ==============================================================================
TRAJ_FILE=$1
S_IDX=$2
E_IDX=$3

if [ -z "$S_IDX" ] || [ -z "$E_IDX" ]; then
    FOLDER_NAME="density_analysis_output"
else
    FOLDER_NAME="density_frames_${S_IDX}_${E_IDX}"
fi

echo "======================================================================"
echo "Starting Job ID: $SLURM_JOB_ID"
echo "Executing: python concentration_profile.py $@"
echo "======================================================================"

# ==============================================================================
# 3. Execute the Analysis Code
# ==============================================================================
python concentration_profile.py "$@"

# ==============================================================================
# 4. Post-processing: Organize Output Images
# ==============================================================================
echo "Analysis finished. Organizing plots..."

mkdir -p "$FOLDER_NAME"

mv analysis_*.png "$FOLDER_NAME/" 2>/dev/null

echo "======================================================================"
echo "All done! Output images are saved in: ${FOLDER_NAME}/"
echo "======================================================================"
