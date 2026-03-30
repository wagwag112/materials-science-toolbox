#!/bin/bash
#SBATCH -A ACD114153
#SBATCH -J MD_GPU_Serial
#SBATCH -p normal2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1     
#SBATCH --ntasks-per-node=1   
#SBATCH --cpus-per-task=4     
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# --- Environment Setup ---
module purge
module load miniconda3/24.11.1
conda deactivate
conda activate /home/$USER/.conda/envs/fairchem_env

# --- Simulation Logic ---
total_time_ps=$1
shift
temps=("$@")

mkdir -p logs results

echo "Launching MD simulations sequentially on 1 GPU..."
echo "Total simulation time = ${total_time_ps} ps"

for T in "${temps[@]}"; do
  echo "-> Starting MD at ${T}K on GPU 0"
  python md.py "$total_time_ps" "$T" > logs/${T}.out 2> logs/${T}.err
  echo "-> Finished MD at ${T}K"
done

echo "All simulations done."
