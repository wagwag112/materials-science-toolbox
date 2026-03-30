#!/bin/bash

# Define directory paths
BASE_DIR=$(pwd)
TEMPLATE_DIR="$BASE_DIR/template"
RAW_DIR="$BASE_DIR/raw"
CALC_DIR="$BASE_DIR/cal"
RUN_SCRIPT="$BASE_DIR/run.sh"

# Create calculation root directory
mkdir -p "$CALC_DIR"

# Sub-folders to process
FOLDERS=("strain" "rattle")

for SUB_FOLDER in "${FOLDERS[@]}"; do
    SEARCH_PATH="$RAW_DIR/$SUB_FOLDER"
    
    if [ ! -d "$SEARCH_PATH" ]; then
        echo "Warning: $SEARCH_PATH not found, skipping..."
        continue
    fi

    echo "Processing folder: $SUB_FOLDER"
    
    for vasp_file in "$SEARCH_PATH"/*.vasp; do
        # Create unique folder name for each structure
        FILE_NAME=$(basename "$vasp_file" .vasp)
        JOB_DIR="$CALC_DIR/${SUB_FOLDER}_${FILE_NAME}"
        
        mkdir -p "$JOB_DIR"
        
        # Copy VASP templates
        cp "$TEMPLATE_DIR/INCAR" "$JOB_DIR/"
        cp "$TEMPLATE_DIR/KPOINTS" "$JOB_DIR/"
        cp "$TEMPLATE_DIR/POTCAR" "$JOB_DIR/"
        
        # Copy geometry file as POSCAR
        cp "$vasp_file" "$JOB_DIR/POSCAR"
        
        # Copy and submit the SLURM script
        cp "$RUN_SCRIPT" "$JOB_DIR/"
        
        cd "$JOB_DIR"
        sbatch run.sh
        cd "$BASE_DIR"
        
        echo "Submitted job for: $FILE_NAME"
    done
done

echo "All jobs have been dispatched to the queue."
