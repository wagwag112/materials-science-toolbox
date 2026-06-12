#!/bin/bash

# Define directory paths
BASE_DIR=$(pwd)
TEMPLATE_DIR="$BASE_DIR/template"
RAW_DIR="$BASE_DIR/raw"
CALC_DIR="$BASE_DIR/cal"
RUN_SCRIPT="$BASE_DIR/run.sh"

# Create calculation root directory
mkdir -p "$CALC_DIR"

# Check if RAW_DIR exists
if [ ! -d "$RAW_DIR" ]; then
    echo "Error: Raw directory $RAW_DIR does not exist!"
    exit 1
fi

# Automatically detect all subdirectories inside RAW_DIR
for SEARCH_PATH in "$RAW_DIR"/*; do
    
    # Check if it is actually a directory
    if [ ! -d "$SEARCH_PATH" ]; then
        continue
    fi

    # Extract the sub-folder name from the full path
    SUB_FOLDER=$(basename "$SEARCH_PATH")

    echo "========================================"
    echo "Processing folder: $SUB_FOLDER"
    echo "========================================"
    
    # Check if there are any .vasp files inside before processing
    # This avoids throwing errors if a sub-folder is empty
    if [ -n "$(find "$SEARCH_PATH" -maxdepth 1 -name "*.vasp" -print -quit)" ]; then
        
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
            
            cd "$JOB_DIR" || continue
            sbatch run.sh
            cd "$BASE_DIR" || exit
            
            echo "Submitted job for: ${SUB_FOLDER}_${FILE_NAME}"
        done
    else
        echo "No .vasp files found in $SUB_FOLDER, skipping..."
    fi
done

echo "All jobs have been dispatched to the queue."
