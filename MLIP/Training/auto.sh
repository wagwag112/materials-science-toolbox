#!/bin/bash

# Define directory paths
BASE_DIR=$(pwd)
TEMPLATE_DIR="$BASE_DIR/template"
RAW_DIR="$BASE_DIR/raw"
CALC_DIR="$BASE_DIR/cal"
RUN_SCRIPT="$BASE_DIR/run.sh"

# Create calculation root directory
mkdir -p "$CALC_DIR"

# Auto-detect sub-folders inside raw (no need to list them manually)
if [ ! -d "$RAW_DIR" ]; then
    echo "Warning: raw directory '$RAW_DIR' not found. Exiting."
    exit 1
fi

found_any=false
for dir in "$RAW_DIR"/*/; do
    [ -d "$dir" ] || continue
    found_any=true
    SUB_FOLDER=$(basename "$dir")
    SEARCH_PATH="$RAW_DIR/$SUB_FOLDER"

    echo "Processing folder: $SUB_FOLDER"

    # Use nullglob so the loop is skipped if no .vasp files exist
    shopt -s nullglob
    vasp_files=("$SEARCH_PATH"/*.vasp)
    shopt -u nullglob

    if [ ${#vasp_files[@]} -eq 0 ]; then
        echo "  No .vasp files found in $SEARCH_PATH, skipping..."
        continue
    fi

    for vasp_file in "${vasp_files[@]}"; do
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

if [ "$found_any" = false ]; then
    echo "Warning: no sub-folders found inside '$RAW_DIR'."
fi

echo "All jobs have been dispatched to the queue."
