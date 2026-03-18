#!/usr/bin/env python3
"""
Merge two atomic structures along a specified axis with a defined gap.

Usage:
      python combine_slabs.py <slab1.vasp> <slab2.vasp> <axis> <gap>
"""

import sys
import os
import numpy as np
from ase.io import read, write
from ase.build import stack

def combine_structures(input_file1, input_file2, stack_axis, gap_distance):
    """
    Merges two atomic structures along a specified axis with a defined gap.
    Supports CIF, VASP, and XYZ. Outputs a VASP file.
    """
    try:
        # 1. Read input structures
        # ASE automatically detects the format (cif, vasp, xyz, etc.)
        atoms1 = read(input_file1)
        atoms2 = read(input_file2)

        # 2. Map axis string to index (x:0, y:1, z:2)
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map.get(stack_axis.lower())

        if axis_idx is None:
            print(f"Error: Invalid axis '{stack_axis}'. Please use x, y, or z.")
            sys.exit(1)

        print(f"--- Merging Process ---")
        print(f"File 1: {input_file1}")
        print(f"File 2: {input_file2}")
        print(f"Stacking Axis: {stack_axis.upper()}")
        print(f"Interface Gap: {gap_distance} Angstrom")

        # 3. Perform Stacking
        # 'distance' is the vacuum gap between the top of atoms1 and bottom of atoms2
        # 'maxstrain=None' allows merging even if there is a tiny numerical mismatch
        combined = stack(
            atoms1, 
            atoms2, 
            axis=axis_idx, 
            distance=gap_distance, 
            maxstrain=None
        )

        # 4. Export to VASP
        output_name = "POSCAR_COMBINED.vasp"
        write(output_name, combined, format='vasp', direct=True, sort=True)
        
        print(f"-----------------------")
        print(f"SUCCESS: Combined structure saved as '{output_name}'")

    except Exception as e:
        print(f"An error occurred during merging: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Command line argument handling
    # Expected: python combine_slabs.py <file1> <file2> <axis> <gap>
    if len(sys.argv) != 5:
        print("Usage: python combine_slabs.py <file1> <file2> <axis> <gap>")
        print("Example: python combine_slabs.py Li3N.vasp LIC.vasp z 1.8")
        sys.exit(1)

    f1 = sys.argv[1]
    f2 = sys.argv[2]
    ax = sys.argv[3]
    gap = float(sys.argv[4])

    combine_structures(f1, f2, ax, gap)
