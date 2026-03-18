#!/usr/bin/env python3
"""
Permute cell axes (x, y, z) and export to VASP POSCAR.

Usage:
    python permute_cell_axes.py input_file axis1 axis2 axis3

Example:
    python permute_cell_axes.py input.cif z y x
"""

import sys
import os
import numpy as np
from ase.io import read, write

def main():
    if len(sys.argv) != 5:
        print("Usage: python permute_cell_axes.py <input_file> <axis1> <axis2> <axis3>")
        sys.exit(1)

    input_path = sys.argv[1]
    new_order_str = [a.lower() for a in sys.argv[2:5]]
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    
    try:
        new_indices = [axis_map[a] for a in new_order_str]
    except KeyError:
        print("Error: Axes must be x, y, or z.")
        sys.exit(1)

    try:
        atoms = read(input_path)
        # 1. Get original cell and Cartesian positions
        orig_cell = atoms.get_cell()
        cartesian_pos = atoms.get_positions()

        # 2. Permute the cell vectors exactly as requested (z, y, x)
        new_cell_matrix = orig_cell[new_indices]

        # 3. Check handedness (determinant)
        if np.linalg.det(new_cell_matrix) < 0:
            print(f"  -> Order {new_order_str} is Left-handed. Flipping the first vector to fix.")
            # Flip the first vector of the new cell to make it Right-handed
            # This preserves the order (z,y,x) and the lengths, but changes direction
            new_cell_matrix[0] = -new_cell_matrix[0]

        # 4. Apply the new cell to atoms
        # We keep Cartesian positions the same, but update the lattice
        atoms.set_cell(new_cell_matrix, scale_atoms=False)
        
        # 5. Wrap atoms to ensure they are within the new 0 to 1 range
        atoms.wrap()

        output_file = f"{os.path.splitext(input_path)[0]}_z_y_x.vasp"
        write(output_file, atoms, format='vasp', direct=True)
        
        print(f"--- SUCCESS ---")
        print(f"New Lattice Order: {new_order_str}")
        print(f"New a-length: {np.linalg.norm(new_cell_matrix[0]):.4f} A (was {new_order_str[0]})")
        print(f"New b-length: {np.linalg.norm(new_cell_matrix[1]):.4f} A (was {new_order_str[1]})")
        print(f"New c-length: {np.linalg.norm(new_cell_matrix[2]):.4f} A (was {new_order_str[2]})")
        print(f"Saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
