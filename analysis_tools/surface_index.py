#!/usr/bin/env python3
"""
Author: Huy Hoang
Select atom indices located within a given thickness from both boundaries
of a simulation cell along a specified Cartesian axis.

Usage:
    python surface_index.py axis thickness_A
"""

import sys
import ase.io as io

INPUT_STRUCTURE = "POSCAR"

if len(sys.argv) != 3:
    print("Usage: python find_surface_atoms.py axis thickness_A")
    sys.exit(1)

axis = sys.argv[1].lower()
X_LIMIT = float(sys.argv[2])

axis_map = {"x":0, "y":1, "z":2}

if axis not in axis_map:
    print("Axis must be x, y, or z")
    sys.exit(1)

ax = axis_map[axis]

# --- LOAD STRUCTURE ---
atoms = io.read(INPUT_STRUCTURE)

pos = atoms.get_positions()
cell = atoms.get_cell().lengths()

coord = pos[:,ax]
L = cell[ax]

# --- FIND ATOMS ---
selected_index = [
    i for i in range(len(atoms))
    if (coord[i] <= X_LIMIT) or (coord[i] >= L - X_LIMIT)
]

# --- OUTPUT ---
print(f"Axis: {axis}")
print(f"Thickness: {X_LIMIT} A")
print(f"Selected atoms: {len(selected_index)}")
print(selected_index)
