#!/usr/bin/env python3
"""
Author: Huy Hoang
Compute the shortest distance from each el1 atom to el2 atoms for multi-frame atomistic trajectories.
Usage: python calc_interatomic_dist.py <input_file> <el1> <el2> <output.csv>
Supported formats: .traj, .cif, .xml, POSCAR, OUTCAR, .xyz, etc.
"""

import sys
import csv
import os
from ase.io import read

def calculate_distances(input_path, el1, el2, output_csv):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return

    try:
        frames = read(input_path, index=":")
        print(f"Status: Loaded {len(frames)} structure(s) from {input_path}")
        
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "index_1", "index_2", "distance"])

            for frame_idx, atoms in enumerate(frames):
                symbols = atoms.get_chemical_symbols()
                idx1 = [i for i, s in enumerate(symbols) if s == el1]
                idx2 = [i for i, s in enumerate(symbols) if s == el2]

                if not idx1 or not idx2:
                    print(f"Warning: Elements {el1} or {el2} not found in frame {frame_idx}")
                    continue

                for i in idx1:
                    dists = atoms.get_distances(i, idx2, mic=True)
                    min_idx = dists.argmin()
                    writer.writerow([frame_idx, i, idx2[min_idx], dists[min_idx]])

        print(f"Status: Success")
        print(f"Output: {output_csv}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python calc_interatomic_dist.py <input_file> <el1> <el2> <output.csv>")
        sys.exit(1)

    calculate_distances(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
