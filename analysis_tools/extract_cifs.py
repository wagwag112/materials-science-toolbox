#!/usr/bin/env python3
"""
Author: Huy Hoang
Random Trajectory Frame Extractor
Extract N random frames from an ASE .traj file and save them as .cif files.
Usage:
    python extract_random_cif.py input.traj output_folder N
"""

import sys
import os
import random
from ase.io import Trajectory, write

def extract_random_frames(input_path, output_dir, n_frames):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")

    os.makedirs(output_dir, exist_ok=True)

    traj = Trajectory(input_path)
    total = len(traj)

    if total == 0:
        raise ValueError("Trajectory is empty")

    count = min(n_frames, total)
    indices = random.sample(range(total), count)

    for i, idx in enumerate(indices):
        atoms = traj[idx]
        out_file = os.path.join(output_dir, f"frame_{i:04d}.cif")
        write(out_file, atoms)

    print("Status: Success")
    print(f"Source: {input_path} ({total} frames)")
    print(f"Extracted: {count} random frames -> {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_random_cif.py input.traj output_folder N")
        sys.exit(1)

    try:
        inp = sys.argv[1]
        out_dir = sys.argv[2]
        n = int(sys.argv[3])
        extract_random_frames(inp, out_dir, n)
    except ValueError:
        print("Error: N must be an integer")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
