#!/usr/bin/env python3
"""
Trajectory Frame Extractor
Author: Huy Hoang
Usage: python extract_traj.py <input.traj> <output.traj> <n_frames>
"""

import sys
import os
from ase.io import Trajectory, write

def extract_frames(input_path, output_path, n_frames):
    try:
        if not os.path.exists(input_path):
            print(f"Error: {input_path} not found")
            return

        traj = Trajectory(input_path)
        total = len(traj)
        
        # Determine number of frames to extract
        count = min(n_frames, total)

        # Extract and write
        subset = traj[:count]
        write(output_path, subset)
        
        # Minimalist output for HPC logs
        print(f"Status: Success")
        print(f"Source: {input_path} ({total} frames)")
        print(f"Extracted: {count} frames -> {output_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check usage
    if len(sys.argv) != 4:
        print("Usage: python extract_traj.py <input.traj> <output.traj> <n_frames>")
        sys.exit(1)

    # Parse arguments
    try:
        inp_file = sys.argv[1]
        out_file = sys.argv[2]
        num = int(sys.argv[3])
        extract_frames(inp_file, out_file, num)
    except ValueError:
        print("Error: n_frames must be an integer")
        sys.exit(1)
