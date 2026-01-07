#!/usr/bin/env python3
"""
Author: Huy Hoang
Slice an ASE trajectory by frame indices and write the result to a new trajectory file.
Usage:
    python slice_traj_frames.py <input.traj> <output.traj> <start> <end>
"""

import sys
import os
from ase.io import Trajectory, write


def slice_trajectory(input_path, output_path, start, end):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")

    traj = Trajectory(input_path)
    total = len(traj)

    if total == 0:
        raise ValueError("Trajectory is empty")

    if start < 0 or start >= total:
        raise ValueError("start index out of range")

    if end == -1:
        end = total

    if end <= start or end > total:
        raise ValueError("end index out of range")

    subset = traj[start:end]
    write(output_path, subset)

    print("Status: Success")
    print(f"Source: {input_path} ({total} frames)")
    print(f"Sliced: frames {start}:{end} -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python slice_traj_frames.py <input.traj> <output.traj> <start> <end>")
        sys.exit(1)

    try:
        inp = sys.argv[1]
        out = sys.argv[2]
        start_idx = int(sys.argv[3])
        end_idx = int(sys.argv[4])

        slice_trajectory(inp, out, start_idx, end_idx)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
