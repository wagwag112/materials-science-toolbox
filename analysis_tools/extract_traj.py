#!/usr/bin/env python3
"""
Author: Huy Hoang
Slice an ASE trajectory by frame indices and write the result to a new trajectory file.
If start and end are omitted, extract the last frame and write it in VASP format.

Usage:
    # Slice frames start:end (end = -1 means to the last frame)
    python extract_traj.py <input.traj> <output.file> <start> <end>

    # If start and end are omitted, extract the last frame and write as VASP
    python extract_traj.py <input.traj> <output.vasp>
"""
import sys
import os
from ase.io import Trajectory, write
from ase import Atoms


def slice_trajectory(input_path: str, output_path: str, start: int = None, end: int = None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")

    traj = Trajectory(input_path)
    total = len(traj)

    if total == 0:
        raise ValueError("Trajectory is empty")

    # Mode 1: no start/end provided -> extract last frame and write as VASP
    if start is None and end is None:
        last = traj[-1]  # Atoms object
        # Force VASP format regardless of output extension
        write(output_path, last, format="vasp")
        print("Status: Success")
        print(f"Source: {input_path} ({total} frames)")
        print(f"Action: extracted last frame -> {output_path} (VASP format)")
        return

    # Mode 2: slicing with provided indices
    if start is None or end is None:
        raise ValueError("Both start and end must be provided for slicing, or omit both to extract last frame")

    if start < 0 or start >= total:
        raise ValueError("start index out of range")

    if end == -1:
        end = total

    if end <= start or end > total:
        raise ValueError("end index out of range")

    # traj[start:end] returns a list of Atoms objects; write will infer format from filename
    subset = traj[start:end]
    write(output_path, subset)
    print("Status: Success")
    print(f"Source: {input_path} ({total} frames)")
    print(f"Sliced: frames {start}:{end} -> {output_path}")


if __name__ == "__main__":
    try:
        argc = len(sys.argv)
        if argc not in (3, 5):
            print("Usage:")
            print("  python extract_traj.py <input.traj> <output.file> <start> <end>")
            print("  python extract_traj.py <input.traj> <output.vasp>   # extracts last frame as VASP")
            sys.exit(1)

        inp = sys.argv[1]
        out = sys.argv[2]

        if argc == 3:
            # No indices provided -> extract last frame and write VASP
            slice_trajectory(inp, out)
        else:
            start_idx = int(sys.argv[3])
            end_idx = int(sys.argv[4])
            slice_trajectory(inp, out, start_idx, end_idx)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
