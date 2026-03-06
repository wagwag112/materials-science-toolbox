#!/usr/bin/env python3
"""
Author: Huy Hoang
Slice an ASE trajectory by frame indices and write the result to a new trajectory file.
If start and end are omitted, extract the last frame and write it in VASP format.

Usage:
    # Slice frames start:end (end = -1 means to the last frame)
    python extract_traj.py <input.traj> <output.file> <start> <end>

    # Provide input and output; extract last frame as VASP
    python extract_traj.py <input.traj> <output.vasp>

    # Provide only input; script auto-generates output name and writes last frame in VASP
    python extract_traj.py <input.traj>
"""
import sys
import os
from ase.io import Trajectory, write


def _unique_path(path: str) -> str:
    """
    If path exists, append a numeric suffix before the extension to make it unique.
    Example: POSCAR.vasp -> POSCAR_1.vasp -> POSCAR_2.vasp ...
    """
    base, ext = os.path.splitext(path)
    candidate = path
    i = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{i}{ext}"
        i += 1
    return candidate


def _auto_output_name(input_path: str) -> str:
    """
    Generate an output filename based on input_path:
    <input_basename>_last.vasp
    Ensure uniqueness to avoid overwriting existing files.
    """
    base = os.path.basename(input_path)
    name, _ = os.path.splitext(base)
    out_name = f"{name}_last.vasp"
    return _unique_path(out_name)


def slice_trajectory(input_path: str, output_path: str = None, start: int = None, end: int = None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")

    traj = Trajectory(input_path)
    total = len(traj)

    if total == 0:
        raise ValueError("Trajectory is empty")

    # Mode: no start/end provided -> extract last frame and write as VASP
    if start is None and end is None:
        last = traj[-1]  # Atoms object
        # If no output_path provided, auto-generate one
        if output_path is None or output_path.strip() == "":
            output_path = _auto_output_name(input_path)
        # Force VASP format regardless of output extension
        write(output_path, last, format="vasp")
        print("Status: Success")
        print(f"Source: {input_path} ({total} frames)")
        print(f"Action: extracted last frame -> {output_path} (VASP format)")
        return

    # Mode: slicing with provided indices
    if start is None or end is None:
        raise ValueError("Both start and end must be provided for slicing, or omit both to extract last frame")

    if start < 0 or start >= total:
        raise ValueError("start index out of range")

    if end == -1:
        end = total

    if end <= start or end > total:
        raise ValueError("end index out of range")

    # If output_path not provided for slicing, raise error (we require explicit output for slices)
    if output_path is None or output_path.strip() == "":
        raise ValueError("Output filename must be provided when slicing frames")

    subset = traj[start:end]
    write(output_path, subset)
    print("Status: Success")
    print(f"Source: {input_path} ({total} frames)")
    print(f"Sliced: frames {start}:{end} -> {output_path}")


if __name__ == "__main__":
    try:
        argc = len(sys.argv)
        # Acceptable invocations:
        # 1) python extract_traj.py <input.traj>                -> auto-generate output, extract last frame
        # 2) python extract_traj.py <input.traj> <output>      -> extract last frame, use provided output
        # 3) python extract_traj.py <input.traj> <output> <start> <end> -> slicing
        if argc not in (2, 3, 5):
            print("Usage:")
            print("  python extract_traj.py <input.traj>                     # auto-generate output, extract last frame (VASP)")
            print("  python extract_traj.py <input.traj> <output.file>       # extract last frame (VASP if desired)")
            print("  python extract_traj.py <input.traj> <output.file> <start> <end>  # slice frames (end=-1 means to last)")
            sys.exit(1)

        inp = sys.argv[1]

        if argc == 2:
            # Only input provided -> auto-generate output name and extract last frame
            slice_trajectory(inp)
        elif argc == 3:
            out = sys.argv[2]
            # Provided output but no indices -> extract last frame and write VASP
            slice_trajectory(inp, out)
        else:
            out = sys.argv[2]
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
