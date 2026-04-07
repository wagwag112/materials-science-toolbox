#!/usr/bin/env python3
"""
Author: Huy Hoang
Extract frames from an ASE trajectory file.

Two modes:

  traj mode  -- slice a range of frames and write a new trajectory file.
                Output name is auto-generated as <input_base>_<start>-<end>.traj

  vasp mode  -- extract a single frame and write it in VASP format.
                Output name is auto-generated as <input_base>_frame<N>.vasp
                Use -1 to extract the last frame.

Usage:
    python extract_traj.py traj <input.traj> <start> <end>
    python extract_traj.py vasp <input.traj> <frame_index>
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


def _base_name(input_path: str) -> str:
    """Return the stem of the input filename (no directory, no extension)."""
    return os.path.splitext(os.path.basename(input_path))[0]


def mode_traj(input_path: str, start: int, end: int):
    """
    Slice frames [start, end) from input_path and write to a new .traj file.
    end = -1 means up to the last frame.
    Output name: <input_base>_<start>-<end>.traj
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")

    traj = Trajectory(input_path)
    total = len(traj)

    if total == 0:
        raise ValueError("Trajectory is empty")

    if start < 0 or start >= total:
        raise ValueError("start index out of range")

    resolved_end = total if end == -1 else end

    if resolved_end <= start or resolved_end > total:
        raise ValueError("end index out of range")

    out_name = _unique_path(f"{_base_name(input_path)}_{start}-{resolved_end}.traj")
    subset = traj[start:resolved_end]
    write(out_name, subset)

    print("Status: Success")
    print(f"Mode:   traj")
    print(f"Source: {input_path} ({total} frames)")
    print(f"Sliced: frames {start}:{resolved_end} -> {out_name}")


def mode_vasp(input_path: str, frame_index: int):
    """
    Extract a single frame from input_path and write it in VASP format.
    frame_index = -1 extracts the last frame.
    Output name: <input_base>_frame<N>.vasp  (N is the resolved index)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")

    traj = Trajectory(input_path)
    total = len(traj)

    if total == 0:
        raise ValueError("Trajectory is empty")

    resolved = (total - 1) if frame_index == -1 else frame_index

    if resolved < 0 or resolved >= total:
        raise ValueError("frame_index out of range")

    out_name = _unique_path(f"{_base_name(input_path)}_frame{resolved}.vasp")
    write(out_name, traj[resolved], format="vasp")

    print("Status: Success")
    print(f"Mode:   vasp")
    print(f"Source: {input_path} ({total} frames)")
    print(f"Frame:  {resolved} -> {out_name} (VASP format)")


if __name__ == "__main__":
    try:
        argc = len(sys.argv)

        if argc < 2 or sys.argv[1] not in ("traj", "vasp"):
            print("Usage:")
            print("  python extract_traj.py traj <input.traj> <start> <end>")
            print("  python extract_traj.py vasp <input.traj> <frame_index>")
            sys.exit(1)

        mode = sys.argv[1]

        if mode == "traj":
            if argc != 5:
                print("Usage: python extract_traj.py traj <input.traj> <start> <end>")
                sys.exit(1)
            mode_traj(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

        elif mode == "vasp":
            if argc != 4:
                print("Usage: python extract_traj.py vasp <input.traj> <frame_index>")
                sys.exit(1)
            mode_vasp(sys.argv[2], int(sys.argv[3]))

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
