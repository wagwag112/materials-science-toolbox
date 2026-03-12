import sys
import ase.io as io

"""
DESCRIPTION:
    Sub-samples frames from an .traj file using a specified stride. The output is saved in .extxyz format.

USAGE:
    python sub_sample_traj.py <input_file.traj> <stride>
"""

def main():
    # --- 1. ARGUMENT PARSING ---
    if len(sys.argv) < 3:
        print("ERROR: Missing arguments.")
        print("Usage: python sub_sample_traj.py <input_file.traj> <stride>")
        sys.exit(1)

    input_file = sys.argv[1]
    
    try:
        stride = int(sys.argv[2])
    except ValueError:
        print("ERROR: Stride must be an integer.")
        sys.exit(1)

    output_file = input_file.replace('.traj', '_sub.extxyz')

    # --- 2. READING DATA ---
    print(f"Reading {input_file} with stride {stride}...")
    try:
        # index='::stride' reads every n-th frame efficiently without loading everything into memory
        frames = io.read(input_file, index=f'::{stride}')
    except Exception as e:
        print(f"ERROR reading file: {e}")
        sys.exit(1)

    # --- 3. WRITING DATA ---
    print(f"Writing {len(frames)} frames to {output_file}...")
    # ASE automatically recognizes .extxyz and includes all properties (charges, forces)
    try:
        io.write(output_file, frames)
        print(f"Done! Created: {output_file}")
    except Exception as e:
        print(f"ERROR writing file: {e}")

if __name__ == "__main__":
    main()
