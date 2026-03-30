import os
import sys
import numpy as np
from ase.io import read, write
import shutil

def generate_dataset(input_file, num_strain, num_rattle):
    # Load the base structure
    try:
        atoms = read(input_file)
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        return

    base_name = os.path.splitext(input_file)[0]
    
    # Create output directories
    strain_dir = f"strain_{base_name}"
    rattle_dir = f"rattle_{base_name}"
    
    for folder in [strain_dir, rattle_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    print(f"--- Generating {num_strain} strained structures ---")
    # Generate Strained structures (Volume scaling from 0.9 to 1.1)
    # This corresponds to linear scaling of ~0.965 to 1.032
    scales = np.linspace(0.96, 1.04, num_strain)
    for i, scale in enumerate(scales):
        strained_atoms = atoms.copy()
        current_cell = atoms.get_cell()
        strained_atoms.set_cell(current_cell * scale, scale_atoms=True)
        
        filename = os.path.join(strain_dir, f"POSCAR_strain_{i:03d}_{scale:.3f}.vasp")
        write(filename, strained_atoms, format='vasp')

    print(f"--- Generating {num_rattle} rattled structures ---")
    # Generate Rattled structures
    # Using a standard deviation of 0.05 Angstrom as recommended for MLIPs
    for i in range(num_rattle):
        rattled_atoms = atoms.copy()
        rattled_atoms.rattle(stdev=0.05, seed=i)
        
        filename = os.path.join(rattle_dir, f"POSCAR_rattle_{i:03d}.vasp")
        write(filename, rattled_atoms, format='vasp')

    print(f"\nDone! Files are in '{strain_dir}' and '{rattle_dir}' folders.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_data.py <input_file.vasp> <num_strain> <num_rattle>")
    else:
        file_input = sys.argv[1]
        n_strain = int(sys.argv[2])
        n_rattle = int(sys.argv[3])
        generate_dataset(file_input, n_strain, n_rattle)
