#!/usr/bin/env python3
import os
import numpy as np
from ase.io import read, write
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter

# FAIRChem imports
try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
except ImportError:
    print("Warning: FAIRChem not found. Ensure you have 'fairchem-core' installed.")

# ==========================================
# RELAX Function (Modified: No Supercell)
# ==========================================
def relax_structure(input_file, output_file, calc):
    # --- Read structure ---
    try:
        atoms = read(input_file)
    except Exception as e:
        print(f"ERROR reading file {input_file}: {e}")
        return

    # --- Attach Calculator ---
    atoms.calc = calc
    
    print(f"   -> Structure loaded. Atoms: {len(atoms)}. Volume: {atoms.get_volume():.2f} A^3")

    # --- SETUP variable to save the best state ---
    best_energy = float('inf')
    best_positions = None
    best_cell = None
    
    def log_best_step(a=atoms):
        nonlocal best_energy, best_positions, best_cell
        
        try:
            current_energy = a.calc.results.get('energy')
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = a.get_positions().copy() 
                best_cell = a.get_cell().copy()
        except Exception:
            pass 

    # --- Run LBFGS relax ---
    ucf = FrechetCellFilter(atoms)
    
    base_name = os.path.basename(output_file)
    log_name = f"relax_{base_name}.log"
    
    optimizer = LBFGS(ucf, logfile=log_name)
    optimizer.attach(log_best_step)

    print("   -> Starting relaxation (Unit Cell)...")
    try:
        optimizer.run(fmax=0.05, steps=100)
        
        if best_positions is not None and best_cell is not None:
            final_energy = atoms.get_potential_energy()
            
            if best_energy < final_energy:
                print(f"   ->  Not fully converged. Reverting to best step (E={best_energy:.4f} vs Final={final_energy:.4f})")
                atoms.set_cell(best_cell)
                atoms.set_positions(best_positions)
            else:
                print(f"   ->  Converged or ended at best state (E={final_energy:.4f})")
        
        print("   -> Relaxation DONE.")
        
    except Exception as e:
        print(f"ERROR during optimization {input_file}: {e}")
        if best_positions is not None:
             print("Attempting to save the best structure found before crash...")
             atoms.set_cell(best_cell)
             atoms.set_positions(best_positions)
        else:
             return

    # --- Save CIF output ---
    try:
        write(output_file, atoms, format='cif')
        print(f"Saved relaxed structure to {output_file}\n")
    except Exception as e:
        print(f"Error saving file: {e}")

# ==========================================
# MAIN
# ==========================================
def main():
    INPUT_FOLDER = "./cifs"
    
    print(" Loading UMA model...")
    try:
        # Load model FAIRChem
        calc = FAIRChemCalculator.from_model_checkpoint(
    		name_or_path="/home/hoang0000/inference_ckpt.pt",
    		task_name="omat",
    		device="cuda"
    )
        print(" Model loaded successfully.")
    except Exception as e:
        print(f" Failed to load model: {e}")
        return

    if not os.path.exists(INPUT_FOLDER):
        print(f"Folder {INPUT_FOLDER} not found.")
        return

    cif_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f == "POSCAR" or f.lower().endswith((".vasp", ".cif", ".xyz"))
    ]
    if not cif_files:
        print("No CIF/ VASP/ XYZ / POSCAR files found!")
        return

    print(f"Found {len(cif_files)} CIF files.")

    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_ROOT = os.path.dirname(CODE_DIR)  

    for cif in cif_files:
        input_path = os.path.join(INPUT_FOLDER, cif)
        base_name = os.path.splitext(cif)[0]
        
        output_dir = os.path.join(OUTPUT_ROOT, base_name)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{base_name}_relaxed.cif")

        print(f"\n=== Processing {cif} ===")
        relax_structure(input_path, output_path, calc)

if __name__ == "__main__":
    main()
