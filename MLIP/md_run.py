#!/usr/bin/env python3
"""
Author: Huy Hoang
Description: Machine Learning Molecular Dynamics (MLMD) engine optimized for 
             sequential HPC execution. Features target-time restart logic and 
             automatic output routing to specific results directories.
Usage:
    python md_run.py <temperature_K> <total_time_ps>
    (This is typically called within a SLURM loop)
"""

import os
import sys
import argparse
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase import units

# --- FAIRChem Imports ---
try:
    from fairchem.core import FAIRChemCalculator
except ImportError:
    print("Error: FAIRChem core not found. Please ensure the environment is active.")
    sys.exit(1)

# --- Hyperparameters ---
TIME_STEP_FS = 2.0
LOG_INTERVAL = 50
CHECKPOINT_PATH = "/home/yourcheckpoint.pt"

def run_md_simulation(temperature_k, total_time_ps):
    """
    Executes NVT Langevin MD with target-time restart logic.
    Outputs are saved in the 'results/' directory.
    """
    total_steps = int(total_time_ps * 1000 / TIME_STEP_FS)
    
    # Matching SLURM's folder structure
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    traj_file = os.path.join(output_dir, f"md_{temperature_k}K.traj")
    log_file = os.path.join(output_dir, f"md_{temperature_k}K.log")

    print(f"--- Protocol: {temperature_k}K | Target Total Time: {total_time_ps} ps ---")

    # 1. RESTART LOGIC: Check existing results in the results/ folder
    atoms = None
    steps_done = 0
    if os.path.exists(traj_file) and os.path.getsize(traj_file) > 0:
        try:
            frames = ase_read(traj_file, index=":")
            if frames:
                # Calculate finished steps based on frames in trajectory
                steps_done = (len(frames) - 1) * LOG_INTERVAL
                atoms = frames[-1]
                print(f"--> Found existing trajectory: {len(frames)} frames found.")
                print(f"--> Progress: {steps_done}/{total_steps} steps.")
        except Exception as e:
            print(f"--> Warning: Trajectory error. Starting fresh. {e}")
            os.rename(traj_file, f"{traj_file}.bak")

    # 2. INITIAL STRUCTURE: Scans current directory for valid inputs
    if atoms is None:
        valid_exts = (".cif", ".vasp")
        valid_names = ("POSCAR", "CONTCAR")
        files = [f for f in os.listdir(".") if f.lower().endswith(valid_exts) or f.upper() in valid_names]
        
        if not files:
            print("Error: No initial structure file found in current directory.")
            return
        
        input_file = files[0]
        print(f"--> Loading initial structure: {input_file}")
        atoms = ase_read(input_file)

    # 3. CHECK COMPLETION
    steps_left = total_steps - steps_done
    if steps_left <= 0:
        print(f"--> Target total time of {total_time_ps} ps already reached. Skipping.")
        return

    # 4. CALCULATOR: Initialize FAIRChem MLIP
    print("--> Initializing FAIRChem Calculator...")
    try:
        calc = FAIRChemCalculator.from_model_checkpoint(
            name_or_path=CHECKPOINT_PATH,
            task_name="omat",
            device="cuda"
        )
        atoms.calc = calc
    except Exception as e:
        print(f"--> Model Error: {e}")
        return

    # 5. DYNAMICS: Langevin NVT
    dyn = Langevin(
        atoms,
        timestep=TIME_STEP_FS * units.fs,
        temperature_K=temperature_k,
        friction=0.01,
        logfile=log_file,
        loginterval=LOG_INTERVAL
    )

    # Attach Trajectory saver (Append mode)
    traj_writer = Trajectory(traj_file, "a", atoms)
    dyn.attach(traj_writer.write, interval=LOG_INTERVAL)

    # 6. RUN
    print(f"--> Starting execution for {steps_left} remaining steps...")
    try:
        dyn.run(steps_left)
        print(f"--> Successfully completed {temperature_k}K run.")
    except Exception as e:
        print(f"--> Dynamics failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="FAIRChem MD Runner")
    # Matches SLURM's call order: python md_run.py "$T" "$total_time_ps"
    parser.add_argument("temperature", type=int, help="Temperature in K")
    parser.add_argument("total_time_ps", type=int, help="Target total time in ps")
    
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    run_md_simulation(args.temperature, args.total_time_ps)

if __name__ == "__main__":
    main()
