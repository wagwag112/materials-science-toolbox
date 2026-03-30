#!/usr/bin/env python3
import os
import sys

# ASE imports
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# FAIRChem imports
from fairchem.core import FAIRChemCalculator

# --- Global Settings ---
# Automatically find the first available structure file
SUPPORTED_FORMATS = ["POSCAR", ".vasp", ".cif", ".xyz"]
INPUT_FILE = next(
    (f for f in os.listdir(".") 
     if f in ["POSCAR"] or any(f.lower().endswith(ext) for ext in SUPPORTED_FORMATS if ext.startswith("."))), 
    None
)

TIME_STEP_FS = 2.0
LOG_INTERVAL = 50
FRICTION = 0.001
CHECKPOINT_PATH = "/home/hoang0000/inference_ckpt.pt"

def run_single_temp(additional_ps, temperature_k):
    if INPUT_FILE is None:
        print("Error: No input structure file found in the current directory.")
        return

    steps_to_add = int(additional_ps * 1000 / TIME_STEP_FS)
    traj_file = f"md_{temperature_k}K.traj"
    log_file = f"md_{temperature_k}K.log"
    
    atoms = None
    steps_already_done = 0

    # 1. Check for restart (Append Mode logic)
    if os.path.exists(traj_file) and os.path.getsize(traj_file) > 0:
        try:
            # Read all frames to calculate current step count and get the last state
            frames = ase_read(traj_file, index=":")
            if frames:
                steps_already_done = len(frames) * LOG_INTERVAL
                atoms = frames[-1]
                print(f"\n--- Restarting {temperature_k}K ---")
                print(f"Existing trajectory found: {steps_already_done} steps.")
        except Exception as e:
            print(f"Warning: Could not read {traj_file}: {e}")
            os.rename(traj_file, f"{traj_file}.bak")

    # 2. Start fresh if no valid trajectory found
    if atoms is None:
        print(f"\n--- Starting Fresh {temperature_k}K ---")
        print(f"Loading initial structure: {INPUT_FILE}")
        atoms = ase_read(INPUT_FILE)
        # Initialize velocities for the target temperature
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k)

    # 3. Setup FAIRChem Calculator
    calc = FAIRChemCalculator.from_model_checkpoint(
        name_or_path=CHECKPOINT_PATH,
        task_name="omat",
        device="cuda"
    )
    atoms.calc = calc

    # 4. Setup Molecular Dynamics
    dyn = Langevin(
        atoms,
        timestep=TIME_STEP_FS * units.fs,
        temperature_K=temperature_k,
        friction=FRICTION,
        logfile=log_file,
        loginterval=LOG_INTERVAL,
        append_trajectory=True # Ensures .log file appends instead of overwriting
    )

    # Attach trajectory in append mode ("a")
    traj = Trajectory(traj_file, "a", atoms)
    dyn.attach(traj.write, interval=LOG_INTERVAL)

    print(f"Target: Adding {additional_ps} ps ({steps_to_add} steps)")
    print(f"Final expected length: {steps_already_done + steps_to_add} steps")
    
    # Run simulation
    dyn.run(steps_to_add)
    
    # Close trajectory file to prevent corruption
    traj.close()
    print(f"Completed {temperature_k}K simulation.")

def main():
    # Expected usage: python script.py <ps> <temp1> <temp2> ...
    if len(sys.argv) < 3:
        print("Usage: python script.py <additional_ps> <temp1> [temp2] [temp3] ...")
        sys.exit(1)

    try:
        additional_ps = float(sys.argv[1])
        temperatures = [int(t) for t in sys.argv[2:]]
    except ValueError:
        print("Error: Please provide numbers for ps and temperatures.")
        sys.exit(1)

    print(f"Job Initialized: Adding {additional_ps} ps to {len(temperatures)} temperature(s).")

    for temp in temperatures:
        run_single_temp(additional_ps, temp)

    print("\nAll tasks completed successfully.")

if __name__ == "__main__":
    main()
