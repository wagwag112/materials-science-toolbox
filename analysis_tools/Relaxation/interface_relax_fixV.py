import sys
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
import os

# --- MLIP Integration (FAIRChem/UMA) ---
try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
except ImportError:
    print("Error: FAIRChem not found. Please install 'fairchem-core'.")
    sys.exit(1)


def apply_interface_constraints(atoms, z_interface, relax_range=6.0):
    """Fix atoms outside active interface region."""
    z_coords = atoms.positions[:, 2]
    mask = (z_coords < (z_interface - relax_range)) | (z_coords > (z_interface + relax_range))
    indices = [i for i, m in enumerate(mask) if m]
    print(f"  -> Interface Constraints: {len(indices)} atoms fixed outside +/- {relax_range} A zone.")
    return indices


def apply_tip_constraints(atoms, out_axis="z", tip_thickness=5):
    """Fix atoms near slab boundaries."""
    axis_idx = {"x": 0, "y": 1, "z": 2}[out_axis]
    coords = atoms.positions[:, axis_idx]
    c_min, c_max = coords.min(), coords.max()
    mask = (coords < (c_min + tip_thickness)) | (coords > (c_max - tip_thickness))
    indices = [i for i, m in enumerate(mask) if m]
    print(f"  -> Tip Constraints: {len(indices)} atoms fixed within {tip_thickness} A of slab ends.")
    return indices


def main():
    # --- FIX #4: Input validation ---
    if len(sys.argv) < 3:
        print("Usage: python relax_interface.py <interface_file> <slab_a_thickness_angstrom>")
        print("  e.g: python relax_interface.py INTERFACE.vasp 12.5")
        sys.exit(1)

    file_interface = sys.argv[1]

    # --- FIX #4: File existence check ---
    if not os.path.exists(file_interface):
        print(f"Error: File not found: {file_interface}")
        sys.exit(1)

    # --- FIX #1: slab A thickness from CLI ---
    try:
        z_a = float(sys.argv[2])
    except ValueError:
        print(f"Error: slab_a_thickness must be a number, got '{sys.argv[2]}'")
        sys.exit(1)

    # 1. Load pre-built interface structure
    final_struct = read(file_interface)

    # 2. ML model setup
    print("  -> Initializing UMA fine-tuned model")
    calc = FAIRChemCalculator.from_model_checkpoint(
        name_or_path="/home/hoang0000/inference_ckpt.pt",
        task_name="omat",
        device="cuda"
    )

    # --- FIX #1: z_interface from slab A thickness, not midpoint ---
    z_coords = final_struct.positions[:, 2]
    z_interface = z_coords.min() + z_a
    print(f"  -> Slab A thickness: {z_a:.2f} A | z_interface: {z_interface:.2f} A")

    # 4. Apply constraints
    idx_interface = apply_interface_constraints(final_struct, z_interface, relax_range=6.0)
    idx_tips = apply_tip_constraints(final_struct, "z", tip_thickness=5)
    final_indices = list(set(idx_interface) | set(idx_tips))

    # --- FIX #2: Free atom check ---
    n_free = len(final_struct) - len(final_indices)
    print(f"  -> Free atoms to relax: {n_free} / {len(final_struct)}")
    if n_free == 0:
        print("  -> WARNING: No free atoms! Slab too thin or relax_range too large. Exiting.")
        sys.exit(1)

    final_struct.set_constraint(FixAtoms(indices=final_indices))

    # 5. Best-energy tracking
    final_struct.calc = calc
    best_energy = float("inf")
    best_positions = None

    # --- FIX #3: Read from calculator cache, avoid double compute ---
    def log_best_step(a=final_struct):
        nonlocal best_energy, best_positions
        try:
            e = a.calc.results.get("energy", None)
            if e is not None and e < best_energy:
                best_energy = e
                best_positions = a.get_positions().copy()
        except:
            pass

    # 6. Phase 1: interface relaxation
    print("  -> Phase 1: Interface relaxation (fmax=0.1, steps=100)...")
    dyn = LBFGS(final_struct, logfile="relax_interface.log")
    dyn.attach(log_best_step, interval=1)
    dyn.run(fmax=0.1, steps=100)

    # 7. Phase 2: keep only anchors
    print("  -> Phase 2: Anchor-only constraint relaxation...")
    idx_anchors_only = apply_tip_constraints(final_struct, "z", tip_thickness=5)
    final_struct.set_constraint(FixAtoms(indices=idx_anchors_only))

    print("  -> Final relaxation (fmax=0.05, steps=200)...")
    dyn_final = LBFGS(final_struct, logfile="relax_full_slab.log")
    dyn_final.attach(log_best_step, interval=1)

    try:
        dyn_final.run(fmax=0.05, steps=200)
    except Exception as e:
        print(f"  -> Error: {e}")

    # 8. Rollback to best structure
    if best_positions is not None:
        final_e = final_struct.get_potential_energy()
        if best_energy < final_e:
            print(f"  -> Reverting to best state: E={best_energy:.4f} eV")
            final_struct.set_positions(best_positions)
        else:
            print(f"  -> Final state is already best: E={final_e:.4f} eV")

    # 9. Export
    input_name = os.path.basename(file_interface)
    name_no_ext = os.path.splitext(input_name)[0]
    output_name = f"{name_no_ext}_relaxed.vasp"

    write(output_name, final_struct, format="vasp")
    print(f"  -> SUCCESS: saved {output_name}")


if __name__ == "__main__":
    main()
