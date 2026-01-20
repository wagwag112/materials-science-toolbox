#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Huy Hoang
Description:
    Calculates Mean Square Displacement (MSD) components (x, y, z, total) 
    from files (.traj), computes diffusion coefficients, 
    and generates Arrhenius plots to estimate activation energy (Ea).

Usage:
    python3 analysis_3D.py <start_fit_ps> <end_fit_ps> <temp1> <temp2> ...
    Example: python3 plot_summary.py 10 100 300 400 500
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from ase.io import read
import matplotlib

# Set backend for non-interactive environments (useful for clusters/SSH)
matplotlib.use('Agg')

# --- Configuration Section ---
CACHE_FILE = "diffusion_data_cache.json"
TIMESTEP_FS = 2.0
LOG_INTERVAL = 50
TARGET_SPECIES = "Li"

# --- Constants ---
k_B = 1.380649e-23  # Boltzmann constant (J/K)
e_charge = 1.602176634e-19  # Elementary charge (C)

# --- Plot Style Configuration ---
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "font.family": "sans-serif",
    "lines.linewidth": 2.5
})


def load_trajectory_safe(filename):
    """Reads atoms trajectory from file with error handling."""
    try:
        traj = read(filename, index=":")
        if not traj:
            sys.exit(f"Error: No frames found in {filename}")
        return traj
    except Exception as e:
        sys.exit(f"Fatal Error: Could not read {filename}: {e}")


def get_unwrapped_positions(frames, target_symbol):
    """Extracts unwrapped coordinates for the target atomic species."""
    print(f"  -> Extracting and unwrapping {target_symbol} positions...")
    indices = [atom.index for atom in frames[0] if atom.symbol == target_symbol]
    if not indices:
        sys.exit(f"Error: Symbol {target_symbol} not found.")

    n_frames = len(frames)
    n_atoms = len(indices)
    positions = np.zeros((n_frames, n_atoms, 3))

    for i, frame in enumerate(frames):
        positions[i] = frame.get_positions(wrap=False)[indices]
    
    return positions


def compute_msd_components(positions):
    """Computes ensemble-averaged MSD for each Cartesian component."""
    print("  -> Computing Ensemble MSD components...")
    r0 = positions[0]
    displacements = positions - r0
    sq_dist = displacements**2
    
    msd_x = np.mean(sq_dist[:, :, 0], axis=1)
    msd_y = np.mean(sq_dist[:, :, 1], axis=1)
    msd_z = np.mean(sq_dist[:, :, 2], axis=1)
    msd_total = np.mean(np.sum(sq_dist, axis=2), axis=1)
    
    return msd_x, msd_y, msd_z, msd_total


def fit_diffusivity(times, msd, start_ps, end_ps, is_total=True):
    """Performs linear regression on MSD data to get diffusion coefficient (D)."""
    idx_start = np.searchsorted(times, start_ps)
    idx_end = np.searchsorted(times, end_ps)
    
    x = times[idx_start:idx_end]
    y = msd[idx_start:idx_end]
    
    if len(x) < 2:
        return 0.0, 0.0, 0.0, []

    slope, intercept, r_val, _, std_err = linregress(x, y)
    
    # D = slope / (2 * dim * dt) -> 1e-4 converts A^2/ps to cm^2/s
    divisor = 6.0 if is_total else 2.0
    D = (slope * 1e-4) / divisor
    D_err = (std_err * 1e-4) / divisor
    
    return D, D_err, r_val**2, (slope * x + intercept)


def main():
    """Main execution block for MSD and Arrhenius analysis."""
    if len(sys.argv) < 4:
        sys.exit("Usage: python3 script.py <start_fit_ps> <end_fit_ps> <temp1> [<temp2> ...]")
    
    try:
        start_fit = float(sys.argv[1])
        end_fit = float(sys.argv[2])
        temps = [int(t) for t in sys.argv[3:]]
    except ValueError:
        sys.exit("Error: Check numeric arguments (Start/End must be float, Temps must be int).")

    # Load cache if exists
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)

    summary = [] 

    for T in temps:
        T_key = str(T)
        traj_file = f"md_{T}K.traj"
        
        if not os.path.isfile(traj_file):
            print(f"Skipping {T}K: File {traj_file} not found.")
            continue

        print(f"\n--- Processing {T}K ---")
        frames = load_trajectory_safe(traj_file)
        pos = get_unwrapped_positions(frames, TARGET_SPECIES)
        
        # Timing calculations
        dt_ps = (TIMESTEP_FS * LOG_INTERVAL) / 1000.0
        times = np.arange(len(pos)) * dt_ps

        # Compute MSD and Diffusivity
        mx, my, mz, mt = compute_msd_components(pos)
        Dx, Ex, R2x, _ = fit_diffusivity(times, mx, start_fit, end_fit, False)
        Dy, Ey, R2y, _ = fit_diffusivity(times, my, start_fit, end_fit, False)
        Dz, Ez, R2z, _ = fit_diffusivity(times, mz, start_fit, end_fit, False)
        Dt, Et, R2t, _ = fit_diffusivity(times, mt, start_fit, end_fit, True)

        print(f"  Dx: {Dx:.2e} ± {Ex:.2e} | Dy: {Dy:.2e} ± {Ey:.2e} | Dz: {Dz:.2e} ± {Ez:.2e}")
        print(f"  Total D: {Dt:.2e} ± {Et:.2e}")

        # Individual MSD Component Plots
        plt.figure(figsize=(10, 8))
        plt.plot(times, mt, 'k-', label=f'$D_{{tot}} = {Dt:.1e} \\pm {Et:.1e}$')
        plt.plot(times, mx, 'r-', label=f'$D_x = {Dx:.1e} \\pm {Ex:.1e}$', alpha=0.7)
        plt.plot(times, my, 'g-', label=f'$D_y = {Dy:.1e} \\pm {Ey:.1e}$', alpha=0.7)
        plt.plot(times, mz, 'b-', label=f'$D_z = {Dz:.1e} \\pm {Ez:.1e}$', alpha=0.7)
        plt.xlabel("Time (ps)")
        plt.ylabel(r"MSD ($\mathrm{\AA}^{2}$)")
        plt.title(f"Ensemble MSD Components at {T}K")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"msd_components_{T}K.png", dpi=300)
        plt.close()

        # Update cache and summary
        cache[T_key] = {"Dt": Dt, "Et": Et, "Dx": Dx, "Ex": Ex, "Dy": Dy, "Ey": Ey, "Dz": Dz, "Ez": Ez}
        summary.append((T, Dt, Et, Dx, Ex, Dy, Ey, Dz, Ez))

    # Save results to cache
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

    # --- Combined Arrhenius Analysis (Needs at least 3 data points) ---
    if len(summary) >= 3:
        summary.sort()
        summary_arr = np.array(summary)
        T_vals = summary_arr[:, 0]
        inv_T = 1.0 / T_vals
        inv_T_1000 = 1000.0 / T_vals

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Configuration for total and components
        configs = [
            (1, 2, "Total", "k"),
            (3, 4, "$D_x$", "r"),
            (5, 6, "$D_y$", "g"),
            (7, 8, "$D_z$", "b")
        ]

        print("\n--- Final Activation Energies ---")
        for d_idx, err_idx, label, color in configs:
            D = summary_arr[:, d_idx]
            E = summary_arr[:, err_idx]
            ln_D = np.log(D)
            ln_D_err = E / D  # Error propagation for log
            
            slope, intercept, r_val, _, std_err = linregress(inv_T, ln_D)
            ea = (-slope * k_B) / e_charge
            ea_err = (std_err * k_B) / e_charge
            
            ax.errorbar(inv_T_1000, ln_D, yerr=ln_D_err, fmt=color+'o', capsize=5, markersize=8)
            ax.plot(inv_T_1000, intercept + slope * inv_T, color + '-', 
                    label=f'{label}: $E_a = {ea:.3f} \\pm {ea_err:.3f}$ eV')
            
            print(f"  Ea ({label}): {ea:.4f} ± {ea_err:.4f} eV (R2: {r_val**2:.4f})")

        ax.set_xlabel("1000 / T (K$^{-1}$)")
        ax.set_ylabel("ln(D)")
        ax.set_title("Combined Arrhenius Plot (Total & Components)")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("arrhenius_combined.png", dpi=300)
        print("\nSuccess: Saved combined Arrhenius plot to arrhenius_combined.png")


if __name__ == "__main__":
    main()
