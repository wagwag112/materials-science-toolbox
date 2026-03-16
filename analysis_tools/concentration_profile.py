"""
Author: Huy Hoang
Calculate smoothed atomic density and concentration profiles from an ASE trajectory using Gaussian Smearing. 
Generates visualization for global chemical composition and individual species distribution along 
the chosen axis (X, Y, or Z).

Usage:
    # Analyze density and concentration for Li, S, and Cl from frame 0 to 1000
    python concentration_profile.py interface.traj 0 1000 Li S Cl
"""

import sys
import os
import matplotlib
matplotlib.use('Agg') # Necessary for HPC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # Added for tick control
from ase.io import read
from scipy.ndimage import gaussian_filter1d

def compute_smoothed_profile(atoms_list, symbols, axis=0, sigma=0.1, res=0.1):
    """
    Calculate smoothed density using Gaussian Smearing.
    """
    first_atoms = atoms_list[0]
    lx = np.linalg.norm(first_atoms.get_cell()[axis])
    vol = first_atoms.get_volume()
    area = vol / lx
    
    grid = np.arange(0, lx, res)
    densities = {s: np.zeros_like(grid) for s in symbols}
    
    print(f"  [LOG] Processing {len(atoms_list)} frames along axis {axis}...")

    for atoms in atoms_list:
        syms = np.array(atoms.get_chemical_symbols())
        positions = atoms.get_positions()
        
        for s in symbols:
            coords = positions[syms == s][:, axis]
            if len(coords) == 0: continue
            
            counts, _ = np.histogram(coords, bins=np.append(grid, lx))
            sigma_px = sigma / res
            smoothed = gaussian_filter1d(counts.astype(float), sigma_px, mode='wrap')
            densities[s] += (smoothed / (res * area))
            
    for s in symbols:
        densities[s] /= len(atoms_list)
        
    return grid, densities

def main():
    # ==============================================================
    # --- CONFIGURATION SECTION ---
    PROJECTION_AXIS = 2      # 0 for X, 1 for Y, 2 for Z
    X_TICK_STEP = 5.0 
    MINOR_TICKS_PER_MAJOR = 5
    # ==============================================================
    
    axis_name = {0: 'X', 1: 'Y', 2: 'Z'}[PROJECTION_AXIS]

    if len(sys.argv) < 5:
        print("Usage: python3 script.py <traj_file> <start> <end> <elem1> <elem2> ...")
        sys.exit(1)

    t_file = sys.argv[1]
    s_idx = int(sys.argv[2])
    e_idx = int(sys.argv[3])
    target_elements = sys.argv[4:]

    if not os.path.exists(t_file):
        print(f"Error: {t_file} not found.")
        sys.exit(1)

    traj = read(t_file, index=":")
    all_system_elements = sorted(list(set(traj[0].get_chemical_symbols())))

    # 1. Compute profiles using the configured PROJECTION_AXIS
    grid, ref_d = compute_smoothed_profile([traj[0]], all_system_elements, axis=PROJECTION_AXIS, sigma=0.4)
    avg_frames = traj[s_idx:e_idx+1]
    _, avg_d = compute_smoothed_profile(avg_frames, all_system_elements, axis=PROJECTION_AXIS, sigma=0.4)

    # 2. GLOBAL PERCENTAGE PLOT
    total_avg_dens = np.sum([avg_d[s] for s in all_system_elements], axis=0)
    total_avg_dens = np.where(total_avg_dens == 0, 1e-10, total_avg_dens)
    
    plt.figure(figsize=(12, 6))
    for s in all_system_elements:
        percentage = (avg_d[s] / total_avg_dens) * 100
        plt.plot(grid, percentage, label=s, lw=2.5)

    # Set tick frequency for X-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(X_TICK_STEP))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(MINOR_TICKS_PER_MAJOR))
    ax.tick_params(which='major', length=7, width=1.5) 
    ax.tick_params(which='minor', length=4, width=1) 

    plt.ylabel("Atomic Concentration (%)")
    plt.xlabel(f"Distance along {axis_name}-axis ($\mathrm{{\AA}}$)")
    plt.title(f"Global Concentration Profile ({s_idx}-{e_idx})")
    plt.ylim(-5, 105)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"concentration_percentage_all_{s_idx}_{e_idx}.png", dpi=300)
    plt.close()

    # 3. INDIVIDUAL DENSITY PLOTS
    for sym in target_elements:
        if sym not in avg_d: continue
            
        plt.figure(figsize=(10, 5))
        plt.plot(grid, avg_d[sym], label=f"{sym} (Avg)", color='blue', lw=2.0)
        plt.plot(grid, ref_d[sym], '--', label=f"{sym} (t=0)", color='red', alpha=0.4, lw=1.2)
        
        # Set tick frequency for X-axis
        ax_ind = plt.gca()
        ax_ind.xaxis.set_major_locator(ticker.MultipleLocator(X_TICK_STEP))
        ax_ind.xaxis.set_minor_locator(ticker.AutoMinorLocator(MINOR_TICKS_PER_MAJOR))
        ax_ind.tick_params(which='major', length=7, width=1.5)
        ax_ind.tick_params(which='minor', length=4, width=1)
        
        plt.ylabel(r"Number Density ($\mathrm{atoms}/\mathrm{\AA}^{3}$)")
        plt.xlabel(f"Distance along {axis_name}-axis ($\mathrm{{\AA}}$)")
        plt.title(f"Density Profile: {sym}")
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"density_{sym}_{s_idx}_{e_idx}.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
