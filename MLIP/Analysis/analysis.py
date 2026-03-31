#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import linregress
from ase.io import read
import math
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# --- Font setup: Helvetica ---
import matplotlib as mpl
from matplotlib import font_manager as fm

def init_helvetica_font(font_dir):
    try:
        if os.path.isdir(font_dir):
            for fname in os.listdir(font_dir):
                if fname.lower().endswith(".ttf"):
                    fm.fontManager.addfont(os.path.join(font_dir, fname))

        font_path = os.path.join(font_dir, "Helvetica.ttf")
        if os.path.isfile(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()

            mpl.rcParams['font.family'] = font_name
            mpl.rcParams['font.serif'] = [font_name]
            mpl.rcParams['font.sans-serif'] = [font_name]
            mpl.rcParams['mathtext.fontset'] = 'custom'
            mpl.rcParams['mathtext.rm'] = font_name

            print(f"--> Using custom font: {font_name}")
            return font_prop
        else:
            print("--> Helvetica.ttf not found, using default font.")
            return None
    except Exception as e:
        print(f"--> Font setup failed: {e}")
        return None

# Custom font directory on HPC cluster
font_dir = r"/home/tsai0000/helvetica/"
font_prop = init_helvetica_font(font_dir)

# --- Configuration Section ---
TIMESTEP_FS = 2.0
LOG_INTERVAL = 50
DIM = 3
MOBILE_ION_SYMBOL = "Li"
N_BLOCKS = 5  

# --- Constants ---
k_B = 1.380649e-23      # J/K
e_charge = 1.602176634e-19  # C

# --- Global plot style settings ---
plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "mathtext.default": "regular"
})

def load_trajectory(filename):
    try:
        traj = read(filename, index=":")
        if not traj:
            sys.exit(f"Fatal Error: No frames found in {filename}")
        return traj
    except Exception as e:
        sys.exit(f"Fatal Error: Could not read {filename}: {e}")

def unwrap_all_atoms_vectorized(frames):
    """Vectorized PBC unwrapping to ensure continuous trajectories."""
    pos_all = np.array([f.get_positions() for f in frames])
    cell = frames[0].get_cell()
    inv_cell = np.linalg.inv(cell)
    
    diffs = np.diff(pos_all, axis=0)
    diffs_frac = np.einsum('ijk,kl->ijl', diffs, inv_cell)
    shifts_frac = -np.round(diffs_frac)
    shifts_cart = np.einsum('ijk,kl->ijl', shifts_frac, cell)
    
    unwrapped = np.zeros_like(pos_all)
    unwrapped[0] = pos_all[0]
    # Fixed logic to avoid accumulation drift
    unwrapped[1:] = pos_all[1:] + np.cumsum(shifts_cart, axis=0)
    
    return unwrapped

def subtract_com_drift(unwrapped_pos_all, framework_indices, target_indices):
    """Subtracts framework Center of Mass drift from target species."""
    framework_pos = unwrapped_pos_all[:, framework_indices, :]
    com_framework = np.mean(framework_pos, axis=1) 
    drift = com_framework - com_framework[0]
    
    target_pos = unwrapped_pos_all[:, target_indices, :]
    corrected_pos = target_pos - drift[:, np.newaxis, :]
    return corrected_pos

def compute_msd(corrected_pos):
    """Computes MSD for a segment using its local starting frame."""
    r0 = corrected_pos[0]
    dr = corrected_pos - r0
    sq_dist = np.sum(dr**2, axis=2) 
    msd = np.mean(sq_dist, axis=1) 
    return msd

def fit_diffusivity(msd, start_time_ps, end_time_ps, timestep_fs, log_interval):
    n_frames = len(msd)
    time_per_frame_ps = (timestep_fs * log_interval) / 1000.0
    times_ps = np.arange(n_frames) * time_per_frame_ps

    start_idx = np.searchsorted(times_ps, start_time_ps, side='left')
    end_idx = np.searchsorted(times_ps, end_time_ps, side='right')

    if end_idx - start_idx < 2:
        return 0.0, 0.0, np.array([]), np.array([])

    x_fit = times_ps[start_idx:end_idx]
    y_fit = msd[start_idx:end_idx]
    
    slope, intercept, r_val, _, _ = linregress(x_fit, y_fit)
    D_cm2_s = (slope * 1e-4) / (2 * DIM)
    fit_line = intercept + slope * x_fit
    
    return D_cm2_s, r_val**2, x_fit, fit_line

def calculate_conductivity(D_cm2_s, D_std, temp_k, volume_ang3, n_ions):
    vol_cm3 = volume_ang3 * 1e-24
    n_density = n_ions / vol_cm3
    q = e_charge
    sigma = (n_density * (q**2) * D_cm2_s) / (k_B * temp_k)
    sigma_std = sigma * (D_std / D_cm2_s) if D_cm2_s > 0 else 0
    return sigma, sigma_std

def plot_msd(msd, fit_t, fit_line, title, filename, fit_range_str, end_time_ps):
    time_per_frame_ps = (TIMESTEP_FS * LOG_INTERVAL) / 1000.0
    times_ps = np.arange(len(msd)) * time_per_frame_ps
    idx_limit = np.searchsorted(times_ps, end_time_ps, side='right')
    
    plt.figure(figsize=(8, 6))
    plt.plot(times_ps[:idx_limit], msd[:idx_limit], color="black", label="MSD Data")
    if len(fit_t) > 0:
        plt.plot(fit_t, fit_line, "--", color="red", label=f"Fit ({fit_range_str})")
    
    plt.xlabel(r"Time (ps)")
    plt.ylabel(r"MSD ($\mathrm{\AA}^{2}$)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    # Robust Argument Parsing with Explicit Flags
    if len(sys.argv) < 2:
        sys.exit("Usage:\n  --full <ps_start> <ps_end> <t1> <t2> ...\n  --slab <z_min> <z_max> <ps_start> <ps_end> <t1> <t2> ...")

    if sys.argv[1] == "--full":
        mode = "FULL"
        start_fit_ps = float(sys.argv[2])
        end_fit_ps   = float(sys.argv[3])
        temps = [int(t) for t in sys.argv[4:]]
        z_min, z_max = None, None
        suffix = "_full"
        print("\n--> Mode: FULL CELL analysis.")
    elif sys.argv[1] == "--slab":
        mode = "SLAB"
        z_min        = float(sys.argv[2])
        z_max        = float(sys.argv[3])
        start_fit_ps = float(sys.argv[4])
        end_fit_ps   = float(sys.argv[5])
        temps = [int(t) for t in sys.argv[6:]]
        suffix = f"_z{int(z_min)}-{int(z_max)}"
        print(f"\n--> Mode: SLAB analysis in Z-range [{z_min:.1f}, {z_max:.1f}].")
    else:
        sys.exit("Error: Invalid flag. Use --full or --slab.")

    results = {}

    for T in temps:
        traj_file = f"md_{T}K.traj"
        if not os.path.isfile(traj_file): continue

        print(f"Processing {T}K...")
        frames = load_trajectory(traj_file)
        
        # Area and Slab Volume Correction
        cell = frames[0].get_cell()
        xy_area = np.linalg.norm(np.cross(cell[0], cell[1]))
        pos_t0 = frames[0].get_positions()
        symbols = np.array(frames[0].get_chemical_symbols())
        
        mobile_set = {atom.index for atom in frames[0] if atom.symbol == MOBILE_ION_SYMBOL}
        framework_indices = np.array([idx for idx in np.arange(len(frames[0])) if idx not in mobile_set])
        
        if mode == "SLAB":
            z_mask = (pos_t0[:, 2] >= z_min) & (pos_t0[:, 2] <= z_max)
            # Added sorted() for index reproducibility
            target_indices = np.array(sorted([idx for idx in mobile_set if z_mask[idx]]))
            vol_eff = xy_area * (z_max - z_min)
        else:
            target_indices = np.array(sorted(list(mobile_set)))
            vol_eff = frames[0].get_volume()
            
        n_target = len(target_indices)
        if n_target == 0:
            print(f"  Warning: No mobile ions found in target region. Skipping {T}K.")
            continue

        # Analysis Pipeline: Unwrap -> COM Correction -> MSD
        unwrapped_all = unwrap_all_atoms_vectorized(frames)
        corrected_target_all = subtract_com_drift(unwrapped_all, framework_indices, target_indices)

        time_per_frame_ps = (TIMESTEP_FS * LOG_INTERVAL) / 1000.0
        start_idx = int(start_fit_ps / time_per_frame_ps)
        end_idx   = int(end_fit_ps   / time_per_frame_ps)
        
        fit_pos = corrected_target_all[start_idx:end_idx]
        block_size = len(fit_pos) // N_BLOCKS
        block_duration = block_size * time_per_frame_ps
        
        # Block duration warning
        if block_duration < 5.0:
            print(f"  Warning [{T}K]: Block duration ({block_duration:.1f} ps) is quite short.")

        block_diffusivities = []
        for b in range(N_BLOCKS):
            block_seg = fit_pos[b*block_size : (b+1)*block_size]
            if len(block_seg) < 2: continue
            b_msd = compute_msd(block_seg)
            D, _, _, _ = fit_diffusivity(b_msd, 0.0, block_duration * 0.9, TIMESTEP_FS, LOG_INTERVAL)
            if D > 0: block_diffusivities.append(D)

        if not block_diffusivities: continue

        D_avg = np.mean(block_diffusivities)
        D_std = np.std(block_diffusivities) / np.sqrt(len(block_diffusivities)) 
        
        full_msd = compute_msd(corrected_target_all)
        D_plot, r2, fit_t, fit_line = fit_diffusivity(full_msd, start_fit_ps, end_fit_ps, TIMESTEP_FS, LOG_INTERVAL)
        
        sigma, sigma_std = calculate_conductivity(D_avg, D_std, T, vol_eff, n_target)
        results[T] = {'D': D_avg, 'D_std': D_std, 'sigma': sigma, 'sigma_std': sigma_std, 'r2': r2}

        # Plot formatting with scientific notation
        exponent = math.floor(math.log10(abs(D_avg))) if D_avg != 0 else 0
        factor = 10**exponent
        png_name = f"msd_{T}K{suffix}.png"
        title_str = fr"$T={T}\,\mathrm{{K}}, D=({D_avg/factor:.2f} \pm {D_std/factor:.2f}) \times 10^{{{exponent}}}\,\mathrm{{cm}}^2/\mathrm{{s}}$"
        plot_msd(full_msd, fit_t, fit_line, title_str, png_name, f"{start_fit_ps}-{end_fit_ps} ps", end_fit_ps)

    if len(results) >= 3:
        sorted_temps = sorted(results.keys())
        inv_T = 1000.0 / np.array(sorted_temps)
        ln_D = np.log([results[t]['D'] for t in sorted_temps])
        ln_D_err = np.array([results[t]['D_std']/results[t]['D'] for t in sorted_temps])
        
        # Arrhenius Fit for Activation Energy
        slope, intercept, r_val, _, std_err = linregress(1.0/np.array(sorted_temps), ln_D)
        ea_ev = (-slope * k_B) / e_charge
        ea_err_ev = (std_err * k_B) / e_charge
        
        plt.figure(figsize=(8, 6))
        plt.errorbar(inv_T, ln_D, yerr=ln_D_err, fmt='ko', capsize=5, label="Data")
        plt.plot(inv_T, intercept + slope * (1.0/np.array(sorted_temps)), 'r--', label="Arrhenius Fit")
        plt.xlabel("1000 / T (K$^{-1}$)")
        plt.ylabel("ln(D)")
        # Fixed LaTeX title formatting
        plt.title(rf"Arrhenius: $E_a = {ea_ev:.3f} \pm {ea_err_ev:.3f}$ eV{suffix.replace('_', ' ')}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"arrhenius{suffix}.png", dpi=300)
        
        print(f"\n--- Final Summary ({mode} Mode) ---")
        print(f"Calculated Ea: {ea_ev:.4f} +/- {ea_err_ev:.4f} eV")
        print(f"R-squared: {r_val**2:.4f}\n")
        for T in sorted_temps:
            r = results[T]
            print(f"T={T}K: D={r['D']:.2e} +/- {r['D_std']:.2e} cm2/s, Sigma={r['sigma']:.4e} +/- {r['sigma_std']:.4e} S/cm")

if __name__ == "__main__":
    main()
