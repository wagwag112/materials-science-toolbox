#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Computational Materials Science Toolkit: Diffusion Analysis
---------------------------------------------------------
Author: Huy Hoang
Description: 
    Calculates Diffusivity (D) and Activation Energy (Ea) from MD trajectories.
    
    Features:
    - MSD Analysis: Time-Averaged (Smooth) & Ensemble t=0 (Raw).
    - Robust Fitting: Auto-detection of diffusive regime.
    - Checkpointing: Uses 'analysis_cache.json' to resume interrupted runs.
    - Visualization: Detailed plots with Ballistic/Fitting/Poor-Statistics regions.
    - Error Analysis: Calculates uncertainties for D and Ea (Activation Energy).

Usage:
    python3 analysis_msd.py <temp1> [<temp2> ...]
"""

import sys
import os
import json  # Added for database management
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for HPC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import linregress
from scipy.signal import savgol_filter
from ase.io import read

# --- Configuration Section ---
CACHE_FILE = "analysis_cache.json"  # Single database file
TIMESTEP_FS = 2.0       # MD timestep in femtoseconds
LOG_INTERVAL = 50       # Number of steps per saved frame
DIM = 3                 # System dimensionality (3 for bulk)
TARGET_SPECIES = "Li"   # Element symbol to analyze

# Analysis Parameters
NUM_POINTS_ROBUST = 500 
FIXED_BALLISTIC_PS = 5.0    # Exclude early ballistic regime (ps)
SLOPE_TOLERANCE = 0.25      # Tolerance for detecting linear regime
MIN_WINDOW_PS = 10.0        # Minimum duration for fitting window

# --- Constants ---
k_B = 1.380649e-23      # Boltzmann constant (J/K)
e_charge = 1.602176634e-19  # Elementary charge (C)

# --- Plot Style ---
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "font.family": "sans-serif",
    "lines.linewidth": 2
})

def load_trajectory_safe(filename):
    """Loads trajectory file safely using ASE."""
    try:
        with open(filename, 'rb') as f:
            return read(f, index=":", format='traj')
    except Exception as e:
        sys.exit(f"Fatal Error: Could not read '{filename}'. Reason: {e}")

def unwrap_trajectory(frames):
    """Corrects positions for Periodic Boundary Conditions (PBC)."""
    print("  -> Unwrapping trajectory (PBC correction)...")
    n_frames = len(frames)
    n_atoms = len(frames[0])
    cell = frames[0].get_cell()
    inv_cell = np.linalg.inv(cell)
    
    positions = np.zeros((n_frames, n_atoms, 3))
    positions[0] = frames[0].get_positions()
    cumulative_shift = np.zeros((n_atoms, 3))
    
    for i in range(1, n_frames):
        pos_current = frames[i].get_positions()
        pos_prev = frames[i-1].get_positions()
        diff = pos_current - pos_prev
        
        diff_frac = np.dot(diff, inv_cell)
        shift_frac = -np.round(diff_frac)
        shift_cart = np.dot(shift_frac, cell)
        
        cumulative_shift += shift_cart
        positions[i] = pos_current + cumulative_shift
        
    return positions

def compute_msd_ensemble_t0(positions):
    """Calculates Ensemble MSD relative to t=0 (Raw MSD)."""
    print("  -> Computing Ensemble MSD (from t=0)...")
    r0 = positions[0]
    displacements = positions - r0
    sq_disp = np.sum(displacements**2, axis=2)
    msd = np.mean(sq_disp, axis=1)
    return msd

def compute_msd_time_averaged(positions, num_points=500):
    """Calculates Time-Averaged MSD (Robust Standard)."""
    print(f"  -> Computing Time-Averaged MSD (Smoothed, {num_points} points)...")
    n_frames = len(positions)
    
    taus = np.unique(np.linspace(1, n_frames-1, num_points).astype(int))
    taus = taus[taus > 0]
    
    msd_list = []
    tau_list = []

    for tau in taus:
        diff = positions[tau:] - positions[:-tau]
        sq_disp = np.sum(diff**2, axis=2)
        mean_sq_disp = np.mean(sq_disp) 
        
        msd_list.append(mean_sq_disp)
        tau_list.append(tau)
        
    return np.array(msd_list), np.array(tau_list)

def auto_find_diffusive_regime(times, msd):
    """Automatically detects the linear regime in log-log MSD plot."""
    total_duration = times[-1]
    is_short_run = total_duration < 150.0
    
    valid_idx = np.where(times > 0.1)[0]
    t_valid = times[valid_idx]
    msd_valid = msd[valid_idx]
    
    t_log = np.log(t_valid)
    msd_log = np.log(msd_valid)
    
    beta = np.gradient(msd_log, t_log)
    
    window_length = min(15, len(beta) // 2 * 2 + 1)
    if window_length > 3:
        beta_smooth = savgol_filter(beta, window_length, 3)
    else:
        beta_smooth = beta
        
    mask = (beta_smooth >= (1.0 - SLOPE_TOLERANCE)) & (beta_smooth <= (1.0 + SLOPE_TOLERANCE))
    
    full_mask = np.zeros(len(times), dtype=bool)
    full_mask[valid_idx] = mask
    
    skip_indices = np.where(times < FIXED_BALLISTIC_PS)[0]
    if len(skip_indices) > 0:
        full_mask[skip_indices] = False

    padded_mask = np.concatenate(([False], full_mask, [False]))
    diff = np.diff(padded_mask.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    best_len = 0
    best_start_idx = 0
    best_end_idx = 0
    found = False
    
    for s, e in zip(starts, ends):
        t_start = times[s]
        t_end = times[e-1]
        duration = t_end - t_start
        min_win = 5.0 if is_short_run else MIN_WINDOW_PS
        
        if duration > min_win and duration > best_len:
            best_len = duration
            best_start_idx = s
            best_end_idx = e-1
            found = True
            
    if not found:
        start_percent = 0.75 if is_short_run else 0.70
        best_start_idx = int(len(times) * start_percent)
        best_end_idx = len(times) - 1
            
    return best_start_idx, best_end_idx

def robust_fit_diffusivity(times, msd, start_idx, end_idx):
    """Performs linear regression to calculate D."""
    fit_time = times[start_idx:end_idx]
    fit_msd = msd[start_idx:end_idx]
    
    if len(fit_time) < 2:
        return 0.0, 0.0, 0.0, fit_time, fit_msd

    slope, intercept, r_value, _, _ = linregress(fit_time, fit_msd)
    D_overall = (slope * 1e-4) / (2 * DIM) 
    fit_line = slope * fit_time + intercept
    r2_overall = r_value**2
    
    n_blocks = 4
    block_size = len(fit_time) // n_blocks
    if block_size > 5:
        D_values = []
        for i in range(n_blocks):
            s = i * block_size
            e = (i + 1) * block_size
            slope_i, _, _, _, _ = linregress(fit_time[s:e], fit_msd[s:e])
            D_i = (slope_i * 1e-4) / (2 * DIM)
            D_values.append(D_i)
        D_std = np.std(D_values) 
    else:
        D_std = 0.0
    
    return D_overall, D_std, r2_overall, fit_line, fit_time

def main():
    if len(sys.argv) < 2:
        script_name = sys.argv[0]
        sys.exit(f"Usage: python3 {script_name} <temp1> [<temp2> ...]")
    
    try:
        temps_to_process = [int(t) for t in sys.argv[1:]]
    except ValueError:
        sys.exit("Error: Temperatures must be integers.")

    # --- Load Cache from JSON ---
    cache_data = {}
    if os.path.isfile(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            print(f"--> Loaded analysis cache from {CACHE_FILE}")
        except Exception as e:
            print(f"--> Warning: Could not read cache file ({e}). Starting fresh.")

    summary = []

    for T in temps_to_process:
        T_str = str(T) # JSON keys are always strings
        traj_file = f"md_{T}K.traj"
        
        # Check Cache
        if T_str in cache_data:
            entry = cache_data[T_str]
            print(f"--- Found cached result for {T} K ---")
            print(f"  -> Loaded: D = ({entry['D']:.2e} +/- {entry['D_err']:.1e}) cm^2/s")
            summary.append((T, entry['D'], entry['D_err'], entry['R2'], entry['start_ps'], entry['end_ps']))
            continue # Skip to next temperature

        if not os.path.isfile(traj_file):
            print(f"Warning: File '{traj_file}' not found, skipping.")
            continue

        print(f"Processing {traj_file} at {T} K...")
        frames = load_trajectory_safe(traj_file)
        
        # 1. Filter Atoms
        positions_all = unwrap_trajectory(frames)
        target_indices = [atom.index for atom in frames[0] if atom.symbol == TARGET_SPECIES]
        
        if len(target_indices) == 0:
            print(f"  [Error] No atoms found with symbol '{TARGET_SPECIES}'! Skipping...")
            continue
            
        print(f"  -> Filtering {len(target_indices)} atoms of type '{TARGET_SPECIES}'...")
        positions_target = positions_all[:, target_indices, :]

        # 2. Compute MSD (Two Methods)
        msd_t0 = compute_msd_ensemble_t0(positions_target)
        real_time_per_frame_ps = (TIMESTEP_FS * LOG_INTERVAL) / 1000.0
        times_ps_t0 = np.arange(len(msd_t0)) * real_time_per_frame_ps
        
        msd_robust, tau_robust = compute_msd_time_averaged(positions_target, num_points=NUM_POINTS_ROBUST)
        times_ps_robust = tau_robust * real_time_per_frame_ps
        
        # 3. Fit Diffusivity
        start_idx, end_idx = auto_find_diffusive_regime(times_ps_robust, msd_robust)
        start_ps = times_ps_robust[start_idx]
        end_ps = times_ps_robust[end_idx]
        
        D, D_err, r2, fit_line, fit_time_axis = robust_fit_diffusivity(
            times_ps_robust, msd_robust, start_idx, end_idx
        )
        
        print(f"  Fit Range: {start_ps:.1f} - {end_ps:.1f} ps")
        print(f"  Result: D = ({D:.2e} +/- {D_err:.1e}) cm^2/s")
        
        # --- PLOTTING ---
        plt.figure(figsize=(9, 7))
        
        # A. Regions
        if start_ps > 0:
            plt.axvspan(0, start_ps, color='gray', alpha=0.15, label="Ballistic/Equil")
        plt.axvspan(start_ps, end_ps, color='yellow', alpha=0.15, label="Fitting Region")
        total_time = times_ps_robust[-1]
        if end_ps < total_time * 0.95: 
            plt.axvspan(end_ps, total_time, color='red', alpha=0.05, label="Poor Statistics")

        # B. Raw MSD
        plt.plot(times_ps_t0, msd_t0, color='gray', alpha=0.4, linewidth=1.0, 
                 label=f"Ensemble MSD (from $t_0$)")
        
        # C. Smooth MSD
        plt.plot(times_ps_robust, msd_robust, color='black', linewidth=2.5, 
                 label=f"Time-Averaged MSD")
        
        # D. Fit Line
        plt.plot(fit_time_axis, fit_line, "--", color="red", linewidth=2.0, 
                 label=f"Linear Fit")
        
        mantissa, exponent = f"{D:.2e}".split('e')
        exponent_val = int(exponent)
        error_val = D_err / float(f"1e{exponent_val}")
        
        plot_title = f"$T$={T}K, $D$=({mantissa}$\\pm${error_val:.2f})$\\times$10$^{{{exponent_val}}}$ cm$^{{2}}$/s"
        plt.title(plot_title)
        plt.xlabel(r"Time (ps)")
        plt.ylabel(r"MSD ($\mathrm{\AA}^{2}$)")
        
        ax = plt.gca()
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
        
        plt.legend(loc="upper left", frameon=True, fontsize=12) 
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        png_filename = traj_file.replace(".traj", ".png")
        plt.savefig(png_filename, dpi=300)
        plt.close()
        
        summary.append((T, D, D_err, r2, start_ps, end_ps)) 
        print(f"  Saved plot to {png_filename}\n")
        
        # --- Update Cache (JSON) ---
        cache_data[T_str] = {
            "D": D,
            "D_err": D_err,
            "R2": r2,
            "start_ps": start_ps,
            "end_ps": end_ps
        }
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=4)
        except Exception as e:
            print(f"  [Warning] Failed to update JSON cache: {e}")

    # --- ARRHENIUS PLOT ---
    if summary:
        summary.sort(key=lambda x: x[0])
    
    valid_summary = [item for item in summary if item[1] > 0]
    if len(valid_summary) < 3:
        print("\n[SKIP] Arrhenius plot skipped: Not enough valid data points.")
    else:
        try:
            temps_arr = np.array([item[0] for item in valid_summary], float)
            Ds_arr = np.array([item[1] for item in valid_summary], float)
            D_errs_arr = np.array([item[2] for item in valid_summary], float)
            
            invT = 1.0 / temps_arr
            lnD = np.log(Ds_arr)
            lnD_err = np.abs(D_errs_arr / Ds_arr)
            
            # linregress returns: slope, intercept, r_value, p_value, std_err
            slope_A, intercept_A, r_A, _, std_err_A = linregress(invT, lnD)
            
            R2_A = r_A**2
            Ea_J = -slope_A * k_B
            Ea_eV = Ea_J / e_charge
            D0 = np.exp(intercept_A)
            
            # Calculate Error for Ea
            # Ea = -slope * kB -> Delta_Ea = Delta_slope * kB
            Ea_err_J = std_err_A * k_B
            Ea_err_eV = Ea_err_J / e_charge
            
            print("\n--- ARRHENIUS FIT RESULTS ---")
            print(f"  Ea = {Ea_eV:.4f} +/- {Ea_err_eV:.4f} eV")
            print(f"  D0 = {D0:.4e} cm^2/s")
            print(f"  R2 = {R2_A:.4f}")
            
            plt.figure(figsize=(8, 6))
            
            # Plot with Error Bars
            plt.errorbar(1000.0/temps_arr, lnD, yerr=lnD_err, fmt='o', 
                         color='black', ecolor='gray', capsize=5, markersize=8, 
                         label="Sim Data")
            
            x_fit_T = np.linspace(min(temps_arr), max(temps_arr), 100)
            y_fit_lnD = intercept_A + slope_A * (1.0 / x_fit_T)
            plt.plot(1000.0/x_fit_T, y_fit_lnD, "--", color="red", label=f"Arrhenius Fit")
            
            plt.xlabel(r"1000 / $T$ (K$^{-1}$)")
            plt.ylabel(r"ln($D$)  ($D$ in cm$^{2}$/s)")
            
            # Update Title with Error
            plt.title(rf"Arrhenius Plot ($E_a$ = {Ea_eV:.3f} $\pm$ {Ea_err_eV:.3f} eV)")
            
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig("arrhenius_fit.png", dpi=300)
            print("  Saved Arrhenius plot to arrhenius_fit.png")

        except Exception as e:
            print(f"\n[ERROR] Arrhenius plot failed: {e}")

if __name__ == "__main__":
    main()
