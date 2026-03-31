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

# Update this path to your actual font directory
font_dir = r"/home/tsai0000/helvetica/"
font_prop = init_helvetica_font(font_dir)

# --- Configuration Section ---
TIMESTEP_FS = 2.0
LOG_INTERVAL = 50
DIM = 3
MOBILE_ION_SYMBOL = "Li"
N_BLOCKS = 5  # Strategy: Block Averaging

# --- Constants ---
k_B = 1.380649e-23
e_charge = 1.602176634e-19

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

def compute_msd(frames, indices):
    """
    Computes MSD specifically for indices of mobile ions.
    """
    n_frames = len(frames)
    # Using wrap=False to handle Periodic Boundary Conditions correctly
    r0 = frames[0].get_positions(wrap=False)[indices]
    msd = np.empty(n_frames, float)
    for i, frame in enumerate(frames):
        pos = frame.get_positions(wrap=False)[indices]
        dr = pos - r0
        msd[i] = np.mean(np.sum(dr**2, axis=1))
    return msd

def fit_diffusivity(msd, start_time_ps, end_time_ps, timestep_fs, log_interval):
    n_frames = len(msd)
    time_per_frame_ps = (timestep_fs * log_interval) / 1000.0
    times_ps = np.arange(n_frames) * time_per_frame_ps

    start_idx = np.searchsorted(times_ps, start_time_ps, side='left')
    end_idx = np.searchsorted(times_ps, end_time_ps, side='right')

    if end_idx - start_idx < 2:
        raise ValueError(f"Fit range too small.")

    x_fit = times_ps[start_idx:end_idx]
    y_fit = msd[start_idx:end_idx]
    
    slope, intercept, r_val, _, _ = linregress(x_fit, y_fit)
    # Conversion from A^2/ps to cm^2/s is 1e-4
    D_cm2_s = (slope * 1e-4) / (2 * DIM)
    fit_line = intercept + slope * x_fit
    
    return D_cm2_s, r_val**2, x_fit, fit_line

def calculate_conductivity(D_cm2_s, temp_k, volume_ang3, n_ions):
    """
    Calculates Ionic Conductivity (S/cm) using Nernst-Einstein equation.
    """
    # Convert volume A^3 to cm^3
    vol_cm3 = volume_ang3 * 1e-24
    # Number density n (ions/cm^3)
    n_density = n_ions / vol_cm3
    # Charge q (assumed Li+)
    q = e_charge
    # Sigma = (n * q^2 * D) / (k_B * T)
    sigma = (n_density * (q**2) * D_cm2_s) / (k_B * temp_k)
    return sigma

def plot_msd(msd, fit_t, fit_line, title, filename, fit_range_str, end_time_ps):
    time_per_frame_ps = (TIMESTEP_FS * LOG_INTERVAL) / 1000.0
    times_ps = np.arange(len(msd)) * time_per_frame_ps
    
    idx_limit = np.searchsorted(times_ps, end_time_ps, side='right')
    t_plot = times_ps[:idx_limit]
    msd_plot = msd[:idx_limit]

    plt.figure(figsize=(8, 6))
    plt.plot(t_plot, msd_plot, color="black", label="MSD Data")
    plt.plot(fit_t, fit_line, "--", color="red", label=f"Fit ({fit_range_str})")
    
    plt.xlabel(r"Time (ps)")
    plt.ylabel(r"MSD ($\mathrm{\AA}^{2}$)")
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    if len(sys.argv) < 4:
        sys.exit("Usage: python3 script.py <start_ps> <end_ps> <temp1> [<temp2> ...]")
    
    try:
        start_fit_ps = float(sys.argv[1])
        end_fit_ps = float(sys.argv[2])
        temps = [int(t) for t in sys.argv[3:]]
    except ValueError:
        sys.exit("Error: Arguments must be numeric.")

    results = {}

    for T in temps:
        traj_file = f"md_{T}K.traj"
        if not os.path.isfile(traj_file):
            print(f"Skipping {traj_file}: File not found.")
            continue

        print(f"Processing {T}K...")
        frames = load_trajectory(traj_file)
        
        # Identify Mobile Ions
        mobile_indices = [atom.index for atom in frames[0] if atom.symbol == MOBILE_ION_SYMBOL]
        n_mobile = len(mobile_indices)
        if n_mobile == 0:
            print(f"  Error: No {MOBILE_ION_SYMBOL} atoms found in {traj_file}")
            continue

        # Block Averaging Strategy (Approach B: split only the fit window)
        time_per_frame_ps = (TIMESTEP_FS * LOG_INTERVAL) / 1000.0
        start_idx = int(start_fit_ps / time_per_frame_ps)
        end_idx   = int(end_fit_ps   / time_per_frame_ps)
        fit_frames = frames[start_idx:end_idx]
        block_size = len(fit_frames) // N_BLOCKS
        block_diffusivities = []

        for b in range(N_BLOCKS):
            block_frames = fit_frames[b*block_size : (b+1)*block_size]
            b_msd = compute_msd(block_frames, mobile_indices)
            try:
                block_duration = len(block_frames) * time_per_frame_ps
                D, _, _, _ = fit_diffusivity(b_msd, 0.0, block_duration * 0.9,
                                             TIMESTEP_FS, LOG_INTERVAL)
                block_diffusivities.append(D)
            except Exception:
                continue

        if not block_diffusivities:
            print(f"  Error: Could not calculate D for {T}K blocks.")
            continue

        D_avg = np.mean(block_diffusivities)
        D_std = np.std(block_diffusivities) / np.sqrt(N_BLOCKS) # Standard Error
        
        # Calculate full trajectory MSD for plotting
        full_msd = compute_msd(frames, mobile_indices)
        D_plot, r2, fit_t, fit_line = fit_diffusivity(full_msd, start_fit_ps, end_fit_ps, 
                                                       TIMESTEP_FS, LOG_INTERVAL)
        
        # Calculate Conductivity
        vol = frames[0].get_volume()
        sigma = calculate_conductivity(D_avg, T, vol, n_mobile)
        
        results[T] = {
            'D_mean': D_avg,
            'D_std': D_std,
            'sigma': sigma,
            'r2': r2
        }

        # Plotting
        exponent = math.floor(math.log10(abs(D_avg))) if D_avg != 0 else 0
        factor = 10**exponent
        
        val_scaled = D_avg / factor
        err_scaled = D_std / factor

        png_name = f"msd_{T}K.png"
        title_str = fr"$T={T}\,\mathrm{{K}}, D=({val_scaled:.2f} \pm {err_scaled:.2f}) \times 10^{{{exponent}}}\,\mathrm{{cm}}^2/\mathrm{{s}}, R^2={r2:.3f}$"
        plot_msd(full_msd, fit_t, fit_line, title_str, png_name, 
                 f"{start_fit_ps}-{end_fit_ps} ps", end_fit_ps)

    # Arrhenius Plot with Error Bars
    if len(results) >= 3:
        sorted_temps = sorted(results.keys())
        temps_arr = np.array(sorted_temps)
        Ds_arr = np.array([results[T]['D_mean'] for T in sorted_temps])
        D_errs = np.array([results[T]['D_std'] for T in sorted_temps])
        
        inv_T = 1.0 / temps_arr
        ln_D = np.log(Ds_arr)
        # Error propagation: delta(ln D) = delta D / D
        ln_D_err = D_errs / Ds_arr
        
        slope, intercept, r_val, _, _ = linregress(inv_T, ln_D)
        ea_ev = (-slope * k_B) / e_charge
        
        plt.figure(figsize=(8, 6))
        plt.errorbar(1000.0/temps_arr, ln_D, yerr=ln_D_err, fmt='ko', 
                     markersize=10, capsize=5, label="Data w/ Error Bars")
        plt.plot(1000.0/temps_arr, intercept + slope * inv_T, 'r--', label="Arrhenius Fit")
        
        plt.xlabel("1000 / T (K$^{-1}$)")
        plt.ylabel("ln(D)")
        plt.title(f"Arrhenius Plot: $E_a$ = {ea_ev:.3f} eV")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("arrhenius_fit.png", dpi=300)
        
        print("\n--- Summary Results ---")
        for T in sorted_temps:
            res = results[T]
            print(f"T = {T}K: D = {res['D_mean']:.2e} +/- {res['D_std']:.2e} cm2/s, Sigma = {res['sigma']:.4e} S/cm")
        print(f"\nFinal Fit: Ea = {ea_ev:.4f} eV, R2 = {r_val**2:.4f}")

if __name__ == "__main__":
    main()
