#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import itertools
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
font_dir = r"/home/hoang0000/helvetica/"
font_prop = init_helvetica_font(font_dir)

# =============================================================================
# --- Configuration Section ---
# =============================================================================
TIMESTEP_FS   = 2.0    # MD timestep in femtoseconds
LOG_INTERVAL  = 50     # Steps per saved frame
DIM           = 3      # Dimensionality (3 for bulk/slab 3D)
MOBILE_ION_SYMBOL = "Li"
N_BLOCKS      = 4      # Number of blocks for error estimation

# --- Auto-fit sliding window parameters ---
BALLISTIC_PS      = 20.0   # Exclude early ballistic regime (ps)
WINDOW_MAX_PS     = 180.0  # Starting (largest) window size (ps)
WINDOW_MIN_PS     = 60.0   # Minimum window size before giving up (ps)
WINDOW_SIZE_STEP_PS = 1.0  # Step to shrink window size if R^2 not met (ps)
WINDOW_STEP_PS    = 1.0    # Sliding step between windows (ps)
MIN_R2            = 0.96   # Minimum R^2 threshold to accept a window

# --- Arrhenius best-subset selection ---
# Set N_ARRHENIUS_POINTS = None to use ALL available temperature points.
# Set N_ARRHENIUS_POINTS = 4 (or any integer >= 2) to select the N-point
# subset that maximises R^2 on the ln(sigma) vs 1/T Arrhenius fit.
N_ARRHENIUS_POINTS = 4

# --- Extrapolation target ---
EXTRAP_TEMP_K = 300    # Temperature to extrapolate D and sigma to (K)

# =============================================================================

# --- Constants ---
k_B      = 1.380649e-23       # J/K
e_charge = 1.602176634e-19    # C

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

# =============================================================================
# --- Core Functions ---
# =============================================================================

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
    pos_all  = np.array([f.get_positions() for f in frames])
    cell     = frames[0].get_cell()
    inv_cell = np.linalg.inv(cell)

    diffs      = np.diff(pos_all, axis=0)
    diffs_frac = np.einsum('ijk,kl->ijl', diffs, inv_cell)
    shifts_frac = -np.round(diffs_frac)
    shifts_cart = np.einsum('ijk,kl->ijl', shifts_frac, cell)

    unwrapped    = np.zeros_like(pos_all)
    unwrapped[0] = pos_all[0]
    unwrapped[1:] = pos_all[1:] + np.cumsum(shifts_cart, axis=0)

    return unwrapped

def subtract_com_drift(unwrapped_pos_all, framework_indices, target_indices):
    """Subtracts framework Center of Mass drift from target species."""
    framework_pos  = unwrapped_pos_all[:, framework_indices, :]
    com_framework  = np.mean(framework_pos, axis=1)
    drift          = com_framework - com_framework[0]

    target_pos     = unwrapped_pos_all[:, target_indices, :]
    corrected_pos  = target_pos - drift[:, np.newaxis, :]
    return corrected_pos

def compute_msd(corrected_pos):
    """Computes MSD for a segment using its local starting frame."""
    r0      = corrected_pos[0]
    dr      = corrected_pos - r0
    sq_dist = np.sum(dr**2, axis=2)
    msd     = np.mean(sq_dist, axis=1)
    return msd

def fit_diffusivity(msd, start_time_ps, end_time_ps, timestep_fs, log_interval):
    n_frames          = len(msd)
    time_per_frame_ps = (timestep_fs * log_interval) / 1000.0
    times_ps          = np.arange(n_frames) * time_per_frame_ps

    start_idx = np.searchsorted(times_ps, start_time_ps, side='left')
    end_idx   = np.searchsorted(times_ps, end_time_ps,   side='right')

    if end_idx - start_idx < 2:
        return 0.0, 0.0, np.array([]), np.array([])

    x_fit = times_ps[start_idx:end_idx]
    y_fit = msd[start_idx:end_idx]

    slope, intercept, r_val, _, _ = linregress(x_fit, y_fit)
    D_cm2_s  = (slope * 1e-4) / (2 * DIM)
    fit_line = intercept + slope * x_fit

    return D_cm2_s, r_val**2, x_fit, fit_line

def calculate_conductivity(D_cm2_s, D_std, temp_k, volume_ang3, n_ions):
    vol_cm3   = volume_ang3 * 1e-24
    n_density = n_ions / vol_cm3
    q         = e_charge
    sigma     = (n_density * (q**2) * D_cm2_s) / (k_B * temp_k)
    sigma_std = sigma * (D_std / D_cm2_s) if D_cm2_s > 0 else 0
    return sigma, sigma_std

def auto_find_best_window(msd, timestep_fs, log_interval,
                          ballistic_ps, window_step_ps):
    time_per_frame_ps = (timestep_fs * log_interval) / 1000.0
    n_frames          = len(msd)
    total_time_ps     = (n_frames - 1) * time_per_frame_ps
    times_ps          = np.arange(n_frames) * time_per_frame_ps

    best_r2    = -1.0
    best_start = ballistic_ps
    best_end   = ballistic_ps + WINDOW_MAX_PS

    window_size = WINDOW_MAX_PS
    while window_size >= WINDOW_MIN_PS:
        t_start = ballistic_ps
        while t_start + window_size <= total_time_ps:
            t_end = t_start + window_size
            si = np.searchsorted(times_ps, t_start, side='left')
            ei = np.searchsorted(times_ps, t_end,   side='right')

            if ei - si >= 4:
                x = times_ps[si:ei]
                y = msd[si:ei]
                slope, _, r_val, _, _ = linregress(x, y)
                r2 = r_val**2
                if slope > 0 and r2 > best_r2:
                    best_r2    = r2
                    best_start = t_start
                    best_end   = t_end

            t_start += window_step_ps

        if best_r2 >= MIN_R2:
            print(f"  Window size {window_size:.0f} ps -> R^2={best_r2:.5f} (accepted)")
            return best_start, best_end, best_r2

        print(f"  Window size {window_size:.0f} ps -> best R^2={best_r2:.5f} < {MIN_R2}, shrinking...")
        window_size -= WINDOW_SIZE_STEP_PS

    print(f"  Warning: No window met MIN_R2={MIN_R2}. Using global best R^2={best_r2:.5f} "
          f"at [{best_start:.0f}-{best_end:.0f}] ps.")
    return best_start, best_end, best_r2

def plot_msd(msd, fit_t, fit_line, title, filename, fit_range_str, end_time_ps):
    time_per_frame_ps = (TIMESTEP_FS * LOG_INTERVAL) / 1000.0
    times_ps          = np.arange(len(msd)) * time_per_frame_ps
    idx_limit         = np.searchsorted(times_ps, end_time_ps, side='right')

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

# =============================================================================
# --- Arrhenius best-subset selection ---
# =============================================================================

def r2_linear(x, y):
    """R^2 for OLS linear fit on arrays x, y."""
    A    = np.vstack([x, np.ones(len(x))]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = A @ coef
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else -1.0


def select_best_arrhenius_subset(sorted_temps, results, n_select, quantity="sigma"):
    """
    From all available temperature points, select the n_select-point subset
    that maximises R^2 on the Arrhenius fit:
        ln(sigma)  vs  1/T   (quantity='sigma')
        ln(D)      vs  1/T   (quantity='D')

    Parameters
    ----------
    sorted_temps : list of int
        Temperatures (K) with valid results, in ascending order.
    results      : dict
        Output dict from main loop keyed by temperature.
    n_select     : int
        Number of points to keep (must be 2 <= n_select <= len(sorted_temps)).
    quantity     : str
        'sigma' or 'D' -- which Arrhenius quantity to optimise for.

    Returns
    -------
    best_temps   : list of int
        Selected temperatures in ascending order.
    best_r2      : float
        R^2 of the best subset.
    """
    n_total = len(sorted_temps)
    if n_select is None or n_select >= n_total:
        return sorted_temps, None   # use all points

    if n_select < 2:
        raise ValueError("N_ARRHENIUS_POINTS must be >= 2.")

    inv_T_all = np.array([1.0 / t for t in sorted_temps])
    if quantity == "sigma":
        y_all = np.array([np.log(results[t]['sigma']) for t in sorted_temps])
    else:
        y_all = np.array([np.log(results[t]['D'])     for t in sorted_temps])

    best_r2  = -np.inf
    best_idx = None

    for idx in itertools.combinations(range(n_total), n_select):
        idx = list(idx)
        r2  = r2_linear(inv_T_all[idx], y_all[idx])
        if r2 > best_r2:
            best_r2  = r2
            best_idx = idx

    best_temps = [sorted_temps[i] for i in best_idx]
    return best_temps, best_r2


# =============================================================================
# --- Main ---
# =============================================================================

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage:\n"
                 "  --full <t1> <t2> ...\n"
                 "  --slab <z_min> <z_max> <t1> <t2> ...")

    if sys.argv[1] == "--full":
        mode, temps, z_min, z_max, suffix = "FULL", [int(t) for t in sys.argv[2:]], None, None, "_full"
        print("\n--> Mode: FULL CELL analysis.")
    elif sys.argv[1] == "--slab":
        mode, z_min, z_max = "SLAB", float(sys.argv[2]), float(sys.argv[3])
        temps, suffix = [int(t) for t in sys.argv[4:]], f"_z{int(z_min)}-{int(z_max)}"
        print(f"\n--> Mode: SLAB analysis in Z-range [{z_min:.1f}, {z_max:.1f}].")
    else:
        sys.exit("Error: Invalid flag.")

    print(f"--> Auto-fit config: ballistic={BALLISTIC_PS} ps | window={WINDOW_MIN_PS}-{WINDOW_MAX_PS} ps | size_step={WINDOW_SIZE_STEP_PS} ps\n")

    results = {}
    last_vol, last_n = 0, 0

    for T in temps:
        traj_file = f"md_{T}K.traj"
        if not os.path.isfile(traj_file): continue

        print(f"Processing {T}K...")
        frames = load_trajectory(traj_file)
        cell = frames[0].get_cell()
        xy_area = np.linalg.norm(np.cross(cell[0], cell[1]))
        pos_t0 = frames[0].get_positions()
        mobile_set = {atom.index for atom in frames[0] if atom.symbol == MOBILE_ION_SYMBOL}
        framework_indices = np.array([idx for idx in np.arange(len(frames[0])) if idx not in mobile_set])

        if mode == "SLAB":
            z_mask = (pos_t0[:, 2] >= z_min) & (pos_t0[:, 2] <= z_max)
            target_indices = np.array(sorted([idx for idx in mobile_set if z_mask[idx]]))
            vol_eff = xy_area * (z_max - z_min)
        else:
            target_indices = np.array(sorted(list(mobile_set)))
            vol_eff = frames[0].get_volume()

        n_target = len(target_indices)
        if n_target == 0: continue
        last_vol, last_n = vol_eff, n_target

        unwrapped_all = unwrap_all_atoms_vectorized(frames)
        corrected_target_all = subtract_com_drift(unwrapped_all, framework_indices, target_indices)
        full_msd = compute_msd(corrected_target_all)

        start_fit_ps, end_fit_ps, window_r2 = auto_find_best_window(full_msd, TIMESTEP_FS, LOG_INTERVAL, BALLISTIC_PS, WINDOW_STEP_PS)

        time_per_frame_ps = (TIMESTEP_FS * LOG_INTERVAL) / 1000.0
        start_idx = int(start_fit_ps / time_per_frame_ps)
        end_idx   = int(end_fit_ps   / time_per_frame_ps)
        fit_pos   = corrected_target_all[start_idx:end_idx]
        block_size = len(fit_pos) // N_BLOCKS

        block_diffusivities = []
        for b in range(N_BLOCKS):
            block_seg = fit_pos[b*block_size : (b+1)*block_size]
            if len(block_seg) < 2: continue
            b_msd = compute_msd(block_seg)
            D, _, _, _ = fit_diffusivity(b_msd, 0.0, (block_size*time_per_frame_ps)*0.9, TIMESTEP_FS, LOG_INTERVAL)
            if D > 0: block_diffusivities.append(D)

        if not block_diffusivities: continue
        D_avg, D_std = np.mean(block_diffusivities), np.std(block_diffusivities) / np.sqrt(len(block_diffusivities))
        sigma, sigma_std = calculate_conductivity(D_avg, D_std, T, vol_eff, n_target)

        results[T] = {'D': D_avg, 'D_std': D_std, 'sigma': sigma, 'sigma_std': sigma_std,
                      'r2': window_r2, 'fit_start': start_fit_ps, 'fit_end': end_fit_ps}

        # --- Plot entire MSD trajectory ---
        total_time_ps = (len(full_msd) - 1) * time_per_frame_ps
        D_plot, r2, fit_t, fit_line = fit_diffusivity(full_msd, start_fit_ps, end_fit_ps, TIMESTEP_FS, LOG_INTERVAL)
        exponent = math.floor(math.log10(abs(D_avg))) if D_avg != 0 else 0
        factor = 10**exponent
        title_str = (fr"$T={T}\,\mathrm{{K}},\;D=({D_avg/factor:.2f}\pm{D_std/factor:.2f})\times10^{{{exponent}}}\,\mathrm{{cm}}^2/\mathrm{{s}}$")
        plot_msd(full_msd, fit_t, fit_line, title_str, f"msd_{T}K{suffix}.png", f"{start_fit_ps:.0f}-{end_fit_ps:.0f} ps", total_time_ps)

    # =========================================================================
    # --- Arrhenius + Extrapolation ---
    # =========================================================================
    if len(results) < 2:
        print("Not enough temperature points for Arrhenius fit.")
        return

    all_sorted_temps = sorted(results.keys())

    # --- Best-subset selection on ln(sigma) vs 1/T ---
    n_avail  = len(all_sorted_temps)
    n_select = N_ARRHENIUS_POINTS

    if n_select is not None and n_select < n_avail:
        print(f"\n--> Selecting best {n_select} out of {n_avail} points for Arrhenius fit "
              f"(optimising R^2 on ln(sigma) vs 1/T) ...")
        selected_temps, subset_r2 = select_best_arrhenius_subset(
            all_sorted_temps, results, n_select, quantity="sigma")
        excluded_temps = [t for t in all_sorted_temps if t not in selected_temps]
        print(f"    Selected temperatures : {selected_temps} K")
        print(f"    Excluded temperatures : {excluded_temps} K")
        print(f"    Subset R^2 (sigma)    : {subset_r2:.6f}")
    else:
        selected_temps = all_sorted_temps
        subset_r2      = None
        print(f"\n--> Using all {n_avail} points for Arrhenius fit.")

    sorted_temps = selected_temps

    # --- Arrhenius linear regression on selected points ---
    inv_T = 1.0 / np.array(sorted_temps, float)
    ln_D  = np.log([results[t]['D']     for t in sorted_temps])
    ln_sig = np.log([results[t]['sigma'] for t in sorted_temps])

    slope, intercept, r_val, _, std_err_slope = linregress(inv_T, ln_D)

    # Residual std for prediction interval
    n_pts = len(sorted_temps)
    ln_D_pred = intercept + slope * inv_T
    residuals  = ln_D - ln_D_pred
    resid_std  = np.sqrt(np.sum(residuals**2) / (n_pts - 2)) if n_pts > 2 else np.nan

    x0         = 1.0 / EXTRAP_TEMP_K
    inv_T_mean = np.mean(inv_T)
    SS_xx      = np.sum((inv_T - inv_T_mean)**2)
    D_extrap   = np.exp(intercept + slope * x0)

    if not np.isnan(resid_std):
        se_pred      = resid_std * np.sqrt(1.0/n_pts + (x0 - inv_T_mean)**2 / SS_xx)
        D_extrap_err = D_extrap * se_pred
    else:
        D_extrap_err = 0.0

    ea_ev, ea_err_ev = (-slope * k_B) / e_charge, (std_err_slope * k_B) / e_charge
    sigma_extrap, sigma_extrap_std = calculate_conductivity(
        D_extrap, D_extrap_err, EXTRAP_TEMP_K, last_vol, last_n)

    # --- Summary Output ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY ({mode}) -- ALL {n_avail} computed points")
    print(f"{'='*70}")
    print(f"  {'T (K)':<10} {'D (cm^2/s)':<26} {'sigma (S/cm)':<26} {'MSD R^2':<10} {'Fit range (ps)':<20} {'Used'}")
    print(f"  {'-' * 105}")
    for T in all_sorted_temps:
        r    = results[T]
        used = "YES" if T in sorted_temps else "NO (excluded)"
        print(f"  {T:<10} "
              f"{r['D']:.2e} +/- {r['D_std']:.1e}   "
              f"{r['sigma']:.4e} +/- {r['sigma_std']:.1e}   "
              f"{r['r2']:.5f}   "
              f"[{r['fit_start']:.0f} - {r['fit_end']:.0f}]"
              f"{'':>8}{used}")
    print(f"  {'-' * 105}")

    print(f"\n  Arrhenius fit on {n_pts} selected points:")
    err_str = f"{D_extrap_err:.1e}" if not np.isnan(resid_std) else "N/A"
    print(f"  {EXTRAP_TEMP_K}K (extrap)  D = {D_extrap:.2e} +/- {err_str} | sigma = {sigma_extrap:.4e} +/- {sigma_extrap_std:.1e}")
    print(f"\n  Ea = {ea_ev:.4f} +/- {ea_err_ev:.4f} eV")
    print(f"  Arrhenius R^2 (D)     = {r_val**2:.6f}")
    if subset_r2 is not None:
        print(f"  Arrhenius R^2 (sigma) = {subset_r2:.6f}  [used for subset selection]")
    print(f"{'='*70}")

    # --- Arrhenius Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot excluded points (grey, open markers) if any
    excluded_temps_plot = [t for t in all_sorted_temps if t not in sorted_temps]
    if excluded_temps_plot:
        inv_T_excl = 1000.0 / np.array(excluded_temps_plot, float)
        ln_D_excl  = [np.log(results[t]['D']) for t in excluded_temps_plot]
        yerr_excl  = [results[t]['D_std'] / results[t]['D'] for t in excluded_temps_plot]
        ax.errorbar(inv_T_excl, ln_D_excl, yerr=yerr_excl,
                    fmt='o', color='grey', markerfacecolor='none',
                    capsize=5, label='Excluded points', zorder=2)

    # Plot selected points (black, filled markers)
    inv_T_sel = 1000.0 / np.array(sorted_temps, float)
    ln_D_sel  = [np.log(results[t]['D']) for t in sorted_temps]
    yerr_sel  = [results[t]['D_std'] / results[t]['D'] for t in sorted_temps]
    ax.errorbar(inv_T_sel, ln_D_sel, yerr=yerr_sel,
                fmt='ko', capsize=5, label='Selected points', zorder=3)

    # Fit line spanning selected range only
    x_smooth = np.linspace(min(inv_T), max(inv_T), 200)
    ax.plot(1000.0 * x_smooth, intercept + slope * x_smooth, 'r--',
            label=fr"Fit: $E_a$={ea_ev:.3f} eV, $R^2$={r_val**2:.4f}")

    ax.set_xlabel(r"1000 / $T$ (K$^{-1}$)")
    ax.set_ylabel(r"ln($D$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"arrhenius{suffix}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
