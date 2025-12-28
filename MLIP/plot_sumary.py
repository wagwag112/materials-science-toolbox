#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Computational Materials Science Toolkit: Combined Arrhenius Plotter
-----------------------------------------------------------------
Author: Huy Hoang
Description: 
    Combines multiple diffusion datasets (from analysis_cache.json files) 
    into a single Arrhenius plot for comparison.

Usage:
    Modify the 'datasets' list in the __main__ block to point to your 
    json files, then run:
    python3 plot_combined.py
"""

import os
import sys
import json  # Added to read the new cache format
import numpy as np
import matplotlib
# Use Agg backend for stability on HPC (Headless mode)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- Plot Style Configuration ---
# Removed local Helvetica font setup. Using system default sans-serif.
plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 16,
    "axes.linewidth": 1.2,
    "font.family": "sans-serif" 
})

# --- Constants ---
k_B = 1.380649e-23      # Boltzmann constant (J/K)
e_charge = 1.602176634e-19  # Elementary charge (C)

def plot_combined_arrhenius(data_sources):
    """
    Reads JSON data sources, calculates Arrhenius fits, and plots them on a single figure.
    """
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # Generate distinct colors for each dataset
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_sources)))

    for source, color in zip(data_sources, colors):
        # Local variables for current dataset
        path = source["path"]
        label = source["label"]

        # --- NEW LOGIC: READ DATA FROM JSON ---
        try:
            with open(path, 'r') as f:
                data_dict = json.load(f)
            
            # Temporary lists to store parsed data
            temps = []
            diffs = []
            errors = []
            
            # iterate through JSON keys (Temperature strings)
            for t_str, values in data_dict.items():
                temps.append(float(t_str))
                diffs.append(values['D'])
                # Check if error exists, otherwise default to 0
                errors.append(values.get('D_err', 0.0))
            
            # Convert to Numpy arrays for vectorization
            T = np.array(temps)
            D = np.array(diffs)
            D_err = np.array(errors)
            
            # Sort by temperature (Crucial for correct line plotting)
            sort_idx = np.argsort(T)
            T = T[sort_idx]
            D = D[sort_idx]
            D_err = D_err[sort_idx]

        except Exception as e:
            print(f"[Error] Could not read {path}: {e}")
            continue
        # --------------------------------------

        # Filter invalid values (D must be positive for log scale)
        mask = (D > 0) & (T > 0)
        T, D, D_err = T[mask], D[mask], D_err[mask]

        if len(T) == 0: continue

        invT_1000 = 1000.0 / T
        
        # --- AUTOMATIC FITTING LOGIC (PRESERVED) ---
        display_label = label
        slope = 0
        intercept = 0
        r_value = 0
        
        # Perform linear regression if we have at least 2 points
        if len(T) >= 2:
            invT_real = 1.0 / T
            lnD = np.log(D)
            slope, intercept, r_value, _, _ = linregress(invT_real, lnD)
            
            # Calculate physical parameters
            Ea_eV = (-slope * k_B) / e_charge
            D0 = np.exp(intercept)
            
            # Create label with R^2 value
            display_label = fr"{label} ($\mathrm{{R}}^2 = {r_value**2:.2f}$)"

            # Generate smooth line for fitting visualization
            T_fit = np.linspace(T.min(), T.max(), 200)
            D_fit = D0 * np.exp(slope / T_fit) 

        # --- PLOTTING ---
        # Plot data points with error bars
        plt.errorbar(
            invT_1000, D, yerr=D_err, fmt="o", color=color,
            capsize=4, markersize=8,
            label=display_label 
        )

        # Plot fitting line (dashed)
        if len(T) >= 2:
            plt.plot(1000.0 / T_fit, D_fit, "--", color=color, alpha=0.85)
            # Print calculated values to terminal for reference
            print(f"Fit {label}: Ea = {Ea_eV:.4f} eV, D0 = {D0:.3e} cm^2/s, R2 = {r_value**2:.2f}")

    # --- Axis and Title Formatting ---
    ax.set_yscale("log")
    plt.xlabel(r"1000 / $T$ (K$^{-1}$)")
    plt.ylabel(r"$D$ (cm$^{2}$/s)")
    plt.title("Combined Arrhenius Plot")

    plt.grid(True, which="both", linestyle="--", alpha=0.25)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    # Save output
    output_name = "combined_arrhenius_plot.png"
    plt.savefig(output_name, dpi=300)
    print(f"--> Saved combined plot to: {output_name}")

if __name__ == "__main__":
    # USER CONFIGURATION:
    # Update these paths to point to your specific 'analysis_cache.json' files.
    # The 'label' field will appear in the plot legend.
    datasets = [
        {"path": "/your/path/analysis_cache.json", "label": "yours_label"},
        {"path": "...", "label": "..."},
        ...
    ]

    plot_combined_arrhenius(datasets)
