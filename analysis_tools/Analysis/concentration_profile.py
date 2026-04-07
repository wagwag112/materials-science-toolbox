import sys
import os
import json
import re
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ase.io import read
from scipy.ndimage import gaussian_filter1d

# --- CONFIGURATION ---
AXIS = 2              # 0:X, 1:Y, 2:Z
SIGMA = 0.4           # Gaussian smoothing width
RESOLUTION = 0.1      # Grid spacing (Angstrom)
JSON_NAME = "interface_database.json"

def compute_smoothed_profile(atoms_list, symbols, axis=2, sigma=0.4, res=0.1):
    """Calculates number density (atoms/A^3) using Gaussian smearing."""
    first_atoms = atoms_list[0]
    lx = np.linalg.norm(first_atoms.get_cell()[axis])
    vol = first_atoms.get_volume()
    area = vol / lx
    grid = np.arange(0, lx, res)
    densities = {s: np.zeros_like(grid) for s in symbols}
    
    for atoms in atoms_list:
        syms = np.array(atoms.get_chemical_symbols())
        positions = atoms.get_positions()
        for s in symbols:
            coords = positions[syms == s][:, axis]
            if len(coords) == 0: continue
            counts, _ = np.histogram(coords, bins=np.append(grid, lx))
            smoothed = gaussian_filter1d(counts.astype(float), sigma/res, mode='nearest')
            densities[s] += (smoothed / (res * area))
            
    for s in symbols:
        densities[s] /= len(atoms_list)
    return grid, densities

def get_percentage(dens_dict):
    """Converts a dictionary of number densities to atomic percentages."""
    all_elements = list(dens_dict.keys())
    dens_matrix = np.array([dens_dict[s] for s in all_elements])
    total_dens = np.sum(dens_matrix, axis=0) + 1e-10
    return {s: (np.array(dens_dict[s])/total_dens)*100 for s in all_elements}

def main():
    if len(sys.argv) < 4:
        print("Usage: python script1.py <traj_file> <start> <end> [ref_elem1 ref_elem2 ...]")
        sys.exit(1)

    traj_path = sys.argv[1]
    s_idx, e_idx = int(sys.argv[2]), int(sys.argv[3])
    ref_elements = sys.argv[4:] # Elements to show dashed lines (t=0)

    # 1. Metadata Extraction
    temp_match = re.search(r'(\d+)K', os.path.basename(traj_path))
    temp_key = temp_match.group(0) if temp_match else "UnknownK"
    time_key = f"frame_{s_idx}_{e_idx}"

    # 2. Compute Current Profile
    print(f"  [LOG] Reading frames {s_idx} to {e_idx} from {traj_path}...")
    traj = read(traj_path, index=":")
    all_elements = sorted(list(set(traj[0].get_chemical_symbols())))
    grid, curr_dens = compute_smoothed_profile(traj[s_idx:e_idx+1], all_elements, axis=AXIS, sigma=SIGMA, res=RESOLUTION)
    curr_perc = get_percentage(curr_dens)

    # 3. Update JSON Database
    db = {}
    if os.path.exists(JSON_NAME):
        with open(JSON_NAME, 'r') as f: db = json.load(f)
    if temp_key not in db: db[temp_key] = {}
    db[temp_key][time_key] = {"grid": grid.tolist(), "profiles": {s: curr_dens[s].tolist() for s in all_elements}}
    with open(JSON_NAME, 'w') as f: json.dump(db, f, indent=2)
    print(f"  [LOG] Database updated in {JSON_NAME}")

    # 4. Search for Reference (t=0) in JSON
    ref_perc = None
    ref_label = ""
    # Look for any key starting with 'frame_0_' under the same temperature
    for key in db.get(temp_key, {}).keys():
        if key.startswith("frame_0_"):
            ref_data = db[temp_key][key]
            ref_perc = get_percentage(ref_data['profiles'])
            ref_label = key
            break

    # 5. Plotting
    plt.figure(figsize=(11, 6))
    
    # Plot SOLID lines (Current state - all elements)
    for s in all_elements:
        line, = plt.plot(grid, curr_perc[s], lw=2.0, label=f"{s}")
        
        # Plot DASHED lines (Reference state - only for selected elements)
        if ref_perc and s in ref_elements:
            plt.plot(grid, ref_perc[s], color=line.get_color(), ls='--', lw=1.5, alpha=0.7, 
                     label=f"{s} (ref)")

    # Formatting
    plt.xlabel(r"Distance along Z-axis ($\mathrm{\AA}$)", fontsize=12)
    plt.ylabel("Atomic Concentration (%)", fontsize=12)
    plt.title(f"Interface Analysis: {temp_key} at {time_key}", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(grid[0], grid[-1])
    plt.ylim(-5, 105)
    
    # Place legend outside
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, frameon=True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    ax.grid(True, which='major', linestyle='-', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.tight_layout()
    
    plot_name = f"analysis_{temp_key}_{time_key}.png"
    plt.savefig(plot_name, dpi=300)
    print(f"  [LOG] Success! Analysis plot saved as {plot_name}")

if __name__ == "__main__":
    main()
