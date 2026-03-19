"""
Automates the generation of crystal configurations with vacancies by 
interpreting site-specific occupancy (OCC) from CIF files.
Usage:
python occ.py <yours>.cif
"""
import numpy as np
import pandas as pd
import re
import io
import glob
import os
import sys

def get_standard_lattice(a, b, c, alpha, beta, gamma):
    """Constructs a standard right-handed lattice matrix."""
    alpha, beta, gamma = np.radians([alpha, beta, gamma])
    ax = a
    ay, az = 0.0, 0.0
    bx, by, bz = b * np.cos(gamma), b * np.sin(gamma), 0.0
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(0, c**2 - cx**2 - cy**2))
    return np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]])

def process_single_cif(input_file, num_configs=1, seed=42):
    print(f"\n{'='*50}")
    print(f"PROCESSING: {input_file}")
    print(f"{'='*50}")
    
    with open(input_file, 'r') as f:
        content = f.read()

    # 1. Flexible Metadata Parsing
    def extract_val(tag):
        match = re.search(rf'{tag}\s+([\d\.\-\(\)]+)', content)
        if not match: return None
        return float(re.sub(r'\(.*\)', '', match.group(1)))

    lattice_params = [
        extract_val('_cell_length_a'), extract_val('_cell_length_b'), extract_val('_cell_length_c'),
        extract_val('_cell_angle_alpha'), extract_val('_cell_angle_beta'), extract_val('_cell_angle_gamma')
    ]
    lat_matrix = get_standard_lattice(*lattice_params)

    # 2. Extract Atom Site Block
    blocks = content.split('loop_')
    atom_block = next(( 'loop_' + b for b in blocks if '_atom_site_label' in b), None)
    
    if not atom_block:
        print(f"Error: Missing _atom_site loop.")
        return

    lines = [l.strip() for l in atom_block.split('\n') if l.strip()]
    header_lines = [l for l in lines if l.startswith('_atom_site_')]
    data_lines = [l for l in lines if not l.startswith('_') and not l.startswith('loop_')]
    
    cols = [h.replace('_atom_site_', '') for h in header_lines]
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=r'\s+', names=cols, engine='python').reset_index(drop=True)
    
    for col in ['fract_x', 'fract_y', 'fract_z', 'occupancy']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'\(.*\)', '', regex=True).replace('?', '1.0').astype(float)

    all_coords = df[['fract_x', 'fract_y', 'fract_z']].values
    base_name = os.path.splitext(input_file)[0]

    # 3. Ensemble Generation Logic
    for n in range(num_configs):
        rng = np.random.default_rng(seed + n)
        final_indices = []
        
        print(f"Configuration #{n+1} Defect Report:")
        print(f"{'Label':<10} | {'Total':<6} | {'OCC':<8} | {'Target':<8} | {'Removed':<8}")
        print("-" * 55)

        # CRITICAL FIX: Group by 'label' instead of 'species' to respect site-specific OCC
        for label, group in df.groupby('label'):
            indices = group.index.to_numpy()
            occ_val = group['occupancy'].iloc[0]
            total_atoms = len(group)
            
            # Scientific Rounding: N_keep = round(N_total * OCC)
            num_keep = int(round(total_atoms * occ_val))
            num_to_remove = total_atoms - num_keep
            
            print(f"{label:<10} | {total_atoms:<6} | {occ_val:<8.4f} | {num_keep:<8} | {num_to_remove:<8}")

            if num_to_remove <= 0:
                final_indices.extend(indices)
                continue
            
            # FPS Vacancy Selection
            min_dists_sq = np.full(len(indices), np.inf)
            removed_indices = []
            
            start_ptr = rng.integers(0, len(indices))
            removed_indices.append(indices[start_ptr])
            
            mask = np.ones(len(indices), dtype=bool)
            mask[start_ptr] = False
            
            for _ in range(num_to_remove - 1):
                last_removed_coord = all_coords[removed_indices[-1]]
                rem_coords = all_coords[indices[mask]]
                
                delta = rem_coords - last_removed_coord
                delta -= np.round(delta)
                cart_delta = np.dot(delta, lat_matrix)
                dists_sq = np.sum(cart_delta**2, axis=1)
                
                min_dists_sq[mask] = np.minimum(min_dists_sq[mask], dists_sq)
                next_ptr = np.argmax(min_dists_sq[mask])
                next_global_idx = indices[mask][next_ptr]
                
                removed_indices.append(next_global_idx)
                actual_loc = np.where(indices == next_global_idx)[0][0]
                mask[actual_loc] = False
            
            final_indices.extend(indices[mask])

        # 4. Save result
        res_df = df.loc[final_indices].copy()
        res_df['occupancy'] = 1.0
        
        formatted_rows = ["   " + "   ".join([f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in cols]) for _, row in res_df.iterrows()]
        new_loop = "\n".join(header_lines) + "\n" + "\n".join(formatted_rows)
        final_content = content.replace(atom_block, "loop_\n" + new_loop + "\n")
        
        output_name = f"{base_name}_vac_{n+1}.cif"
        with open(output_name, 'w') as f:
            f.write(final_content)
        print(f"\nSuccessfully saved: {output_name}\n")

if __name__ == "__main__":
    target_files = sys.argv[1:] if len(sys.argv) > 1 else [f for f in glob.glob("*.cif") if "_vac_" not in f]
    for file in target_files:
        if os.path.exists(file):
            process_single_cif(file, num_configs=1, seed=42)
