"""
Author: Huy Hoang
Convert all .txt files in the current directory containing unit cell and atomic coordinates 
into CIF files. The .txt file format should have the space group on the first line, 
cell parameters on the next six lines, and atom data starting from line 8.

Usage:
    python txt_to_cif.py
"""

import os

def process_txt_to_cif():
    # Get all .txt files in the current directory
    files = [f for f in os.listdir('.') if f.endswith('.txt')]
    
    if not files:
        print("No .txt files found in the directory.")
        return

    for file_name in files:
        try:
            with open(file_name, 'r') as f:
                # Filter out empty lines
                lines = [line.strip() for line in f if line.strip()]

            if len(lines) < 8:
                print(f"Skipping {file_name}: Insufficient data.")
                continue

            # Parsing unit cell parameters (First 7 lines)
            # Line 0: Space Group (Removes "Space Group " prefix if exists)
            spgr = lines[0].replace("Space Group", "").strip()
            a, b, c = lines[1], lines[2], lines[3]
            alpha, beta, gamma = lines[4], lines[5], lines[6]

            # Output filename (same as text file but with .cif extension)
            base_name = os.path.splitext(file_name)[0]
            output_name = f"{base_name}.cif"

            with open(output_name, 'w') as out_f:
                # Write CIF Header
                out_f.write(f"data_{base_name}\n")
                out_f.write(f"_cell_length_a {a}\n")
                out_f.write(f"_cell_length_b {b}\n")
                out_f.write(f"_cell_length_c {c}\n")
                out_f.write(f"_cell_angle_alpha {alpha}\n")
                out_f.write(f"_cell_angle_beta {beta}\n")
                out_f.write(f"_cell_angle_gamma {gamma}\n")
                out_f.write(f"_symmetry_space_group_name_H-M '{spgr}'\n\n")

                # Write Atom Loop (Only first 5 columns)
                out_f.write("loop_\n")
                out_f.write("_atom_site_label\n")
                out_f.write("_atom_site_fract_x\n")
                out_f.write("_atom_site_fract_y\n")
                out_f.write("_atom_site_fract_z\n")
                out_f.write("_atom_site_occupancy\n")

                for line in lines[7:]:
                    parts = line.split()
                    if len(parts) >= 5:
                        # Extract only: Label, x, y, z, occ
                        label, x, y, z, occ = parts[0], parts[1], parts[2], parts[3], parts[4]
                        out_f.write(f"{label:<6} {x:>10} {y:>10} {z:>10} {occ:>8}\n")

            print(f"Successfully created: {output_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    process_txt_to_cif()
