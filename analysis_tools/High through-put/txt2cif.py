import os

# Try to use spglib for symmetry operations
try:
    from pymatgen.symmetry.groups import SpaceGroup
    USE_PYMATGEN = True
except ImportError:
    USE_PYMATGEN = False
    print("Warning: pymatgen not found. Symmetry operations will not be written.")
    print("Install with: pip install pymatgen")

def get_symmetry_ops(spgr_symbol):
    """Get symmetry equivalent positions for a space group."""
    if not USE_PYMATGEN:
        return None, None
    try:
        sg = SpaceGroup(spgr_symbol)
        it_number = sg.int_number
        ops = [str(op.as_xyz_str()) for op in sg.symmetry_ops]
        return it_number, ops
    except Exception as e:
        print(f"  Warning: Could not resolve space group '{spgr_symbol}': {e}")
        return None, None

def process_txt_to_cif():
    files = [f for f in os.listdir('.') if f.endswith('.txt')]
    
    if not files:
        print("No .txt files found in the directory.")
        return

    for file_name in files:
        try:
            with open(file_name, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]

            if len(lines) < 8:
                print(f"Skipping {file_name}: Insufficient data.")
                continue

            # Parse header
            spgr = lines[0].replace("Space Group", "").strip()
            a, b, c = lines[1], lines[2], lines[3]
            alpha, beta, gamma = lines[4], lines[5], lines[6]

            base_name = os.path.splitext(file_name)[0]
            output_name = f"{base_name}.cif"

            # Get symmetry info
            it_number, sym_ops = get_symmetry_ops(spgr)

            with open(output_name, 'w') as out_f:
                # CIF Header
                out_f.write(f"data_{base_name}\n\n")
                out_f.write(f"_cell_length_a                  {a}\n")
                out_f.write(f"_cell_length_b                  {b}\n")
                out_f.write(f"_cell_length_c                  {c}\n")
                out_f.write(f"_cell_angle_alpha               {alpha}\n")
                out_f.write(f"_cell_angle_beta                {beta}\n")
                out_f.write(f"_cell_angle_gamma               {gamma}\n\n")

                # Symmetry block
                out_f.write(f"_symmetry_space_group_name_H-M  '{spgr}'\n")
                if it_number:
                    out_f.write(f"_symmetry_Int_Tables_number     {it_number}\n")
                
                # Write symmetry equivalent positions loop
                if sym_ops:
                    out_f.write("\nloop_\n")
                    out_f.write("_symmetry_equiv_pos_as_xyz\n")
                    for op in sym_ops:
                        out_f.write(f"  '{op}'\n")

                # Atom sites loop
                out_f.write("\nloop_\n")
                out_f.write("_atom_site_label\n")
                out_f.write("_atom_site_fract_x\n")
                out_f.write("_atom_site_fract_y\n")
                out_f.write("_atom_site_fract_z\n")
                out_f.write("_atom_site_occupancy\n")

                for line in lines[7:]:
                    parts = line.split()
                    if len(parts) >= 5:
                        label, x, y, z, occ = parts[0], parts[1], parts[2], parts[3], parts[4]
                        out_f.write(f"{label:<6} {x:>12} {y:>12} {z:>12} {occ:>8}\n")

            status = f"(SG #{it_number}, {len(sym_ops)} ops)" if it_number else "(no symmetry ops)"
            print(f"Created: {output_name} {status}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    process_txt_to_cif()
