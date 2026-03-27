import sys
import numpy as np
from ase.io import read, write
import os


# =============================================================================
# PBC / FPS utilities
# =============================================================================

def get_pbc_distances(target_point, all_points, cell):
    """Minimum image convention distances from target to all_points."""
    inv_cell = np.linalg.inv(cell)
    frac_target = np.dot(target_point, inv_cell)
    frac_all    = np.dot(all_points,   inv_cell)
    delta = frac_all - frac_target
    delta -= np.round(delta)
    delta_cart = np.dot(delta, cell)
    return np.sqrt(np.sum(delta_cart ** 2, axis=1))


def fps_select(coords, cell, n):
    """Farthest Point Sampling under PBC."""
    if n <= 0:
        return []
    if n >= len(coords):
        return list(range(len(coords)))

    selected = [0]
    min_dists = get_pbc_distances(coords[0], coords, cell)

    for _ in range(n - 1):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        new_dists = get_pbc_distances(coords[next_idx], coords, cell)
        min_dists = np.minimum(min_dists, new_dists)

    return selected


# =============================================================================
# Auto-detect VASP file
# =============================================================================

def find_vasp_file():
    files = [f for f in os.listdir('.') if f.endswith('.vasp')]
    if len(files) == 0:
        print("ERROR: No .vasp file found in current directory.")
        sys.exit(1)
    elif len(files) > 1:
        print("ERROR: Multiple .vasp files found. Keep only one.")
        sys.exit(1)
    return files[0]


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args(argv):
    """
    Usage:
    python occ.py V <occ> <start> <end> [V ...] [M <elem> <occ> <start> <end> ...]

    Example:
    python occ.py V 0.92 1 8 M Ta 0.9 13 16
    """
    vasp_file = find_vasp_file()
    tokens = argv[1:]

    vacancy_ops = []
    mixed_ops   = []

    i = 0
    while i < len(tokens):
        mode = tokens[i].upper()

        if mode == "V":
            try:
                occ   = float(tokens[i + 1])
                start = int(tokens[i + 2]) - 1
                end   = int(tokens[i + 3]) - 1
            except:
                print("ERROR: V requires: occ start end")
                sys.exit(1)

            vacancy_ops.append((occ, start, end))
            i += 4

        elif mode == "M":
            try:
                elem  = tokens[i + 1]
                occ   = float(tokens[i + 2])
                start = int(tokens[i + 3]) - 1
                end   = int(tokens[i + 4]) - 1
            except:
                print("ERROR: M requires: elem occ start end")
                sys.exit(1)

            mixed_ops.append((elem, occ, start, end))
            i += 5

        else:
            print(f"ERROR: Unknown mode '{tokens[i]}'. Use V or M.")
            sys.exit(1)

    return vasp_file, vacancy_ops, mixed_ops


# =============================================================================
# Main
# =============================================================================

def main():
    vasp_file, vacancy_ops, mixed_ops = parse_args(sys.argv)

    if not os.path.exists(vasp_file):
        print(f"ERROR: File not found: {vasp_file}")
        sys.exit(1)

    atoms = read(vasp_file)
    cell  = atoms.cell[:]
    positions = atoms.get_positions()
    symbols   = list(atoms.get_chemical_symbols())
    n_atoms   = len(atoms)

    print(f"\n  -> Loaded: {vasp_file} ({n_atoms} atoms)")

    delete_indices = set()
    substitute_map = {}

    # -------------------------------------------------------------------------
    # VACANCY
    # -------------------------------------------------------------------------
    for (occ, start, end) in vacancy_ops:
        if start < 0 or end >= n_atoms or start > end:
            print(f"ERROR: V range [{start+1}, {end+1}] invalid.")
            sys.exit(1)

        range_indices = list(range(start, end + 1))
        range_coords  = positions[range_indices]

        n_sites = len(range_indices)
        n = int(round((1.0 - occ) * n_sites))

        print(f"  -> V: occ={occ:.3f}, total={n_sites}, deleting {n} atoms")

        if n <= 0:
            continue

        selected_local = fps_select(range_coords, cell, n)
        selected_global = [range_indices[i] for i in selected_local]

        for g in selected_global:
            print(f"       atom {g+1:>5} ({symbols[g]})")

        delete_indices.update(selected_global)

    # -------------------------------------------------------------------------
    # MIXED
    # -------------------------------------------------------------------------
    for (elem, occ, start, end) in mixed_ops:
        if start < 0 or end >= n_atoms or start > end:
            print(f"ERROR: M range [{start+1}, {end+1}] invalid.")
            sys.exit(1)

        range_indices = list(range(start, end + 1))
        range_coords  = positions[range_indices]

        n_sites = len(range_indices)
        n = int(round((1.0 - occ) * n_sites))

        print(f"  -> M: occ={occ:.3f}, total={n_sites}, replacing {n} atoms ? {elem}")

        if n <= 0:
            continue

        selected_local  = fps_select(range_coords, cell, n)
        selected_global = [range_indices[i] for i in selected_local]

        for g in selected_global:
            print(f"       atom {g+1:>5} ({symbols[g]} ? {elem})")

        for g in selected_global:
            substitute_map[g] = elem

    # -------------------------------------------------------------------------
    # APPLY
    # -------------------------------------------------------------------------
    new_symbols = symbols[:]
    for idx, new_elem in substitute_map.items():
        new_symbols[idx] = new_elem

    keep_indices = [i for i in range(n_atoms) if i not in delete_indices]

    final_positions = positions[keep_indices]
    final_symbols   = [new_symbols[i] for i in keep_indices]

    from ase import Atoms
    final_atoms = Atoms(
        symbols=final_symbols,
        positions=final_positions,
        cell=atoms.cell,
        pbc=atoms.pbc
    )

    print(f"\n  -> Original atoms : {n_atoms}")
    print(f"  -> Deleted        : {len(delete_indices)}")
    print(f"  -> Substituted    : {len(substitute_map)}")
    print(f"  -> Final atoms    : {len(final_atoms)}")

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------
    name_no_ext = os.path.splitext(os.path.basename(vasp_file))[0]
    output_name = f"{name_no_ext}_occ.vasp"

    write(output_name, final_atoms, format="vasp")
    print(f"  -> SUCCESS: saved {output_name}\n")


if __name__ == "__main__":
    main()
