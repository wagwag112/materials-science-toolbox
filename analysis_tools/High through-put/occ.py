import sys
import numpy as np
from ase.io import read, write
from ase import Atoms
import os


# =============================================================================
# PBC / FPS utilities
# =============================================================================

def get_pbc_distances(target_point, all_points, cell):
    inv_cell = np.linalg.inv(cell)
    frac_target = np.dot(target_point, inv_cell)
    frac_all    = np.dot(all_points,   inv_cell)
    delta = frac_all - frac_target
    delta -= np.round(delta)
    delta_cart = np.dot(delta, cell)
    return np.sqrt(np.sum(delta_cart ** 2, axis=1))


def fps_select(coords, cell, n, anchor_coords=None):
    # FPS under PBC.
    # anchor_coords: extra points already "occupied" -- new selection
    # will try to be as far from these as possible too.
    if n <= 0:
        return []
    if n >= len(coords):
        return list(range(len(coords)))

    INF = 1e18
    min_dists = np.full(len(coords), INF)

    if anchor_coords is not None and len(anchor_coords) > 0:
        for pt in anchor_coords:
            d = get_pbc_distances(pt, coords, cell)
            min_dists = np.minimum(min_dists, d)
        first = int(np.argmax(min_dists))
    else:
        first = 0
        min_dists = get_pbc_distances(coords[0], coords, cell)

    selected = [first]
    min_dists = np.minimum(
        min_dists,
        get_pbc_distances(coords[first], coords, cell)
    )

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
    # Usage:
    #   python occ.py V <occ> <start> <end> [M <elem> <occ> <start> <end> ...]
    #
    # V and M sharing the same range means:
    #   - V removes (1-occ_V)*n atoms via FPS
    #   - M places new element into subset of those removed slots via FPS
    #   - remaining removed slots = true vacancies
    #
    # Example:
    #   python occ.py V 0.7 1 10 M Mn 0.2 1 10
    #   -> 10 Fe sites: remove 3, put Mn in 2 of them, 1 true vacancy

    vasp_file = find_vasp_file()
    tokens = argv[1:]

    vacancy_ops = []   # list of (occ, start, end)
    mixed_ops   = []   # list of (elem, occ, start, end)

    i = 0
    while i < len(tokens):
        mode = tokens[i].upper()

        if mode == "V":
            try:
                occ   = float(tokens[i + 1])
                start = int(tokens[i + 2]) - 1
                end   = int(tokens[i + 3]) - 1
            except Exception:
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
            except Exception:
                print("ERROR: M requires: elem occ start end")
                sys.exit(1)
            mixed_ops.append((elem, occ, start, end))
            i += 5

        else:
            print("ERROR: Unknown mode '%s'. Use V or M." % tokens[i])
            sys.exit(1)

    return vasp_file, vacancy_ops, mixed_ops


# =============================================================================
# Main
# =============================================================================

def main():
    vasp_file, vacancy_ops, mixed_ops = parse_args(sys.argv)

    atoms     = read(vasp_file)
    cell      = atoms.cell[:]
    positions = atoms.get_positions()
    symbols   = list(atoms.get_chemical_symbols())
    n_atoms   = len(atoms)

    print("\n  -> Loaded: %s (%d atoms)" % (vasp_file, n_atoms))

    # deleted_pool[range_key] = list of global indices removed by V in that range
    # range_key = (start, end) so M can look up matching V pool
    deleted_pool = {}   # (start, end) -> [global_idx, ...]
    delete_set   = set()
    substitute_map = {}  # global_idx -> new_elem

    # -------------------------------------------------------------------------
    # VACANCY pass: FPS-select atoms to remove, build deleted_pool
    # -------------------------------------------------------------------------
    for (occ, start, end) in vacancy_ops:
        if start < 0 or end >= n_atoms or start > end:
            print("ERROR: V range [%d, %d] invalid." % (start+1, end+1))
            sys.exit(1)

        range_indices = list(range(start, end + 1))
        range_coords  = positions[range_indices]
        n_sites       = len(range_indices)
        n_remove      = int(round((1.0 - occ) * n_sites))

        print("  -> V: occ=%.3f, total=%d, removing %d atoms from [%d-%d]"
              % (occ, n_sites, n_remove, start+1, end+1))

        if n_remove <= 0:
            deleted_pool[(start, end)] = []
            continue

        sel_local  = fps_select(range_coords, cell, n_remove)
        sel_global = [range_indices[i] for i in sel_local]

        for g in sel_global:
            print("       atom %5d (%s) marked for removal" % (g+1, symbols[g]))

        delete_set.update(sel_global)
        deleted_pool[(start, end)] = sel_global

    # -------------------------------------------------------------------------
    # MIXED pass: place new element into subset of V-deleted slots
    # FPS for M is aware of the kept atoms in the same range (anchors)
    # -------------------------------------------------------------------------
    for (elem, occ, start, end) in mixed_ops:
        if start < 0 or end >= n_atoms or start > end:
            print("ERROR: M range [%d, %d] invalid." % (start+1, end+1))
            sys.exit(1)

        # find matching V pool for this range
        pool_global = deleted_pool.get((start, end), None)
        if pool_global is None:
            print("ERROR: M [%d-%d] has no matching V with same range."
                  % (start+1, end+1))
            sys.exit(1)

        n_sites  = end - start + 1
        n_place  = int(round(occ * n_sites))

        print("  -> M: elem=%s, occ=%.3f, placing %d atoms into V-pool of %d slots [%d-%d]"
              % (elem, occ, n_place, len(pool_global), start+1, end+1))

        if n_place <= 0:
            continue

        if n_place > len(pool_global):
            print("WARNING: M requests %d slots but V only freed %d. "
                  "Clamping to %d." % (n_place, len(pool_global), len(pool_global)))
            n_place = len(pool_global)

        # pool coords = positions of V-deleted slots
        pool_coords = positions[pool_global]

        # anchors = positions of atoms KEPT in the same range (not deleted)
        kept_in_range = [i for i in range(start, end+1) if i not in delete_set]
        anchor_coords = positions[kept_in_range] if kept_in_range else None

        sel_local  = fps_select(pool_coords, cell, n_place,
                                anchor_coords=anchor_coords)
        sel_global = [pool_global[i] for i in sel_local]

        for g in sel_global:
            print("       atom %5d: vacancy -> %s" % (g+1, elem))

        for g in sel_global:
            substitute_map[g] = elem

    # -------------------------------------------------------------------------
    # APPLY: atoms that are in delete_set but got substituted -> keep + rename
    #        atoms in delete_set and NOT substituted -> true vacancy, remove
    # -------------------------------------------------------------------------
    true_vacancies = delete_set - set(substitute_map.keys())

    new_symbols = symbols[:]
    for idx, new_elem in substitute_map.items():
        new_symbols[idx] = new_elem

    keep_indices    = [i for i in range(n_atoms) if i not in true_vacancies]
    final_positions = positions[keep_indices]
    final_symbols   = [new_symbols[i] for i in keep_indices]

    final_atoms = Atoms(
        symbols=final_symbols,
        positions=final_positions,
        cell=atoms.cell,
        pbc=atoms.pbc
    )

    print("\n  -> Original atoms : %d" % n_atoms)
    print("  -> True vacancies : %d" % len(true_vacancies))
    print("  -> Substituted    : %d" % len(substitute_map))
    print("  -> Final atoms    : %d" % len(final_atoms))

    name_no_ext = os.path.splitext(os.path.basename(vasp_file))[0]
    output_name = "%s_occ.vasp" % name_no_ext
    write(output_name, final_atoms, format="vasp")
    print("  -> SUCCESS: saved %s\n" % output_name)


if __name__ == "__main__":
    main()
