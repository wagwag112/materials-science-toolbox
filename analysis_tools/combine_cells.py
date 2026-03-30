import sys
import numpy as np
from ase.io import read, write
from ase import Atoms


def prepare_and_align(atoms, stack_axis):
    """
    Aligns the stacking axis to Z and cleans the cell to 90 degrees.
    """
    idx_map = {'x': 0, 'y': 1, 'z': 2}
    src_idx = idx_map[stack_axis.lower()]

    pos = atoms.get_positions()
    cell_lengths = atoms.get_cell_lengths_and_angles()[:3]

    new_pos = pos.copy()
    new_lengths = list(cell_lengths)

    if src_idx != 2:
        new_pos[:, [2, src_idx]] = new_pos[:, [src_idx, 2]]
        new_lengths[2], new_lengths[src_idx] = new_lengths[src_idx], new_lengths[2]

    new_atoms = Atoms(symbols=atoms.get_chemical_symbols(),
                      positions=new_pos,
                      pbc=True)
    new_atoms.set_cell([new_lengths[0], new_lengths[1], new_lengths[2], 90, 90, 90])

    # Normalize: shift so min X=0, min Y=0, min Z=0
    new_pos = new_atoms.get_positions()
    xyz_min = new_pos.min(axis=0)
    new_atoms.translate([-xyz_min[0], -xyz_min[1], -xyz_min[2]])

    return new_atoms


def center_xy(atoms, target_cx, target_cy):
    """
    Translate atoms so their XY bounding-box center aligns with (target_cx, target_cy).
    """
    pos = atoms.get_positions()
    current_cx = (pos[:, 0].min() + pos[:, 0].max()) / 2
    current_cy = (pos[:, 1].min() + pos[:, 1].max()) / 2
    dx = target_cx - current_cx
    dy = target_cy - current_cy
    atoms.translate([dx, dy, 0])
    return atoms


def merge_pro(f1, ax1, f2, ax2, gap=2.0, vacuum=0.0):
    print(f"--- Pro Merging: {f1}({ax1}) + {f2}({ax2}) ---")

    # 1. Load and Align
    s1 = prepare_and_align(read(f1), ax1)
    s2 = prepare_and_align(read(f2), ax2)

    # 2. Measure atomic thickness of each slab
    thick1 = s1.get_positions()[:, 2].max()
    thick2 = s2.get_positions()[:, 2].max()

    # 3. Lateral dimensions, take maximum of both for final cell
    cell1 = s1.get_cell_lengths_and_angles()
    cell2 = s2.get_cell_lengths_and_angles()
    max_x = max(cell1[0], cell2[0])
    max_y = max(cell1[1], cell2[1])

    # 4. Final cell height
    total_z = thick1 + gap + thick2 + 2 * vacuum
    block_start = vacuum

    # 5. Position slabs along Z
    s1.translate([0, 0, block_start])
    s2.translate([0, 0, block_start + thick1 + gap])

    # 6. Center both slabs to the XY center of the final cell
    cell_cx = max_x / 2
    cell_cy = max_y / 2

    s1 = center_xy(s1, cell_cx, cell_cy)
    s2 = center_xy(s2, cell_cx, cell_cy)

    # 7. Combine
    combined = s1 + s2
    combined.set_cell([max_x, max_y, total_z, 90, 90, 90])

    # 8. Verification report
    s1_pos = s1.get_positions()
    s2_pos = s2.get_positions()
    cx1 = (s1_pos[:, 0].min() + s1_pos[:, 0].max()) / 2
    cy1 = (s1_pos[:, 1].min() + s1_pos[:, 1].max()) / 2
    cx2 = (s2_pos[:, 0].min() + s2_pos[:, 0].max()) / 2
    cy2 = (s2_pos[:, 1].min() + s2_pos[:, 1].max()) / 2
    print(f"\n  XY centering check:")
    print(f"  -> Cell XY center:    ({cell_cx:.3f}, {cell_cy:.3f}) A")
    print(f"  -> Slab 1 XY center:  ({cx1:.3f}, {cy1:.3f}) A")
    print(f"  -> Slab 2 XY center:  ({cx2:.3f}, {cy2:.3f}) A")

    output_name = f"MERGED_{ax1}_{ax2}_G{gap}_V{vacuum}.vasp"
    write(output_name, combined, format='vasp', direct=True, sort=True)

    print(f"\nSUCCESS!")
    print(f"  -> Slab 1 thickness:   {thick1:.3f} A")
    print(f"  -> Slab 2 thickness:   {thick2:.3f} A")
    print(f"  -> Gap:                {gap:.3f} A")
    print(f"  -> Vacuum (each side): {vacuum:.3f} A")
    print(f"  -> Final cell:         {max_x:.3f} x {max_y:.3f} x {total_z:.3f} A")
    print(f"  -> Saved to:           {output_name}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python merge_pro.py <f1> <ax1> <f2> <ax2> [gap] [vacuum]")
        sys.exit(1)

    f1, ax1, f2, ax2 = sys.argv[1:5]
    gap = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    vac = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0

    merge_pro(f1, ax1, f2, ax2, gap, vac)
