#monoclinic
import numpy as np
from ase.io import read, write
from ase import Atoms


def shift_to_origin_along_b(atoms):
    """
    Shift structure so that minimum projection along b becomes 0.
    """

    bvec = atoms.get_cell()[1]
    b_hat = bvec / np.linalg.norm(bvec)

    proj = atoms.positions @ b_hat
    min_proj = proj.min()

    atoms.translate(-min_proj * b_hat)

    return atoms


def slab_thickness_along_b(atoms):
    """
    Physical thickness projected along b direction.
    """

    bvec = atoms.get_cell()[1]
    b_hat = bvec / np.linalg.norm(bvec)

    proj = atoms.positions @ b_hat

    return proj.max() - proj.min()


def merge_monoclinic_b(
    file1,
    file2,
    gap=2.0,
    vacuum=5.0,
    output="MERGED_MONOCLINIC.vasp"
):

    print("\n==============================")
    print(" Monoclinic slab merge along b")
    print("==============================\n")

    s1 = read(file1)
    s2 = read(file2)

    # -----------------------------------
    # Check lattice compatibility
    # -----------------------------------

    cell1 = s1.get_cell().array
    cell2 = s2.get_cell().array

    tol = 1e-3

    # compare a and c vectors
    if not np.allclose(cell1[0], cell2[0], atol=tol):
        raise ValueError("a vectors are not identical")

    if not np.allclose(cell1[2], cell2[2], atol=tol):
        raise ValueError("c vectors are not identical")

    print("Lattice check PASSED")
    print("a and c vectors match.\n")

    # -----------------------------------
    # Move slabs so bottom starts at 0
    # -----------------------------------

    s1 = shift_to_origin_along_b(s1)
    s2 = shift_to_origin_along_b(s2)

    # -----------------------------------
    # Thickness along b
    # -----------------------------------

    thick1 = slab_thickness_along_b(s1)
    thick2 = slab_thickness_along_b(s2)

    print(f"Slab 1 thickness along b = {thick1:.3f} Å")
    print(f"Slab 2 thickness along b = {thick2:.3f} Å")

    # -----------------------------------
    # Unit vector along b
    # -----------------------------------

    bvec = cell1[1]
    b_hat = bvec / np.linalg.norm(bvec)

    # -----------------------------------
    # Place slabs
    # -----------------------------------

    start1 = vacuum
    start2 = vacuum + thick1 + gap

    s1.translate(start1 * b_hat)
    s2.translate(start2 * b_hat)

    # -----------------------------------
    # Final b length
    # -----------------------------------

    total_b = vacuum + thick1 + gap + thick2 + vacuum

    # scale b vector only
    new_bvec = b_hat * total_b

    final_cell = cell1.copy()
    final_cell[1] = new_bvec

    # -----------------------------------
    # Merge
    # -----------------------------------

    combined = s1 + s2

    combined.set_cell(final_cell)
    combined.set_pbc(True)

    # wrap atoms back into cell
    combined.wrap()

    # -----------------------------------
    # Save
    # -----------------------------------

    write(
        output,
        combined,
        format="vasp",
        direct=True,
        sort=True
    )

    print("\nSUCCESS")
    print(f"Gap     = {gap:.3f} Å")
    print(f"Vacuum  = {vacuum:.3f} Å")
    print(f"Final b = {total_b:.3f} Å")
    print(f"Saved   = {output}")


# ============================================================
# INPUT
# ============================================================

F1 = r"C:\Users\huyuy\OneDrive\Documents\Interface2\LIC_NMC\LIC.vasp"
#F1 = r"C:\Users\huyuy\Downloads\NMC_mono\89.vasp"
F2 = r"C:\Users\huyuy\OneDrive\Documents\Interface2\LIC_NMC\78.vasp"
#F2 = r"C:\Users\huyuy\Downloads\NMC_mono\Li3InCl6.vasp"

GAP = 2
VACUUM = 1

OUTPUT = "LIC_NMC_INTERFACE.vasp"

# ============================================================

merge_monoclinic_b(
    F1,
    F2,
    gap=GAP,
    vacuum=VACUUM,
    output=OUTPUT
)
