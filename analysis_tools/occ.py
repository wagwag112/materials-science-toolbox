"""
Select atoms for deletion using Farthest Point Sampling (FPS) under
periodic boundary conditions (PBC) for a general triclinic cell.

The script reads Cartesian coordinates, constructs the lattice from
(a, b, c, alpha, beta, gamma), and iteratively selects atoms that are
maximally separated using the Minimum Image Convention (MIC).

Usage:
    python occ.py coords.txt N a b c alpha beta gamma
"""
import numpy as np
import argparse
import sys

def get_lattice_matrix(a, b, c, alpha, beta, gamma):
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    v1 = np.array([a, 0, 0])
    v2 = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])

    c1 = c * np.cos(beta_rad)
    c2 = c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
    c3 = np.sqrt(c**2 - c1**2 - c2**2)

    v3 = np.array([c1, c2, c3])

    return np.array([v1, v2, v3])


def get_pbc_distances(target_point, all_points, inv_lattice, lattice):
    frac_target = np.dot(target_point, inv_lattice)
    frac_all = np.dot(all_points, inv_lattice)

    delta_frac = frac_all - frac_target
    delta_frac -= np.round(delta_frac)

    delta_cart = np.dot(delta_frac, lattice)

    return np.sqrt(np.sum(delta_cart**2, axis=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("num_delete", type=int)
    parser.add_argument("a", type=float)
    parser.add_argument("b", type=float)
    parser.add_argument("c", type=float)
    parser.add_argument("alpha", type=float)
    parser.add_argument("beta", type=float)
    parser.add_argument("gamma", type=float)

    args = parser.parse_args()

    coords = np.loadtxt(args.file)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    lattice = get_lattice_matrix(args.a, args.b, args.c, args.alpha, args.beta, args.gamma)
    inv_lattice = np.linalg.inv(lattice)

    # =========================
    # FPS (UNCHANGED)
    # =========================
    selected_indices = []
    current_idx = 0
    selected_indices.append(current_idx)

    min_distances = get_pbc_distances(coords[current_idx], coords, inv_lattice, lattice)

    for _ in range(args.num_delete - 1):
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)
        new_dists = get_pbc_distances(coords[next_idx], coords, inv_lattice, lattice)
        min_distances = np.minimum(min_distances, new_dists)

    # =========================
    # OUTPUT FIX (IMPORTANT PART)
    # =========================

    # convert to (ID, index) pairs
    indexed = [(idx + 1, idx) for idx in selected_indices]

    # sort by ID (index + 1)
    indexed_sorted = sorted(indexed, key=lambda x: x[0])

    print("\n" + "=" * 85)
    print(f"{'ID':<6} | {'Cartesian (X Y Z)':<30} | {'Fractional (a b c)':<30}")
    print("-" * 85)

    for display_id, idx in indexed_sorted:
        cart = coords[idx]
        frac = np.dot(cart, inv_lattice)

        print(f"{display_id:<6} | "
              f"{cart[0]:8.5f} {cart[1]:8.5f} {cart[2]:8.5f} | "
              f"{frac[0]:8.5f} {frac[1]:8.5f} {frac[2]:8.5f}")

    print("=" * 85)
    print(f"Total atoms selected for deletion: {len(selected_indices)}\n")


if __name__ == "__main__":
    main()
