import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry import get_distances
import os

# ── CONFIG ───────────────────────────────────────────────────────────
#BASE_PATH = r"C:\Users\huyuy\Downloads\LPSC_LIC"
BASE_PATH = r"C:\Users\huyuy\Downloads\temp\Li3N_LIC"
#Z_MIN, Z_MAX = 20.0, 40.0
Z_BUFFER = 3.0
#Z_NEIGHBOR_MIN = Z_MIN - Z_BUFFER
#Z_NEIGHBOR_MAX = Z_MAX + Z_BUFFER
R_MAX = 4.0
N_BINS = 150
FPS = 1
TIME_WINDOWS_PS = [(0, 10), (740, 750), (1490, 1500), (2240, 2250), (2990, 3000)]
FONT_SIZE = 18
TICK_SIZE = 14


# ── HELPERS ──────────────────────────────────────────────────────────
def get_rdf_corrected(traj, pair, z_min, z_max, z_neighbor_min, z_neighbor_max, r_max, n_bins):
    dr = r_max / n_bins
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = r_edges[:-1] + dr / 2
    pair_counts = np.zeros(n_bins)
    n_central_total = 0
    n_neighbor_list = []

    for atoms in traj:
        pos = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        symbols = np.array(atoms.get_chemical_symbols())
        central_mask = (pos[:, 2] >= z_min) & (pos[:, 2] <= z_max) & (symbols == pair[0])
        neighbor_mask = (pos[:, 2] >= z_neighbor_min) & (pos[:, 2] <= z_neighbor_max) & (symbols == pair[1])
        central_idx = np.where(central_mask)[0]
        neighbor_idx = np.where(neighbor_mask)[0]
        if len(central_idx) == 0 or len(neighbor_idx) == 0:
            continue
        n_central_total += len(central_idx)
        n_neighbor_list.append(len(neighbor_idx))
        pos_central = pos[central_idx]
        pos_neighbor = pos[neighbor_idx]
        _, dists = get_distances(pos_central, pos_neighbor, cell=cell, pbc=pbc)
        if pair[0] == pair[1]:
            for i, ci in enumerate(central_idx):
                for j, nj in enumerate(neighbor_idx):
                    if ci == nj:
                        dists[i, j] = np.inf
        counts, _ = np.histogram(dists.ravel(), bins=r_edges)
        pair_counts += counts

    if n_central_total == 0 or len(n_neighbor_list) == 0:
        return r_centers, np.zeros(n_bins), np.zeros(n_bins)

    cell = traj[-1].get_cell()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    slab_vol = area * (z_neighbor_max - z_neighbor_min)
    rho = np.mean(n_neighbor_list) / slab_vol
    shell_vols = (4.0 / 3.0) * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)
    rdf = pair_counts / (n_central_total * rho * shell_vols)
    cn_cum = np.cumsum(pair_counts) / n_central_total
    return r_centers, rdf, cn_cum


def mark_first_peak(ax, r, rdf_val, cn_val, pair):
    mask = r > 1.5
    if np.max(rdf_val[mask]) <= 0.1:
        return
    peak_idx = np.argmax(rdf_val[mask]) + np.sum(~mask)
    peak_r = r[peak_idx]
    peak_g = rdf_val[peak_idx]
    ax.axvline(x=peak_r, color='gray', linestyle='--', alpha=0.6, lw=1)
    
    # Kiểm tra nếu chữ bị vượt quá giới hạn khung (y-limit) thì đẩy sang bên cạnh
    ylim_max = ax.get_ylim()[1]
    if peak_g * 1.05 > ylim_max:
        ax.text(peak_r + 0.05, peak_g, f'{peak_r:.2f}', ha='left', va='center',
                fontsize=TICK_SIZE - 2, fontweight='bold')
    else:
        ax.text(peak_r, peak_g * 1.05, f'{peak_r:.2f}', ha='center', va='bottom',
                fontsize=TICK_SIZE - 2, fontweight='bold')
                
    search_range = (r > peak_r) & (r < peak_r + 1.0)
    if np.any(search_range):
        min_after_peak_idx = np.argmin(rdf_val[search_range]) + np.where(search_range)[0][0]
        r_cutoff = r[min_after_peak_idx]
        cn_at_cutoff = cn_val[min_after_peak_idx]
        ax.text(0.95, 0.95, f'CN: {cn_at_cutoff:.2f}\n(at {r_cutoff:.2f}Å)',
                transform=ax.transAxes, ha='right', va='top', color='darkorange',
                fontsize=TICK_SIZE - 2, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


def plot_grid(fig, axes, traj_fn, pairs_config, windows):
    for col, key in enumerate(windows):
        t0, t1 = key
        traj = read(traj_fn, index=f"{t0}:{t1}")
        title = f"{t0}-{t1} ps"
        color = 'crimson'

        for row, (pair, z_min_eff, z_max_eff) in enumerate(pairs_config):
            r, rdf_val, cn_val = get_rdf_corrected(
                traj, pair, z_min_eff, z_max_eff,
                z_min_eff - Z_BUFFER, z_max_eff + Z_BUFFER, R_MAX, N_BINS)
            ax = axes[row, col]
            ax.plot(r, rdf_val, color=color, lw=2)
            ax_cn = ax.twinx()
            ax_cn.set_ylim(0, 10)
            ax_cn.set_yticklabels([])
            mark_first_peak(ax, r, rdf_val, cn_val, pair)
            ax.set_xlim(0, R_MAX)
            ax.tick_params(axis='both', labelsize=TICK_SIZE)
            if row == 0:
                ax.set_title(title, fontsize=FONT_SIZE + 4, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f"{pair[0]}-{pair[1]}\ng(r)", fontsize=FONT_SIZE, fontweight='bold')
            ax.grid(alpha=0.2)


# ── PHẦN QUẢN LÝ CÁC CẶP ĐẦU VÀO & TÙY CHỈNH VÙNG CHẤT ────────────────
# Anh có thể thay đổi cặp chất, z_min, z_max linh hoạt tại đây.
PAIRS_CONFIG = [
    (('In', 'N'),   25.0, 55.0),
    (('S', 'Li'),  25.0, 55.0),
    (('Cl', 'Li'), 25.0, 55.0),
    (('P', 'Li'), 25.0, 55.0)
]


# ── PHẦN THEO THỜI GIAN (900K) ───────────────────────────────────────
print("Processing Time Plot...")
# Tự động điều chỉnh kích thước hàng (subplots) theo số lượng cấu hình cặp chất ở trên
fig_time, axes_time = plt.subplots(len(PAIRS_CONFIG), len(TIME_WINDOWS_PS), 
                                   figsize=(24, 16), sharex=True, sharey='row', squeeze=False)

fpath_900 = os.path.join(BASE_PATH, "md_800K.extxyz")
plot_grid(fig_time, axes_time, fpath_900, PAIRS_CONFIG, TIME_WINDOWS_PS)

plt.tight_layout()
fig_time.savefig('RDF_CN_vs_Time_400K.png', dpi=450, bbox_inches='tight')
plt.show()
