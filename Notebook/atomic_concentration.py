import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ase.io import read
from scipy.ndimage import gaussian_filter1d

# --- CONFIGURATION ---
REF_START = 0
REF_WINDOW = 5
AXIS = 2
SIGMA = 1.6
RESOLUTION = 0.4
THRESHOLD = 11
MAX_CONTRIB = 4.5
FRAME_TO_PS = 1.0
ATOM_COLORS = {
    'Li': (0, 0.5, 1),
    'P':  (1.0, 0.5, 1),
    'S':  (1.0, 0.98, 0.0),
    'Cl': (49/255, 252/255, 2/255),
    'In': (1, 0, 0),
    'Co': (0, 0, 175/255),
    'Mn': (168/255, 8/255, 158/255),
    'Ni': (183/255, 187/255, 189/255),
    'O':  (0.5, 0.5, 0),
}
DEFAULT_COLOR = 'gray'


def compute_smoothed_profile(atoms_list, symbols, axis=2, sigma=0.4, res=0.1):
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
            if len(coords) == 0:
                continue
            counts, _ = np.histogram(coords, bins=np.append(grid, lx))
            smoothed = gaussian_filter1d(counts.astype(float), sigma/res, mode='nearest')
            densities[s] += (smoothed / (res * area))
    for s in symbols:
        densities[s] /= len(atoms_list)
    return grid, densities


def get_percentage(dens_dict):
    elements = list(dens_dict.keys())
    dens_matrix = np.array([dens_dict[s] for s in elements])
    total = np.sum(dens_matrix, axis=0) + 1e-10
    return {s: dens_dict[s] / total * 100 for s in elements}


def check_interphase_condition_at_bin(curr_perc, ref_perc, elements, idx):
    """
    LOGIC THỐNG NHẤT MỚI:
    - 1 element: biến động tuyệt đối > 6%
    - >= 2 elements: ít nhất 2 elements > 4% VÀ tất cả các elements còn lại phải > 2%
    """
    # Lọc các nguyên tố thực sự hiện diện tại mốc ban đầu (ref) ở bin này (nồng độ > 0.5%)
    active_elements = [s for s in elements if ref_perc[s][idx] > 0.5]
    n_elements = len(active_elements)
    
    if n_elements == 0:
        return False

    # Tính độ biến động tuyệt đối cục bộ của từng nguyên tố
    deviations = {s: abs(curr_perc[s][idx] - ref_perc[s][idx]) for s in active_elements}

    # --- NHÁNH 1 ELEMENT ---
    if n_elements == 1:
        main_element = active_elements[0]
        if deviations[main_element] > 6.0:
            return True
            
    # --- NHÁNH TỪ 2 ELEMENTS TRỞ LÊN ---
    else:
        # Sắp xếp danh sách các biến động giảm dần để kiểm tra thứ hạng điều kiện
        sorted_devs = sorted(deviations.values(), reverse=True)
        
        top_1 = sorted_devs[0]
        top_2 = sorted_devs[1]
        remaining_devs = sorted_devs[2:]
        
        cond_1 = (top_1 > 5.0 and top_2 > 5.0)            # Ít nhất 2 elements > 4%
        cond_2 = all(dev > 1.0 for dev in remaining_devs)   # Tất cả các cấu tử còn lại > 2%
        
        if cond_1 and cond_2:
            return True
            
    return False


def identify_interface_boundaries(grid, curr_perc, ref_perc, elements):
    """
    Quét từ hai vách sâu của hộp mô phỏng tiến vào giữa để tìm đầu mút giao diện
    """
    start_z, end_z = None, None
    box_mid_idx = len(grid) // 2

    # 1. Quét tìm Biên Trái (start_z): Từ thành trái (Z = 0) tiến dần vào giữa
    for i in range(box_mid_idx):
        if check_interphase_condition_at_bin(curr_perc, ref_perc, elements, i):
            start_z = grid[i]
            break

    # 2. Quét tìm Biên Phải (end_z): Từ thành phải (Z = max) quét ngược vào giữa
    for i in range(len(grid) - 1, box_mid_idx - 1, -1):
        if check_interphase_condition_at_bin(curr_perc, ref_perc, elements, i):
            end_z = grid[i]
            break

    return start_z, end_z


def run_analysis(traj_path, s_idx, e_idx, ref_elements=None):
    if ref_elements is None:
        ref_elements = []
        
    temp_match = re.search(r'(\d+)K', os.path.basename(traj_path))
    temp_key = temp_match.group(0) if temp_match else "UnknownK"
    time_ps = e_idx * FRAME_TO_PS
    time_label = f"{time_ps:.0f}ps"
    
    print(f"  [LOG] Reading frames {s_idx} to {e_idx} from {traj_path}...")
    traj = read(traj_path, index=":")
    all_elements = sorted(set(traj[0].get_chemical_symbols()))
    
    # Tính toán hồ sơ nồng độ trung bình của phân đoạn thời gian hiện tại
    grid, curr_dens = compute_smoothed_profile(traj[s_idx:e_idx+1], all_elements, axis=AXIS, sigma=SIGMA, res=RESOLUTION)
    curr_perc = get_percentage(curr_dens)
    
    # Tính toán hồ sơ nồng độ của trạng thái ban đầu ổn định
    ref_end = min(REF_START + REF_WINDOW, len(traj))
    print(f"  [LOG] Computing reference profile from frames {REF_START} to {ref_end-1}...")
    _, ref_dens = compute_smoothed_profile(traj[REF_START:ref_end], all_elements, axis=AXIS, sigma=SIGMA, res=RESOLUTION)
    ref_perc = get_percentage(ref_dens)
    ref_perc = get_percentage(ref_dens)
    
    # Xác định vị trí hai đầu mút dựa trên logic thích ứng
    start_z, end_z = identify_interface_boundaries(grid, curr_perc, ref_perc, all_elements)

    # --- KHỐI TIẾN HÀNH VẼ ĐỒ THỊ ---
    fig = plt.figure(figsize=(10, 6))
    for s in all_elements:
        color = ATOM_COLORS.get(s, DEFAULT_COLOR)
        plt.plot(grid, curr_perc[s], lw=2.0, color=color, label=f"{s} ({time_label})")
        if ref_perc and s in ref_elements:
            plt.plot(grid, ref_perc[s], color=color, ls='--', lw=1.5, alpha=0.7, label=f"{s} ({REF_START*FRAME_TO_PS:.0f}ps)")

    # Hiển thị dải kính mờ và vạch ranh giới interphase tự động
    if start_z is not None and end_z is not None:
        plt.axvline(x=start_z, color='red', linestyle=':', lw=2, alpha=0.8)
        plt.axvline(x=end_z, color='red', linestyle=':', lw=2, alpha=0.8)
        plt.axvspan(start_z, end_z, color='gray', alpha=0.1)
        mid_z = (start_z + end_z) / 2
        plt.text(mid_z, 67, rf"$\delta \approx {abs(end_z-start_z):.1f}$ $\mathrm{{\AA}}$",
                 color='red', fontweight='bold', ha='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.5'))
        print(f"  [LOG] Success! Interface identified: {start_z:.2f} to {end_z:.2f} A")
    else:
        plt.text(grid.mean(), 67, r"No valid interphase detected ($\delta=0$)",
                 color='red', fontweight='bold', ha='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.5'))
        print("  [LOG] No valid interface detected.")

    plt.xlabel(r"Distance along Z-axis ($\mathrm{\AA}$)", fontsize=12)
    plt.ylabel("Atomic Concentration (%)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(grid[0], grid[-1])
    plt.ylim(-5, 105)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.25), fontsize=10, frameon=True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    plt.tight_layout()
    
    plot_name = f"analysis_{temp_key}_{time_label}.png"
    plt.savefig(plot_name, dpi=450)
    plt.show()
    plt.close(fig)
    print(f"  [LOG] Analysis plot saved as {plot_name}")


if __name__ == "__main__":
    run_analysis(
        r"C:\Users\huyuy\Downloads\temp\Li3N_LIC\md_900K.extxyz",
        #r"C:\Users\huyuy\Downloads\LPSC_LIC\md_900K.extxyz",
        290, 300,
        ref_elements=["Li", "In", "Cl", "Ni", "Mn", "Co", "P", "S"]
    )
