import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ase.io import read
from scipy.ndimage import gaussian_filter1d
import os

# --- CONFIG CỐ ĐỊNH ---
AXIS = 2          # Trục Z làm hướng phân tích chính
SIGMA = 1.6       # Độ mịn mượt màng Gauss để triệt tiêu dao động nhiệt cục bộ
RES = 0.4         # Độ phân giải lát cắt (bin width) tính bằng Angstrom
FRAMES_PER_PS = 1
STEP_FRAME = 20   # Bước nhảy khung hình khi quét động học theo thời gian


def get_profile(atoms, symbols, res=RES, sigma=SIGMA):
    """
    Trích xuất giản đồ nồng độ phần trăm mượt mà của tất cả cấu tử dọc theo trục AXIS
    """
    lx = np.linalg.norm(atoms.get_cell()[AXIS])
    grid = np.arange(0, lx, res)

    pos = atoms.get_positions()
    syms = np.array(atoms.get_chemical_symbols())

    densities = {}
    for s in symbols:
        coords = pos[syms == s][:, AXIS]
        # Xử lý trường hợp nguyên tố không xuất hiện trong frame hiện tại
        if len(coords) == 0:
            densities[s] = np.zeros_like(grid)
            continue
        counts, _ = np.histogram(coords, bins=np.append(grid, lx))
        densities[s] = gaussian_filter1d(counts.astype(float), sigma / res, mode='wrap')

    total = np.sum(list(densities.values()), axis=0) + 1e-10
    return grid, {s: densities[s] / total * 100 for s in symbols}


def detect_initial_interface(grid, ref_perc, reference_element):
    """
    Xác định mặt tiếp xúc hình học ban đầu (Frame 0) dựa trên 50% max nồng độ 
    của nguyên tố đặc trưng đại diện để làm mốc phân tách hướng quét.
    """
    profile = ref_perc[reference_element]
    profile_smooth = gaussian_filter1d(profile, sigma=2)
    threshold = 0.5 * np.max(profile_smooth)
    
    idx = np.argmin(np.abs(profile_smooth - threshold))
    return grid[idx]


def detect_orientation(grid, ref_perc, reference_element):
    """
    Xác định cấu trúc Slab nghiêng về nửa trái hay nửa phải của hộp mô phỏng
    """
    profile = ref_perc[reference_element]
    centroid = np.sum(grid * profile) / (np.sum(profile) + 1e-12)
    box_mid = 0.5 * np.max(grid)
    return "left" if centroid < box_mid else "right"


def check_interphase_condition(curr_perc, ref_perc, elements, idx):
    """
    - 1 element: biến động tuyệt đối > 6%
    - >= 2 elements: có ít nhất 2 elements thay đổi > 4% VÀ tất cả các elements còn lại phải thay đổi > 2%
    """
    # Lọc các nguyên tố thực sự hiện diện tại mốc ban đầu (Frame 0) ở bin này (nồng độ > 0.5%)
    active_elements = [s for s in elements if ref_perc[s][idx] > 0.5]
    n_elements = len(active_elements)
    
    if n_elements == 0:
        return False

    # Tính toán độ biến động tuyệt đối của từng cấu tử tại bin này so với ban đầu
    deviations = {s: abs(curr_perc[s][idx] - ref_perc[s][idx]) for s in active_elements}

    # --- NHÁNH THỨ NHẤT: CHỈ CÓ 1 ELEMENT ---
    if n_elements == 1:
        main_element = active_elements[0]
        # Điều kiện: Thay đổi cần lớn hơn 6%
        if deviations[main_element] > 5.0:
            return True

    # --- NHÁNH THỨ HAI: CÓ TỪ 2 ELEMENTS TRỞ LÊN ---
    else:
        # Sắp xếp các giá trị biến động theo thứ tự giảm dần
        sorted_devs = sorted(deviations.values(), reverse=True)
        
        # Lấy ra 2 giá trị biến động lớn nhất
        top_1 = sorted_devs[0]
        top_2 = sorted_devs[1]
        
        # Lấy các giá trị biến động còn lại (nếu có)
        remaining_devs = sorted_devs[2:]
        
        # Đk1: Có ít nhất 2 elements thay đổi trên 4%
        cond_1 = (top_1 > 5.0 and top_2 > 5.0)
        
        # Đk2: Các elements còn lại phải thay đổi trên 2%
        # Nếu chỉ có đúng 2 elements thì remaining_devs rỗng -> cond_2 mặc định là True
        cond_2 = all(dev > 2.0 for dev in remaining_devs)
        
        if cond_1 and cond_2:
            return True
            
    return False


def run_combined_analysis(traj_path):
    print(f"[LOG] Reading trajectory from file: {traj_path}")
    traj = read(traj_path, index=":")

    # --- NHẬN DIỆN CẤU TỬ TỰ ĐỘNG (FRAME 0) ---
    all_elements = sorted(set(traj[0].get_chemical_symbols()))
    print(f"[LOG] Automatically detected elements in system: {all_elements}")
    
    grid, ref_perc = get_profile(traj[0], all_elements)

    # Chọn nguyên tố đầu tiên xuất hiện để làm mốc định vị hướng cho hệ thống hình học nhị phân
    ref_el = all_elements[0]
    z_interface_0 = detect_initial_interface(grid, ref_perc, ref_el)
    orientation = detect_orientation(grid, ref_perc, ref_el)
    
    print(f"[LOG] Geometrical contact plane at Z = {z_interface_0:.2f} Å")
    print(f"[LOG] Identified slab orientation: {orientation}")

    # --- THIẾT LẬP HƯỚNG QUÉT BIÊN AN TOÀN ---
    if orientation == "left":
        left_scan = range(len(grid))
        right_scan = range(len(grid) - 1, -1, -1)
        left_stop = lambda z: z >= z_interface_0
        right_stop = lambda z: z <= z_interface_0
    else:
        left_scan = range(len(grid) - 1, -1, -1)
        right_scan = range(len(grid))
        left_stop = lambda z: z <= z_interface_0
        right_stop = lambda z: z >= z_interface_0

    times = []
    z_min_list = []
    z_max_list = []

    # --- KHỐI QUÉT ĐỘNG HỌC THEO THỜI GIAN ---
    for i in range(0, len(traj), STEP_FRAME):
        atoms = traj[i]
        time_ps = i / FRAMES_PER_PS

        grid, curr_perc = get_profile(atoms, all_elements)

        # Quét tìm Biên Trái (Z_min) xuất phát từ thành sâu bên trái vào giữa
        z_min = z_interface_0
        for idx in left_scan:
            if left_stop(grid[idx]):
                break
            if check_interphase_condition(curr_perc, ref_perc, all_elements, idx):
                z_min = grid[idx]
                break

        # Quét tìm Biên Phải (Z_max) xuất phát từ thành sâu bên phải ngược vào giữa
        z_max = z_interface_0
        for idx in right_scan:
            if right_stop(grid[idx]):
                break
            if check_interphase_condition(curr_perc, ref_perc, all_elements, idx):
                z_max = grid[idx]
                break

        times.append(time_ps)
        z_min_list.append(z_min)
        z_max_list.append(z_max)
        
    # --- POST PROCESSING ---
    z_max_arr = np.maximum(np.array(z_max_list), z_max_list[0])
    for i in range(1, len(z_max_arr)):
        delta = z_max_arr[i] - z_max_arr[i-1]
        if delta > 4:
            z_max_arr[i] = z_max_arr[i-1] + 2
        elif delta < -4:
            z_max_arr[i] = z_max_arr[i-1] - 2
    z_max_list = z_max_arr.tolist()

    z_min_arr = np.minimum(np.array(z_min_list), z_min_list[0])
    for i in range(1, len(z_min_arr)):
        delta = z_min_arr[i] - z_min_arr[i-1]
        if delta < -4:
            z_min_arr[i] = z_min_arr[i-1] - 2
        elif delta > 4:
            z_min_arr[i] = z_min_arr[i-1] + 2
    z_min_list = z_min_arr.tolist()
    # --- XUẤT DỮ LIỆU ĐỊNH LƯỢNG (.CSV) ---
    times = np.array(times)
    z_min = np.array(z_min_list)
    z_max = np.array(z_max_list)
    thickness = np.abs(z_max - z_min)

    out_base = os.path.basename(traj_path).split('.')[0]
    df = pd.DataFrame({
        "Time_ps": times,
        "Z_min_frontier": z_min,
        "Z_max_frontier": z_max,
        "Interphase_thickness": thickness
    })

    csv_name = f"generalized_interphase_{out_base}.csv"
    df.to_csv(csv_name, index=False)
    print(f"[LOG] Output saved successfully to: {csv_name}")

    # --- ĐỒ THỊ TIẾN HÓA THỜI GIAN (PLOT) ---
    plt.figure(figsize=(9, 5))
    plt.plot(times, z_min, 'o-', color='#1f77b4', label='Left Frontier (Z_min)')
    plt.plot(times, z_max, 'o-', color='#ff7f0e', label='Right Frontier (Z_max)')
    plt.axhline(z_interface_0, ls='--', color='black', alpha=0.7, label='Initial Contact Plane')

    plt.xlabel("Time (ps)", fontsize=12)
    plt.ylabel("Z Coordinate (Å)", fontsize=12)
    plt.title("Interphase Evolution", fontsize=13, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    png_name = f"generalized_interphase_{out_base}.png"
    plt.savefig(png_name, dpi=300, bbox_inches='tight')
    print(f"[LOG] Analysis figure saved to: {png_name}")
    plt.show()


if __name__ == "__main__":
    # Anh chỉ cần thay đường dẫn file cấu trúc hệ mới bất kỳ (.extxyz) vào đây để chạy
    run_combined_analysis(r"C:\Users\huyuy\Downloads\temp\Li_LPSC\md_400K.extxyz")
