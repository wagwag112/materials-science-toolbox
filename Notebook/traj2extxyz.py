import ase.io as io

# --- INPUT ---
input_file = r"C:\Users\huyuy\Downloads\LPSC_LIC\md_300K.traj"
stride = 10  # lấy mỗi 10 frame (anh chỉnh tùy ý)

# --- OUTPUT ---
output_file = input_file.replace('.traj', f'_sub_{stride}.extxyz')

# --- READING ---
print(f"Reading {input_file} with stride {stride}...")
try:
    frames = io.read(input_file, index=f'::{stride}')
    print(f"Loaded {len(frames)} frames.")
except Exception as e:
    print(f"ERROR reading file: {e}")
    raise

# --- WRITING ---
print(f"Writing to {output_file}...")
try:
    io.write(output_file, frames)
    print(f"Done! Created: {output_file}")
except Exception as e:
    print(f"ERROR writing file: {e}")
    raise
