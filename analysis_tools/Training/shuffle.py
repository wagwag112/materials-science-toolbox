import os
import glob
from ase.io import read, write
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import random
import sys

# --- CONFIGURATION ---
BASE_PATH = '/home/hoang0000/uma/NMC_new/train/data_vasp/'
OUTPUT_BASE = '/home/hoang0000/uma/NMC_new/train/final_extxyz_inputs/'
TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_BASE, 'train')
VAL_OUTPUT_DIR = os.path.join(OUTPUT_BASE, 'val')

os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_OUTPUT_DIR, exist_ok=True)

# Split ratio
TRAIN_RATIO = 0.90
VAL_RATIO = 0.10
RANDOM_SEED = 42

# =======================
# STRIDE SETTING (IMPORTANT)
# =======================
STRIDE = 25   

all_selected_structures = []

# --- DATA COLLECTION ---
xml_files = glob.glob(os.path.join(BASE_PATH, '**', 'vasprun.xml'), recursive=True)

for f in xml_files:
    try:
        frames = read(f, index=':')
        total_frames_in_file = len(frames)

        # =======================
        # STRIDE SAMPLING
        # =======================
        if total_frames_in_file == 1:
            # always keep single-frame files
            selected_frames = frames
        else:
            selected_frames = frames[::STRIDE]

            # safety: ensure at least 1 frame per file
            if len(selected_frames) == 0:
                selected_frames = [frames[0]]

        dir_name = os.path.basename(os.path.dirname(f))

        for frame in selected_frames:
            frame.info['source_dir'] = dir_name
            all_selected_structures.append(frame)

    except Exception as e:
        print(f"Skip corrupted file {f}: {e}")

# --- SHUFFLING AND SPLITTING ---
random.seed(RANDOM_SEED)
random.shuffle(all_selected_structures)

final_count = len(all_selected_structures)

if final_count == 0:
    print("No structures found. Check your BASE_PATH.")
    sys.exit()

# Split logic for Train and Validation only
train_structures, val_structures = train_test_split(
    all_selected_structures,
    test_size=VAL_RATIO,
    random_state=RANDOM_SEED
)

# --- SAVE OUTPUTS ---
def save_structures_to_extxyz(structures, output_dir, filename):
    full_path = os.path.join(output_dir, filename)
    write(full_path, structures, format='extxyz')
    print(f"Saved {len(structures)} structures to {full_path}")

save_structures_to_extxyz(train_structures, TRAIN_OUTPUT_DIR, 'train_data.extxyz')
save_structures_to_extxyz(val_structures, VAL_OUTPUT_DIR, 'val_data.extxyz')

print(f"Total processed: {final_count}")
print(f"Train size: {len(train_structures)}")
print(f"Val size: {len(val_structures)}")
