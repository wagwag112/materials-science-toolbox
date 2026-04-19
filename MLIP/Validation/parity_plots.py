#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from ase.io import read

from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator

import glob
import os
import random

from scipy.stats import gaussian_kde


# ======================
# PATHS
# ======================
OUTCAR_DIR = "/home/hoang0000/uma/NMC_new/train/test_folders"
outcar_files = glob.glob(os.path.join(OUTCAR_DIR, "**/vasprun.xml"), recursive=True)

#CHECKPOINT_PATH = "/home/hoang0000/inference_ckpt.pt"
CHECKPOINT_PATH = "/home/hoang0000/uma-s-1p2.pt"
#CHECKPOINT_PATH = "/home/hoang0000/uma_finetune/v2/202604-0514-5257-19ea/checkpoints/final/inference_ckpt.pt"

OUTPUT_FORCE = "force_parity.png"
OUTPUT_ENERGY = "energy_parity.png"

N_FRAMES_PER_FILE = 70  # number of random frames to sample per file


# ======================
# DEVICE
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)
torch.set_grad_enabled(False)


# ======================
# LOAD ML MODEL
# ======================
print("Loading ML model...")

predictor = load_predict_unit(CHECKPOINT_PATH, device=DEVICE)
calc = FAIRChemCalculator(predictor, task_name="omat")
print("Model loaded")


# ======================
# FIND OUTCAR FILES
# ======================
print(f"Found {len(outcar_files)} OUTCAR files")


# ======================
# COLLECT DATA
# ======================
dft_forces = []
ml_forces = []

dft_energy = []
ml_energy = []

print("Running ML inference...")

for path in outcar_files:
    print(f"Reading: {path}")

    try:
        frames = read(path, format="vasp-xml", index=":")
    except Exception as e:
        print(f"Skip file {path}: {e}")
        continue

    print(f"Total frames: {len(frames)}")

    # ======================
    # RANDOM FRAME SAMPLING
    # ======================
    n_sample = min(N_FRAMES_PER_FILE, len(frames))
    frames = random.sample(frames, n_sample)
    print(f"Sampled frames: {len(frames)}")

    # ======================
    # FILTER VALID FRAMES
    # ======================
    valid_frames = []

    for i, atoms in enumerate(frames):
        try:
            _ = atoms.get_positions()
            _ = atoms.get_forces()
            _ = atoms.get_potential_energy()
            valid_frames.append(atoms)
        except Exception as e:
            print(f"Skip step {i}: {e}")

    frames = valid_frames
    print(f"Valid frames: {len(frames)}")

    if len(frames) == 0:
        continue

    # ======================
    # RUN INFERENCE
    # ======================
    for i, atoms in enumerate(frames):
        try:
            # ---- DFT ----
            f_dft = atoms.get_forces()
            e_dft = atoms.get_potential_energy() / len(atoms)*1000

            # ---- ML ----
            atoms.calc = calc
            f_ml = atoms.get_forces()
            e_ml = atoms.get_potential_energy() / len(atoms)*1000

            dft_forces.append(f_dft.reshape(-1))
            ml_forces.append(f_ml.reshape(-1))

            dft_energy.append(e_dft)
            ml_energy.append(e_ml)

        except Exception as e:
            print(f"Skip step {i}: {e}")


if len(dft_forces) == 0:
    raise RuntimeError("No data collected")


# ======================
# CONCAT DATA
# ======================
y_true = np.concatenate(dft_forces)
y_pred = np.concatenate(ml_forces)

e_true = np.array(dft_energy)
e_pred = np.array(ml_energy)


# ======================
# CENTER ENERGY (shift both by same DFT mean)
# ======================
offset = np.mean(e_true)
e_true = e_true - offset
e_pred = e_pred - offset


# ======================
# METRICS
# ======================
mae_f = np.mean(np.abs(y_true - y_pred))
rmse_f = np.sqrt(np.mean((y_true - y_pred) ** 2))

mae_e = np.mean(np.abs(e_true - e_pred))
rmse_e = np.sqrt(np.mean((e_true - e_pred) ** 2))

print("\n===== FORCE RESULTS =====")
print(f"MAE  = {mae_f:.3f} eV/$\\AA$")
print(f"RMSE = {rmse_f:.3f} eV/$\\AA$")
print(f"N    = {len(y_true)}")

print("\n===== ENERGY RESULTS =====")
print(f"MAE  = {mae_e:.1f} meV/atom")
print(f"RMSE = {rmse_e:.1f} meV/atom")
print(f"N    = {len(e_true)}")


# ======================
# FONT SIZE
# ======================
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})


# ======================
# FORCE PARITY (HEXBIN)
# ======================
plt.figure(figsize=(8, 8), dpi=450)

limit = max(np.max(np.abs(y_true)), np.max(np.abs(y_pred))) * 1.05

hb = plt.hexbin(
    y_true,
    y_pred,
    gridsize=120,
    bins="log",
    mincnt=1,
    cmap="viridis"
)

plt.colorbar(hb, label="log10(count)")
plt.plot([-limit, limit], [-limit, limit], "r--", linewidth=1)

plt.xlim(-limit, limit)
plt.ylim(-limit, limit)

plt.xlabel("DFT force (eV/$\\AA$)")
plt.ylabel("ML force (eV/$\\AA$)")
plt.title("Force Parity Plot", fontsize=24)

plt.text(
    0.05,
    0.95,
    f"MAE = {mae_f:.3f} eV/$\\AA$\nRMSE = {rmse_f:.3f} eV/$\\AA$\nPoints = {len(y_true)}",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="white", alpha=0.6),
    verticalalignment="top",
    fontsize=14,
)

plt.tight_layout()
plt.savefig(OUTPUT_FORCE)
plt.close()

print(f"Saved plot: {OUTPUT_FORCE}")


# ======================
# ENERGY PARITY (KDE SCATTER)
# ======================
plt.figure(figsize=(8, 8), dpi=450)

# zoom robust
low = np.percentile(e_true, 1)
high = np.percentile(e_true, 99)
limit_e = max(abs(low), abs(high)) * 1.1

# KDE density
xy = np.vstack([e_true, e_pred])
z = gaussian_kde(xy)(xy)

# sort for better visualization
idx = z.argsort()
x, y, z = e_true[idx], e_pred[idx], z[idx]

sc = plt.scatter(x, y, c=z, s=10, cmap="viridis")

plt.colorbar(sc, label="Density")

# diagonal
plt.plot([-limit_e, limit_e], [-limit_e, limit_e], "r--", linewidth=1)

plt.xlim(-limit_e, limit_e)
plt.ylim(-limit_e, limit_e)

plt.xlabel("DFT energy (meV/atom)")
plt.ylabel("ML energy (meV/atom)")
plt.title("Energy Parity Plot", fontsize=24)

plt.text(
    0.05,
    0.95,
    f"MAE = {mae_e:.1f} meV/atom\nRMSE = {rmse_e:.1f} meV/atom\nPoints = {len(e_true)}",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="white", alpha=0.6),
    verticalalignment="top",
    fontsize=14,
)

plt.tight_layout()
plt.savefig(OUTPUT_ENERGY)
plt.close()

print(f"Saved plot: {OUTPUT_ENERGY}")
