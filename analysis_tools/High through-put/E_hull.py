import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm

from pymatgen.core import Structure, Composition, Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry, PDPlotter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

# ? FIXED IMPORTS
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter

import torch
from fairchem.core import pretrained_mlip, FAIRChemCalculator

# -- CONFIGURATION -------------------------------------------------------------
API_KEY      = "8cLREHJ2D9HWs4iL8yWdL08lNyIDXv2y"
CACHE_FILE   = "cache.json"
INPUT_FOLDER = "input"
E_HULL_CUTOFF = 0.01   

# -- MODEL LOADER --------------------------------------------------------------
def get_uma_calculator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading UMA on {device.upper()}...")
    try:
        calc = FAIRChemCalculator.from_model_checkpoint(
    		name_or_path="/home/hoang0000/inference_ckpt.pt",
    		task_name="omat",
    		device="cuda"
    )
        return calc
    except Exception as e:
        print(f"  Model load failed: {e}")
        return None

# -- STRUCTURE UTILITIES -------------------------------------------------------
def get_primitive(structure_pmg):
    try:
        sga = SpacegroupAnalyzer(structure_pmg, symprec=0.1)
        return sga.get_primitive_standard_structure()
    except Exception:
        return structure_pmg

# -- RELAXATION ----------------------------------------------------------------
def _track_lowest(atoms, state):
    e = atoms.get_potential_energy()
    if e < state["best_energy"]:
        state["best_energy"] = e
        state["best_atoms"]  = atoms.copy()


def relax_and_get_entry(structure_pmg, calculator, name=None,
                         stage1_steps=0, stage2_steps=100, fmax=0.02):

    prim_struct = get_primitive(structure_pmg)
    atoms = AseAtomsAdaptor.get_atoms(prim_struct)
    atoms.calc = calculator

    state = {"best_energy": float("inf"), "best_atoms": None}

    try:
        # ? FIXED Stage 1
        ucf  = FrechetCellFilter(atoms)
        dyn1 = LBFGS(ucf, logfile=os.devnull)
        dyn1.attach(_track_lowest, interval=1, atoms=atoms, state=state)
        dyn1.run(fmax=fmax, steps=stage1_steps)

        # Stage 2
        atoms.set_constraint(None)

        # ? FIXED Stage 2
        dyn2 = LBFGS(atoms, logfile=os.devnull)
        dyn2.attach(_track_lowest, interval=1, atoms=atoms, state=state)
        dyn2.run(fmax=fmax, steps=stage2_steps)

        if state["best_atoms"] is not None:
            best = state["best_atoms"]
        else:
            best = atoms

        best_energy = state["best_energy"]
        best_struct = AseAtomsAdaptor.get_structure(best)

        entry = PDEntry(best_struct.composition, best_energy, attribute=name)
        return entry

    except Exception as e:
        print(f"  Relaxation error [{name}]: {e}")
        return None

# -- CACHE I/O -----------------------------------------------------------------
def load_local_cache(filename):
    if not os.path.exists(filename):
        return []
    with open(filename, 'r') as f:
        data = json.load(f)
    entries = [
        PDEntry(Composition(d['composition']), d['energy'], attribute=d.get('name'))
        for d in data
    ]
    print(f"  Loaded {len(entries)} entries from cache.")
    return entries

def save_to_cache(entries, filename):
    data = [
        {
            'composition': e.composition.formula.replace(" ", ""),
            'energy': e.energy,
            'name': e.attribute
        }
        for e in entries
    ]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  Saved {len(entries)} entries to cache.")

# -- STRUCTURE PREPARATION -----------------------------------------------------
def _scan_input_files(folder_path):
    SUPPORTED_EXT = {".cif", ".vasp", ".xyz", ".json"}
    BARE_NAMES    = {"POSCAR", "CONTCAR"}

    found = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue
        _, ext = os.path.splitext(fname)
        if fname in BARE_NAMES or ext.lower() in SUPPORTED_EXT:
            found.append(fpath)
    return sorted(found)


def prepare_structures_to_calc(folder_path, api_key, max_hull_energy=0.2):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Input folder not found: '{folder_path}'")

    input_files = _scan_input_files(folder_path)
    if not input_files:
        raise FileNotFoundError(
            f"No supported structure files found in '{folder_path}'. "
            f"Accepted: .cif  .vasp  POSCAR  CONTCAR  .xyz  .json"
        )

    print(f"  Found {len(input_files)} structure file(s):")
    for p in input_files:
        print(f"    {os.path.basename(p)}")

    structures_to_calc = []
    mp_chemsys_cache   = {}
    added_ids          = set()

    for input_path in input_files:
        file_name = os.path.basename(input_path)
        try:
            struct = Structure.from_file(input_path)
            struct.remove_oxidation_states()

            if file_name not in added_ids:
                structures_to_calc.append((struct, file_name))
                added_ids.add(file_name)

            elements = sorted(set(str(e) for e in struct.composition.elements))
            if "Li" not in elements:
                elements.append("Li")
                elements = sorted(elements)
            sys_key = tuple(elements)

            if sys_key not in mp_chemsys_cache:
                print(f"  Fetching MP: {list(sys_key)}")
                try:
                    with MPRester(api_key) as mpr:
                        mp_entries = mpr.get_entries_in_chemsys(list(sys_key))
                    mp_chemsys_cache[sys_key] = mp_entries
                except Exception as e:
                    print(f"  MP fetch error {list(sys_key)}: {e}")
                    mp_chemsys_cache[sys_key] = []

            mp_entries = mp_chemsys_cache[sys_key]
            if mp_entries:
                pd_mp = PhaseDiagram(mp_entries)
                for e in mp_entries:
                    if e.entry_id in added_ids:
                        continue
                    try:
                        if pd_mp.get_e_above_hull(e) > max_hull_energy:
                            continue
                    except Exception:
                        continue
                    try:
                        e.structure.remove_oxidation_states()
                    except Exception:
                        pass
                    structures_to_calc.append((e.structure, e.entry_id))
                    added_ids.add(e.entry_id)

        except Exception as e:
            print(f"  Error reading {file_name}: {e}")

    print(f"  Total structures to relax: {len(structures_to_calc)}")
    return structures_to_calc

# -- NEUTRAL COMPOSITION FIX ---------------------------------------------------
def ensure_neutral_entries(entries):
    return [
        PDEntry(Composition(e.composition.formula), e.energy, attribute=e.attribute)
        for e in entries
    ]

# -- E_HULL ANALYSIS -----------------------------------------------------------
def analyse_ehull(target_name, target_entry, all_clean_entries):
    target_elements = set(target_entry.composition.elements)
    filtered = [
        e for e in all_clean_entries
        if set(e.composition.elements).issubset(target_elements)
    ]

    try:
        pd_mlip = PhaseDiagram(filtered)
    except Exception as e:
        print(f"  PhaseDiagram build failed [{target_name}]: {e}")
        return

    ehull = pd_mlip.get_e_above_hull(target_entry)
    formula = target_entry.composition.reduced_formula

    if ehull <= 0.025:
        status = "Stable"
    elif ehull <= 0.05:
        status = "Metastable"
    elif ehull <= 0.1:
        status = "Marginally unstable"
    else:
        status = "Unstable"

    print(f"  E_hull = {ehull:.4f} eV/atom  ?  {status}")

    try:
        plotter = PDPlotter(pd_mlip, show_unstable=0.1, backend="matplotlib")
        plotter.get_plot()
        stem  = target_name.replace('.cif', '')
        fname = f"PD_{formula}_{stem}.png"
        plt.title(f"Phase diagram: {formula}  ({target_name})")
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Phase diagram saved: {fname}")
    except Exception as e:
        print(f"  Phase diagram plot failed: {e}")

    stem      = target_name.replace('.cif', '')
    stab_file = f"stability_{stem}.txt"
    with open(stab_file, 'w') as f:
        f.write(f"File    : {target_name}\n")
        f.write(f"Formula : {formula}\n")
        f.write(f"E_hull  : {ehull:.4f} eV/atom\n")
        f.write(f"Status  : {status}\n")
    print(f"  Stability report saved: {stab_file}")

# -- MAIN ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  E_hull screening    MLIP unified reference frame")
    print("=" * 60)

    calculated_entries = load_local_cache(CACHE_FILE)
    calculated_names   = {e.attribute for e in calculated_entries}

    todo_list = prepare_structures_to_calc(INPUT_FOLDER, API_KEY, max_hull_energy=E_HULL_CUTOFF)
    real_todo = [(s, n) for s, n in todo_list if n not in calculated_names]

    print(f"\n  Cache : {len(calculated_names)} existing")
    print(f"  New   : {len(real_todo)} structures to relax")

    if real_todo:
        calc = get_uma_calculator()
        if calc is None:
            raise RuntimeError("Failed to load UMA model  aborting.")

        for struct, name in tqdm(real_todo, desc="Relaxing"):
            struct.remove_oxidation_states()
            entry = relax_and_get_entry(struct, calc, name=name)
            if entry:
                calculated_entries.append(entry)
                calculated_names.add(name)

                if len(calculated_entries) % 5 == 0:
                    save_to_cache(calculated_entries, CACHE_FILE)

        save_to_cache(calculated_entries, CACHE_FILE)
    else:
        print("  All structures in cache  skipping relaxation.")

    print("\n" + "=" * 60)
    print("  Analysis")
    print("=" * 60)

    clean_entries = ensure_neutral_entries(calculated_entries)
    input_files   = _scan_input_files(INPUT_FOLDER)

    for input_path in input_files:
        target_name  = os.path.basename(input_path)
        target_entry = next(
            (e for e in clean_entries if e.attribute == target_name), None
        )
        print(f"\n>>> {target_name}")
        if target_entry is None:
            print(f"  No relaxed entry found in cache  skipping.")
            continue
        analyse_ehull(target_name, target_entry, clean_entries)

    print("\n  Done.")
