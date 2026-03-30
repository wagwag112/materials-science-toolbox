import os
import json
import warnings
import numpy as np
import pandas as pd
import re
import matplotlib
import glob  # Added for folder scanning

# Set non-interactive backend for HPC to avoid display errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Replace tqdm.notebook with standard tqdm for terminal/HPC
from tqdm import tqdm

# Pymatgen & ASE Imports
from pymatgen.core import Structure, Composition, Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry, PDPlotter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

# ASE & Optimization Imports
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS

# Fairchem / MLIP Imports
import torch
from fairchem.core import pretrained_mlip, FAIRChemCalculator

# --- CONFIGURATION ---
# It is recommended to store your API KEY as an environment variable for security
API_KEY = "6jFoVpNm3MMLb20ajVGv9sY1JNMJ67oc"
CACHE_FILE = "mlip_relaxed_entries.json"
INPUT_FOLDER = "cifs_relaxed"  # Folder containing your CIF files
E_HULL_CUTOFF = 0.025  # Slightly loosen threshold to capture more phases

# --- PROCESSING FUNCTIONS ---

def get_uma_calculator():
    """Load UMA model from fairchem."""
    try:
        # Automatically detect GPU on HPC
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f" Loading UMA model on device: {device.upper()}...")
        
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
        calc = FAIRChemCalculator(predictor, task_name="omat")
        print(" Model is ready!")
        return calc
    except Exception as e:
        print(f" Error loading model: {e}")
        return None

def get_primitive(structure_pmg):
    try:
        sga = SpacegroupAnalyzer(structure_pmg)
        return sga.get_primitive_standard_structure()
    except:
        return structure_pmg

def relax_and_get_entry(structure_pmg, calculator, name=None):
    # 1. Convert to primitive cell
    prim_struct = get_primitive(structure_pmg)
    
    # 2. Convert to ASE Atoms
    atoms = AseAtomsAdaptor.get_atoms(prim_struct)
    atoms.calc = calculator
    
    best_energy = float('inf')
    
    def log_step(a=atoms):
        nonlocal best_energy
        e = a.get_potential_energy()
        if e < best_energy:
            best_energy = e

    try:
        ucf = ExpCellFilter(atoms)
        # Write log to file instead of None if debugging is needed
        dyn = BFGS(ucf, logfile='relaxation.log')
        dyn.attach(log_step)
        dyn.run(fmax=0.001, steps=500)  # Increase steps to ensure convergence
        
        final_energy = atoms.get_potential_energy()
        energy_to_use = min(final_energy, best_energy)
        
        final_struct_pmg = AseAtomsAdaptor.get_structure(atoms)
        entry = PDEntry(final_struct_pmg.composition, energy_to_use, attribute=name)
        return entry

    except Exception as e:
        print(f"?? Relaxation error ({name}): {e}")
        return None

def load_local_cache(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        entries = []
        for d in data:
            comp = Composition(d['composition'])
            entries.append(PDEntry(comp, d['energy'], attribute=d.get('name')))
        print(f" Loaded {len(entries)} entries from cache '{filename}'.")
        return entries
    return []

def save_to_cache(entries, filename):
    data = []
    for e in entries:
        data.append({
            'composition': e.composition.formula.replace(" ", ""),
            'energy': e.energy,
            'name': e.attribute
        })
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f" Saved {len(entries)} entries to '{filename}'.")

def prepare_structures_to_calc(folder_path, api_key, max_hull_energy=0.2):
    structures_to_calc = []
    
    # Check folder existence
    if not os.path.exists(folder_path):
         raise FileNotFoundError(
            f"\n CRITICAL ERROR \n"
            f"Input folder not found: '{folder_path}'\n"
            f"Please create this folder and upload your CIF files into it."
        )

    # Get all CIF files
    cif_files = glob.glob(os.path.join(folder_path, "*.cif"))
    
    if not cif_files:
        raise FileNotFoundError(f"No .cif files found in folder '{folder_path}'")

    print(f" Found {len(cif_files)} CIF files in '{folder_path}'. Processing...")

    # Caches to avoid redundant work
    mp_chemsys_cache = {} # Cache MP queries by chemical system tuple
    added_ids = set()     # Track IDs to avoid duplicates in list

    for cif_path in cif_files:
        cif_name = os.path.basename(cif_path)
        
        # 1. Read CIF
        try:
            custom_struct = Structure.from_file(cif_path)
            custom_struct.remove_oxidation_states()
            
            # Add to list (Target structure always gets calculated)
            if cif_name not in added_ids:
                structures_to_calc.append((custom_struct, cif_name))
                added_ids.add(cif_name)
            
            # Extract chemical system
            elements = [str(e) for e in custom_struct.composition.elements]
            if "Li" not in elements: elements.append("Li")
            elements_tuple = tuple(sorted(elements)) # Create a hashable key
            
            print(f" [{cif_name}] System: {elements}")

            # 2. Fetch from MP (if not already fetched for this system)
            if elements_tuple not in mp_chemsys_cache:
                print(f"   Fetching MP data for system {elements}...")
                try:
                    with MPRester(api_key) as mpr:
                        mp_entries = mpr.get_entries_in_chemsys(list(elements_tuple))
                    mp_chemsys_cache[elements_tuple] = mp_entries # Cache it
                except Exception as e:
                    print(f"   MP Error for {elements}: {e}")
                    mp_chemsys_cache[elements_tuple] = []
            
            # 3. Process MP entries for this system
            mp_entries = mp_chemsys_cache[elements_tuple]
            if mp_entries:
                pd_mp = PhaseDiagram(mp_entries)
                
                for e in mp_entries:
                    # Skip if already added
                    if e.entry_id in added_ids:
                        continue

                    # Filter by Hull Energy
                    try:
                        if pd_mp.get_e_above_hull(e) > max_hull_energy:
                            continue
                    except:
                        continue # Skip if PD calculation fails for entry

                    # Remove oxidation states for consistency
                    try:
                        e.structure.remove_oxidation_states()
                    except: pass

                    # Add unique stable MP structure to calculation list
                    structures_to_calc.append((e.structure, e.entry_id))
                    added_ids.add(e.entry_id)

        except Exception as e:
            print(f"   Error processing {cif_name}: {e}")
            continue

    print(f" Total structures prepared for relaxation (Targets + MP References): {len(structures_to_calc)}")
    return structures_to_calc

def ensure_neutral_entries(entries):
    cleaned = []
    for e in entries:
        neutral_comp = Composition(e.composition.formula)
        new_entry = PDEntry(neutral_comp, e.energy, attribute=e.attribute)
        cleaned.append(new_entry)
    return cleaned

def plot_professional_voltage(profile, target_formula, ref_energy):
    if not profile:
        return

    # Plot styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(figsize=(10, 6))

    voltages = []
    evolutions = []
    labels = []

    for d in profile:
        voltage = -(d["chempot"] - ref_energy)
        evolution = d["evolution"]
        voltages.append(voltage)
        evolutions.append(evolution)
        products = [p.reduced_formula for p in d["reaction"].products if p.reduced_formula != "Li"]
        label = " + ".join(products)
        label = re.sub(r"(\d+)", r"$_{\1}$", label)
        labels.append(label)

    for i in range(len(voltages)):
        v = voltages[i]
        evo = evolutions[i]
        x_start = voltages[i-1] if i > 0 else 0
        x_end = v
        if i == len(voltages) - 1:
            x_end = x_start + 1.5
        color = plt.cm.viridis(0.2 + 0.6 * (i / len(voltages)))

        ax.plot([x_start, x_end], [evo, evo], color='black', linewidth=2.5, zorder=2)
        ax.fill_between([x_start, x_end], 0, evo, color=color, alpha=0.2)

        if i > 0:
            prev_evo = evolutions[i-1]
            ax.plot([x_start, x_start], [prev_evo, evo], color='gray', linestyle='--', linewidth=1.5)

        mid_x = (x_start + x_end) / 2
        ax.text(
            mid_x,
            evo + 0.05 * max(evolutions),
            labels[i],
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='darkblue',
            rotation=45
        )

    ax.set_xlabel("Voltage vs. Li/Li$^{+}$ (V)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Li Atoms Consumed", fontsize=14, fontweight='bold')
    ax.set_title(f"Voltage Profile: {target_formula}", fontsize=16, pad=15)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_xlim(0, max(voltages) + 1.5)
    ax.set_ylim(0, max(evolutions) * 1.4)

    # Save figure instead of showing
    figname = f"voltage_profile_{target_formula}.png"
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    print(f" Voltage profile saved to: {figname}")
    plt.close()

def generate_report(profile, ref_energy, target_formula):
    if not profile:
        return
    data = []
    for i, d in enumerate(profile):
        voltage = -(d["chempot"] - ref_energy)
        reactants = [r.reduced_formula for r in d["reaction"].reactants]
        products = [p.reduced_formula for p in d["reaction"].products]
        entry = {
            "Step": i + 1,
            "Voltage (V)": f"{voltage:.4f}",
            "Key Phase": ", ".join([p for p in products if p != "Li"]),
            "Li Consumption": f"{d['evolution']:.2f}",
            "Full Reaction": str(d['reaction'])
        }
        data.append(entry)

    df = pd.DataFrame(data)
    csv_name = f"voltage_report_{target_formula}.csv"
    df.to_csv(csv_name, index=False)
    print(f" Reaction report saved to: {csv_name}")
    # print(df.to_string()) # Uncomment to see in log

# --- MAIN EXECUTION (FOLDER BATCH MODE) ---
if __name__ == "__main__":
    print(" Initializing batch processing program...")

    # 1. Load Cache & Prepare List
    calculated_entries = load_local_cache(CACHE_FILE)
    calculated_names = set([e.attribute for e in calculated_entries])

    # Retrieve all unique structures (Targets + MP references) needed for all files in folder
    todo_list = prepare_structures_to_calc(INPUT_FOLDER, API_KEY, max_hull_energy=E_HULL_CUTOFF)

    # Filter only the structures that truly need new calculations
    real_todo = []
    for struct, name in todo_list:
        if name not in calculated_names:
            real_todo.append((struct, name))

    print(f" Cache status: Existing {len(calculated_names)} | New calculations required: {len(real_todo)}")

    # 2. Relax Structures
    if len(real_todo) > 0:
        print(f"? Detected {len(real_todo)} new structures. Loading AI model...")
        calc = get_uma_calculator()

        if calc is None:
            print(" Error: Failed to load ML model. Program stopped.")
            exit()

        print(f"\n Starting relaxation for {len(real_todo)} structures...")

        new_calculations = False
        for struct, name in tqdm(real_todo, desc="Relaxing"):
            if name in calculated_names:
                continue

            struct.remove_oxidation_states()
            entry = relax_and_get_entry(struct, calc, name=name)

            if entry:
                calculated_entries.append(entry)
                calculated_names.add(name)
                new_calculations = True

                if len(calculated_entries) % 5 == 0:
                    save_to_cache(calculated_entries, CACHE_FILE)

        if new_calculations:
            save_to_cache(calculated_entries, CACHE_FILE)
    else:
        print(" All structures already exist in cache. Skipping Model Load & Relaxation.")

    # 3. Batch Analysis & Plotting
    print("\n --- Starting Analysis for Input Files ---")
    clean_entries = ensure_neutral_entries(calculated_entries)

    # Scan the input folder again to process each target file
    input_cif_files = glob.glob(os.path.join(INPUT_FOLDER, "*.cif"))

    for cif_path in input_cif_files:
        target_name = os.path.basename(cif_path)
        print(f"\n >>> Analyzing: {target_name} <<<")

        # Find the relaxed entry for this file
        target_entry = next((e for e in clean_entries if e.attribute == target_name), None)

        if target_entry:
            # Filter entries relevant to THIS target's chemical system
            target_elements = set(target_entry.composition.elements)
            filtered_entries = [e for e in clean_entries if set(e.composition.elements).issubset(target_elements)]

            try:
                # Build Phase Diagram for this specific system
                pd_mlip = PhaseDiagram(filtered_entries)

                # 1. Plot Phase Diagram
                plotter = PDPlotter(pd_mlip, show_unstable=0.1, backend="matplotlib")
                plotter.get_plot()
                figname_pd = f"PD_{target_entry.composition.reduced_formula}_{target_name.replace('.cif','')}.png"
                plt.title(f"PD: {target_entry.composition.reduced_formula} ({target_name})")
                plt.savefig(figname_pd, dpi=300)
                plt.close()
                print(f"   Phase diagram saved: {figname_pd}")

                # 2. Stability
                ehull = pd_mlip.get_e_above_hull(target_entry)
                print(f"   Energy Above Hull: {ehull:.4f} eV/atom")

                # Save stability result
                stab_file = f"stability_{target_name.replace('.cif','')}.txt"
                with open(stab_file, "w") as f:
                    f.write(f"File: {target_name}\n")
                    f.write(f"Formula: {target_entry.composition.reduced_formula}\n")
                    f.write(f"Energy Above Hull: {ehull:.4f} eV/atom\n")
                    if ehull <= 0.05:
                        f.write("Status: Stable / Metastable\n")
                    else:
                        f.write("Status: Unstable\n")

                # 3. Voltage Profile (if Li exists)
                li_candidates = [e for e in filtered_entries if e.composition.reduced_formula == "Li"]
                if li_candidates:
                    u_li_0 = min(li_candidates, key=lambda e: e.energy_per_atom).energy_per_atom
                    try:
                        el_profile = pd_mlip.get_element_profile(Element("Li"), target_entry.composition)
                        
                        safe_formula_name = f"{target_entry.composition.reduced_formula}_{target_name.replace('.cif','')}"
                        
                        plot_professional_voltage(el_profile, safe_formula_name, u_li_0)
                        generate_report(el_profile, u_li_0, safe_formula_name)
                    except Exception as e:
                        print(f"   Voltage calc error (possibly no lithiation path): {e}")
                else:
                    print("   Skipping Voltage: No Li reference in system.")

            except Exception as e:
                print(f"   Analysis failed for {target_name}: {e}")
        else:
            print(f"   Error: Relaxed entry for {target_name} not found in cache.")
