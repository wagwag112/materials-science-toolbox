import os
import shutil

# Get current working directory
root_dir = os.getcwd()

# Output directory to store collected vasprun files
output_dir = os.path.join(root_dir, "collected_OUTCAR")
os.makedirs(output_dir, exist_ok=True)

vasprun_files = []

# Traverse all subdirectories to find vasprun.xml files
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file == "OUTCAR":
            full_path = os.path.join(dirpath, file)
            vasprun_files.append(full_path)

# Copy each vasprun.xml into a folder named after its parent directory
for file_path in vasprun_files:
    # Get the name of the folder containing vasprun.xml
    parent_folder_name = os.path.basename(os.path.dirname(file_path))

    # Create a new folder with that name inside output_dir
    new_folder = os.path.join(output_dir, parent_folder_name)
    os.makedirs(new_folder, exist_ok=True)

    # Destination path for vasprun.xml
    dest_file = os.path.join(new_folder, "OUTCAR")

    # Copy file (preserve metadata)
    shutil.copy2(file_path, dest_file)

print(f"Copied {len(vasprun_files)} vasprun.xml files")
