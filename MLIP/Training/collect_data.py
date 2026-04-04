import os
import shutil
import sys

# Get filename from command line argument
if len(sys.argv) < 2:
    print("Usage: python collect_files.py <FILENAME>")
    print("Example: python collect_files.py OUTCAR")
    sys.exit(1)

target_filename = sys.argv[1]

# Get current working directory
root_dir = os.getcwd()

# Output directory named after the target file
output_dir = os.path.join(root_dir, f"collected_{target_filename}")
os.makedirs(output_dir, exist_ok=True)

found_files = []

# Traverse all subdirectories
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file == target_filename:
            full_path = os.path.join(dirpath, file)
            found_files.append(full_path)

# Copy each file into a folder named after its parent directory
for file_path in found_files:
    parent_folder_name = os.path.basename(os.path.dirname(file_path))
    new_folder = os.path.join(output_dir, parent_folder_name)
    os.makedirs(new_folder, exist_ok=True)
    dest_file = os.path.join(new_folder, target_filename)
    shutil.copy2(file_path, dest_file)

print(f"Copied {len(found_files)} '{target_filename}' files ? {output_dir}")
