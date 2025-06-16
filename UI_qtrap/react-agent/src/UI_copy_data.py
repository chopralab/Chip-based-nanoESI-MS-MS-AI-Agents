import os
import shutil
from pathlib import Path

# --- CONFIGURE PATHS ---
SRC_DIR = "/mnt/d_drive/Analyst Data/Projects/API Instrument/Data"
DST_DIR = os.path.expanduser("~/Chip-based-nanoESI-MS-MS-AI-Agents/server_data")

def get_last_n_files(folder, n=6):
    # Get all files in folder
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f))]
    # Sort by modification time (descending)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[:n]

def copy_files(files, dst_folder):
    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    for file in files:
        fname = os.path.basename(file)
        dst = os.path.join(dst_folder, fname)
        print(f"Copying {file} -> {dst}")
        shutil.copy2(file, dst)

def main():
    last_files = get_last_n_files(SRC_DIR, n=6)
    if not last_files:
        print("No files found in source directory.")
        return
    copy_files(last_files, DST_DIR)
    print("Copy complete.")

if __name__ == "__main__":
    main()
