import os
import shutil
import subprocess
from pathlib import Path

# --- CONFIGURATION ---
SRC_CSV = os.path.expanduser(
    "~/Chip-based-nanoESI-MS-MS-AI-Agents/UI_2/react-agent/src/react_agent/worklist/test_worklist.csv"
)
DST_DIR = "/mnt/d_drive/Analyst Data/Projects/API Instrument/Batch/QTRAP_worklist"
UI_WORKLIST_PY = os.path.expanduser(
    "~/Chip-based-nanoESI-MS-MS-AI-Agents/UI_2/react-agent/src/UI_worklist.py"
)

def move_worklist_csv(src, dst_folder):
    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    dst = os.path.join(dst_folder, os.path.basename(src))
    print(f"Moving {src} -> {dst}")
    shutil.move(src, dst)
    print("Move complete.")
    return dst

def copy_worklist_csv(src, dst_folder):
    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    dst = os.path.join(dst_folder, os.path.basename(src))
    print(f"Copying {src} -> {dst}")
    shutil.copy2(src, dst)
    print("Copy complete.")
    return dst

def run_ui_worklist():
    print(f"Running {UI_WORKLIST_PY} ...")
    # Call python on the script
    result = subprocess.run(
        ["python3", UI_WORKLIST_PY], capture_output=True, text=True
    )
    print("Script output:\n", result.stdout)
    if result.stderr:
        print("Script errors:\n", result.stderr)

def main():
    # Choose either copy or move:
    # moved_csv = move_worklist_csv(SRC_CSV, DST_DIR)
    copied_csv = copy_worklist_csv(SRC_CSV, DST_DIR)
    # Run the worklist script
    run_ui_worklist()

if __name__ == "__main__":
    main()
