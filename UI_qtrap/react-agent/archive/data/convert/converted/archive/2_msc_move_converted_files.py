#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def move_converted_files():
    # Define paths
    windows_converted_dir = "/mnt/c/Users/iyer95/OneDrive - purdue.edu/Desktop/MSConvert/converted_files"
    wsl_target_dir = os.path.expanduser("/home/sanjay/QTRAP_memory/sciborg_dev/UI_2/react-agent/data/convert/converted")

    # Create target directory if it doesn't exist
    Path(wsl_target_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Get list of .txt files
        txt_files = Path(windows_converted_dir).glob("*.txt")

        # Copy each file
        print(f"Moving converted files to {wsl_target_dir}...")
        files_moved = 0
        for file in txt_files:
            shutil.copy2(file, wsl_target_dir)
            files_moved += 1
            print(f"Moved: {file.name}")

        print(f"\nSuccessfully moved {files_moved} files to {wsl_target_dir}")
        print("\nFiles in target directory:")
        for file in sorted(Path(wsl_target_dir).glob("*.txt")):
            print(f"- {file.name}")

    except Exception as e:
        print(f"Error moving files: {str(e)}")

if __name__ == "__main__":
    move_converted_files()