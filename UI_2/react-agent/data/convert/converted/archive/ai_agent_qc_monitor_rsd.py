#!/usr/bin/env python3
"""
Simple wrapper to run conversion scripts and collect outputs.
"""
import argparse
import subprocess
import time
import glob
import os
from pathlib import Path
import sys


def run(script_path):
    """
    Execute a script if it exists.
    """
    script = Path(script_path)
    if not script.is_file():
        print(f"[WARN] Script not found: {script}")
        return
    subprocess.run([sys.executable, str(script)], check=True)


def main():
    base = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Run conversion scripts and move outputs"
    )
    parser.add_argument(
        "--script1",
        default=str(base / "1_msc_run_convert_files.py"),
        help="First conversion script path",
    )
    parser.add_argument(
        "--script2",
        default=str(base / "2_msc_move_converted_files.py"),
        help="Second conversion script path",
    )
    parser.add_argument(
        "--script3",
        default=str(base / "3_qc_data_monitor.py"),
        help="Extended monitoring script path",
    )
    parser.add_argument(
        "--converted-dir",
        default=str(base / "converted"),
        help="Directory to save converted .txt files",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between loops",
    )
    args = parser.parse_args()

    converted_dir = Path(args.converted_dir)
    converted_dir.mkdir(parents=True, exist_ok=True)
    moved = set()

    while True:
        for path in (args.script1, args.script2, args.script3):
            run(path)

        for txt in glob.glob("*.txt"):
            if txt not in moved:
                dest = converted_dir / txt
                os.replace(txt, dest)
                moved.add(txt)
                print(f"Moved {txt} -> {dest}")

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
