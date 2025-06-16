#!/usr/bin/env python3
"""
QCWorkflow: Convert raw files to .txt via MSConvert, copy outputs into WSL, and loop continuously.
"""

import sys
import time
import subprocess
import shutil
import logging
from pathlib import Path

class QCWorkflow:
    """
    Runs MSConvert batch on Windows via cmd.exe, moves any new .txt files from the WSL-mounted source
    into a designated WSL target directory, and repeats at a fixed interval.
    """
    def __init__(
        self,
        windows_msconvert_dir: str = r"C:\Users\iyer95\OneDrive - purdue.edu\Desktop\MSConvert",
        wsl_mount_dir: str = "/mnt/c/Users/iyer95/OneDrive - purdue.edu/Desktop/MSConvert",
        wsl_target_dir: str = "/home/qtrap/Chip-based-nanoESI-MS-MS-AI-Agents/UI_qtrap/react-agent/src/react_agent/convert/converted_files",
        poll_interval: float = 5.0
    ):
        """
        Initialize paths and logging.

        Args:
            windows_msconvert_dir: Windows-style path for cmd.exe context
            wsl_mount_dir: corresponding WSL-mounted path for file checks
            wsl_target_dir: destination for .txt files in WSL
            poll_interval: seconds between each conversion+move cycle
        """
        # Paths
        self.windows_ms_dir = windows_msconvert_dir
        self.wsl_ms_dir = Path(wsl_mount_dir)
        self.wsl_source_dir = self.wsl_ms_dir / "converted_files"
        self.wsl_target_dir = Path(wsl_target_dir)
        self.poll_interval = poll_interval

        # Setup logs directory
        script_dir = Path(__file__).parent
        logs_dir = script_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "convert.log"

        # Configure logging to both console and file
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file)
            ]
        )

        # Report path checks
        logging.info("Logging to: %s", log_file)
        for p in (self.wsl_ms_dir, self.wsl_source_dir, self.wsl_target_dir):
            logging.info("Path check: %s exists? %s", p, p.exists())

        # Ensure target directory exists
        self.wsl_target_dir.mkdir(parents=True, exist_ok=True)

    def run_conversion(self) -> None:
        """
        Invoke the MSConvert batch file via cmd.exe in the correct Windows directory.
        """
        batch_wsl = self.wsl_ms_dir / "convert_files.bat"
        logging.info("Checking for batch file at: %s", batch_wsl)
        if not batch_wsl.exists():
            logging.warning("Batch file not found at WSL path: %s", batch_wsl)
            return

        # Run cmd.exe in Windows context
        cmd = [
            "cmd.exe",
            "/c",
            f"cd /d {self.windows_ms_dir} && convert_files.bat"
        ]
        logging.info("Executing conversion command: %s", ' '.join(cmd))
        try:
            subprocess.run(cmd, check=True)
            logging.info("Conversion step completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error("Conversion failed with error: %s", e)

    def move_converted(self) -> None:
        """
        Copy any new .txt files from the MSConvert output directory to the WSL target directory.
        """
        files = list(self.wsl_source_dir.glob("*.txt"))
        logging.info("Found %d .txt files in source: %s", len(files), self.wsl_source_dir)
        if not files:
            return

        for src in files:
            dest = self.wsl_target_dir / src.name
            try:
                shutil.copy2(src, dest)
                logging.info("Copied: %s -> %s", src.name, dest)
            except Exception as e:
                logging.error("Error copying %s: %s", src.name, e)

    def run(self) -> None:
        """
        Main loop: run conversion then move files, sleeping between cycles.
        """
        logging.info("Starting QCWorkflow loop. Press Ctrl+C to stop.")
        try:
            while True:
                self.run_conversion()
                self.move_converted()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logging.info("Workflow interrupted by user. Exiting.")


def main():
    QCWorkflow().run()


if __name__ == "__main__":
    main()
