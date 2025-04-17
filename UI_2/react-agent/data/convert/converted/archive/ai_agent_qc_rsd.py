#!/usr/bin/env python3
"""
qcrsd.py

This script defines a class QCRSD that continuously monitors a target directory for
new files. Every check interval (default 10 seconds), it checks for new files. If new
files are added, it logs the updated count, identifies the newest file, and runs a
parse script (e.g., qc_data_parse.py) to extract TIC_RSD and log any quality issues.

Usage (in a separate script):
    from qcrsd import QCRSD

    def main():
        monitor = QCRSD(
            input_dir='../qc_txt_files/',
            count_file='file_count.txt',
            parse_script='qc_data_parse.py',
            pause_flag='pause_monitoring.flag',
            log_file='qc_data_monitor.log',
            check_interval_seconds=10.0
        )
        monitor.main()
"""

import os
import sys
import time
import logging
import subprocess
from datetime import datetime


class QCRSD:
    """
    A class to monitor a directory for new text files and process them via a parsing script.
    """

    def __init__(self,
                 input_dir: str = '../qc_txt_files/',
                 count_file: str = 'file_count.txt',
                 output_dir: str = '../qc_result/',
                 parse_script: str = 'qc_data_parse.py',
                 pause_flag: str = 'pause_monitoring.flag',
                 log_file: str = 'qc_data_monitor.log',
                 check_interval_seconds: float = 10.0):
        """
        Initializes the QCRSD monitor with default or user-specified directories, files,
        and logging configuration.

        Args:
            input_dir (str): Path to watch for new files.
            count_file (str): Path to a text file storing previously seen files.
            output_dir (str): Directory for parsed data output (if needed).
            parse_script (str): Python script to call for parsing new files.
            pause_flag (str): A file that, if present, pauses monitoring.
            log_file (str): Log file for monitoring outputs and errors.
            check_interval_seconds (float): Frequency (in seconds) to check for new files.
        """
        self.script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.input_dir: str = (
            os.path.join(self.script_dir, input_dir)
            if not os.path.isabs(input_dir)
            else input_dir
        )
        self.count_file: str = (
            os.path.join(self.script_dir, count_file)
            if not os.path.isabs(count_file)
            else count_file
        )
        self.output_dir: str = (
            os.path.join(self.script_dir, output_dir)
            if not os.path.isabs(output_dir)
            else output_dir
        )
        self.parse_script: str = (
            os.path.join(self.script_dir, parse_script)
            if not os.path.isabs(parse_script)
            else parse_script
        )
        self.pause_flag: str = (
            os.path.join(self.script_dir, pause_flag)
            if not os.path.isabs(pause_flag)
            else pause_flag
        )
        self.log_file: str = (
            os.path.join(self.script_dir, log_file)
            if not os.path.isabs(log_file)
            else log_file
        )
        self.check_interval_seconds: float = check_interval_seconds

    def setup_logging(self) -> int:
        """
        Configures logging to both console and a log file.

        Returns:
            int: 0 if setup succeeded, 1 otherwise.
        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Clear existing handlers to avoid duplication
        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler
        try:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up file handler for '{self.log_file}': {e}")
            return 1

        # Stream handler (console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)  # Adjust to DEBUG if you want more verbosity
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logging.info("Logging has been set up successfully.")
        return 0

    def _notify_size_increase(self, new_count: int, newest_file: str) -> int:
        """
        Prints and logs a notification when the file count increases.

        Args:
            new_count (int): The new file count.
            newest_file (str): The newest file name.

        Returns:
            int: 0 if notification completed successfully.
        """
        msg: str = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"The file size has increased to {new_count} and the newest file name is {newest_file}"
        )
        print(msg)
        logging.info(msg)
        return 0

    def _process_new_file(self, new_file: str) -> int:
        """
        Runs the parse script on the new file and extracts the TIC_RSD value.

        Args:
            new_file (str): Filename of the new file in the monitored directory.

        Returns:
            int: 0 if processing succeeded, 1 if an error occurred.
        """
        file_path: str = os.path.join(self.input_dir, new_file)
        cmd: list[str] = [
            sys.executable,
            self.parse_script,
            file_path
        ]
        logging.debug(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logging.debug(f"Subprocess STDOUT: {result.stdout}")
            logging.debug(f"Subprocess STDERR: {result.stderr}")

            # Look for "TIC_RSD:" in stdout
            tic_rsd_found: bool = False
            for line in result.stdout.splitlines():
                if line.startswith("TIC_RSD:"):
                    tic_rsd_str: str = line.split(":", 1)[1].strip().rstrip('%')
                    try:
                        tic_rsd: float = float(tic_rsd_str)
                    except ValueError:
                        tic_rsd = -1.0  # Indicate invalid float parse

                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"TIC_RSD value for '{new_file}': {tic_rsd_str}%"
                    )
                    logging.info(f"TIC_RSD for '{new_file}': {tic_rsd_str}%")
                    tic_rsd_found = True

                    # Additional summary if RSD too high
                    if tic_rsd > 25:
                        print('_________________________')
                        print(f"Please rerun '{new_file}', RSD too high to quantify accurately.")
                        print('_________________________')
                        logging.warning(
                            f"Please rerun '{new_file}', RSD too high to quantify accurately."
                        )
                    break

            if not tic_rsd_found:
                msg: str = (
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    "TIC_RSD value not found in parsing script output."
                )
                print(msg)
                logging.warning("TIC_RSD value not found in parsing script output.")

            return 0

        except subprocess.CalledProcessError as e:
            logging.error(f"Error running parse script on '{new_file}': {e}")
            logging.error(f"Subprocess STDOUT: {e.stdout}")
            logging.error(f"Subprocess STDERR: {e.stderr}")
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Error running parse script on '{new_file}': {e}"
            )
            return 1
        except Exception as e:
            logging.error(f"Unexpected error while processing '{new_file}': {e}")
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Unexpected error while processing '{new_file}': {e}"
            )
            return 1

    def check_input_dir(self) -> int:
        """
        Checks the input directory for new files, processes them if found,
        and updates the count file.

        Returns:
            int: 0 if check completed successfully, 1 if an error occurred.
        """
        if not os.path.isdir(self.input_dir):
            msg: str = f"Directory '{self.input_dir}' does not exist."
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
            logging.error(msg)
            return 1

        current_files: list[str] = [
            f for f in os.listdir(self.input_dir)
            if os.path.isfile(os.path.join(self.input_dir, f))
        ]
        current_count: int = len(current_files)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Current number of files: {current_count}")
        logging.info(f"Current number of files in '{self.input_dir}': {current_count}")

        # Load previous files from count_file
        previous_files = set()
        if os.path.exists(self.count_file):
            try:
                with open(self.count_file, 'r') as f:
                    previous_files = set(line.strip() for line in f if line.strip())
            except (ValueError, IOError) as e:
                msg: str = f"Error reading '{self.count_file}': {e}"
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
                logging.error(msg)
                # We'll continue, but treat it as if we have no previous files

        # Identify new files
        current_files_set: set[str] = set(current_files)
        new_files: list[str] = list(current_files_set - previous_files)

        if new_files:
            msg: str = f"Number of new files detected: {len(new_files)}"
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
            logging.info(msg)

            for new_file in new_files:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing new file: {new_file}")
                logging.info(f"Processing new file: {new_file}")
                process_code: int = self._process_new_file(new_file)
                if process_code != 0:
                    # If parsing failed, we can decide to continue or return 1
                    # Here, let's just log it and continue
                    logging.warning(f"Failed to process file '{new_file}' properly.")

            # Find the newest file among the new ones
            newest_file: str = max(
                new_files,
                key=lambda f: os.path.getmtime(os.path.join(self.input_dir, f))
            )
            self._notify_size_increase(current_count, newest_file)
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No new files detected since the last check.")
            logging.info("No new files detected since the last check.")

        # Update the count_file with the current file list
        try:
            with open(self.count_file, 'w') as f:
                for file_name in current_files:
                    f.write(f"{file_name}\n")
            logging.debug(f"Updated '{self.count_file}' with the current file list.")
        except IOError as e:
            msg: str = f"Error writing to '{self.count_file}': {e}"
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
            logging.error(msg)
            return 1

        return 0

    def run_monitor(self) -> int:
        """
        Continuously monitors the directory, checking for new files every
        check_interval_seconds. Monitoring can be paused by creating the
        pause_flag file, and resumed by removing it.

        Returns:
            int: 0 for normal completion, 1 if an error occurs.
        """
        try:
            while True:
                # Check if pause flag exists
                if os.path.exists(self.pause_flag):
                    msg: str = (
                        f"Monitoring paused. Remove '{os.path.basename(self.pause_flag)}' to resume."
                    )
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
                    logging.info(msg)

                    while os.path.exists(self.pause_flag):
                        time.sleep(5)  # Check the pause flag every 5 seconds

                    resumed_msg: str = "Monitoring resumed."
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {resumed_msg}")
                    logging.info(resumed_msg)

                check_code: int = self.check_input_dir()
                if check_code != 0:
                    logging.warning("An error occurred during directory check. Continuing monitoring...")

                time.sleep(self.check_interval_seconds)

        except KeyboardInterrupt:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Monitoring stopped by user.")
            logging.info("Monitoring stopped by user.")
            return 0
        except Exception as e:
            logging.error(f"Unexpected error in run_monitor: {e}")
            return 1

    def main(self) -> int:
        """
        Main entry point to set up logging and start the monitor loop.

        Returns:
            int: 0 for normal termination, 1 for errors.
        """
        setup_code: int = self.setup_logging()
        if setup_code != 0:
            return 1

        logging.info("Starting QCRSD monitor...")
        monitor_code: int = self.run_monitor()
        return monitor_code
