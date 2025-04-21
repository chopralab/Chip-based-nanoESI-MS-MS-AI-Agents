#!/usr/bin/env python3
"""
QCMonitorRSD

This single class merges the logic from two separate classes/scripts:
1) QCRSD (originally from qcrsd.py)
2) QCCheck (originally from ai_agent_qc.py)
3) Additional script3 for extended monitoring

Usage example:
    from qc_monitor_rsd import QCMonitorRSD

    def main():
        monitor = QCMonitorRSD(
            script1='/home/sanjay/QTRAP_memory/sciborg_dev/UI_2/react-agent/data/convert/1_msc_run_convert_files.py.py',
            script2='/home/sanjay/QTRAP_memory/sciborg_dev/UI_2/react-agent/data/convert/2_msc_move_converted_files.py.py',
            script3='/home/sanjay/QTRAP_memory/sciborg_dev/UI_2/react-agent/data/convert/3_qc_data_monitor.py.py',  # New script
            input_dir='../qc_txt_files/',
            count_file='file_count.txt',
            output_dir='../qc_result/',
            parse_script='qc_data_parse.py',
            pause_flag='pause_monitoring.flag',
            log_file='combined_monitor.log',
            check_interval_seconds=10.0,
            delay_seconds=10.0,
            duration_minutes=10.0
        )
        result = monitor.run_all()
        if result == 0:
            print("QCMonitorRSD started successfully.")
        else:
            print("QCMonitorRSD encountered an issue during startup.")
"""

import os
import sys
import time
import logging
import subprocess
import threading
from datetime import datetime, timedelta

class QCMonitorRSD:
    """
    A merged class that:
      - Continuously runs two Python scripts in sequence (from the QCCheck logic).
      - Monitors a directory for new files and parses them (from the QCRSD logic).
      - Runs a third script in parallel for extended monitoring or additional tasks.
    """

    def __init__(self,
                 script1: str = 'scripts/1_msc_run_convert_files.py',
                 script2: str = 'scripts/2_msc_move_converted_files.py',
                 script3: str = 'scripts/3_qc_data_monitor.py',  # New script
                 input_dir: str = '../qc_txt_files/',
                 count_file: str = 'file_count.txt',
                 output_dir: str = '../qc_result/',
                 parse_script: str = 'qc_data_parse.py',
                 pause_flag: str = 'pause_monitoring.flag',
                 log_file: str = 'combined_monitor.log',
                 check_interval_seconds: float = 10.0,
                 delay_seconds: float = 10.0,
                 duration_minutes: float = 10.0):
        """
        Constructor that combines parameters from both original classes.

        Args:
            script1 (str): Path to the first script to run continuously.
            script2 (str): Path to the second script to run continuously.
            script3 (str): Path to the third script to run in parallel.
            input_dir (str): Directory to watch for new files.
            count_file (str): Path to a file storing previously seen file names.
            output_dir (str): Directory for parsed data output.
            parse_script (str): Python script to parse any new files.
            pause_flag (str): A file whose presence pauses directory monitoring.
            log_file (str): A combined log file name/path.
            check_interval_seconds (float): Interval (in seconds) to check for new files.
            delay_seconds (float): Delay (in seconds) between script1 and script2.
            duration_minutes (float): Duration (in minutes) to run scripts continuously.
        """
        # --- QCCheck-related attributes ---
        self.script1: str = script1
        self.script2: str = script2
        self.script3: str = script3  # New script

        # --- QCRSD-related attributes ---
        self.input_dir: str = (
            os.path.abspath(input_dir) if not os.path.isabs(input_dir) else input_dir
        )
        self.count_file: str = (
            os.path.abspath(count_file) if not os.path.isabs(count_file) else count_file
        )
        self.output_dir: str = (
            os.path.abspath(output_dir) if not os.path.isabs(output_dir) else output_dir
        )
        self.parse_script: str = (
            os.path.abspath(parse_script) if not os.path.isabs(parse_script) else parse_script
        )
        self.pause_flag: str = (
            os.path.abspath(pause_flag) if not os.path.isabs(pause_flag) else pause_flag
        )

        # Logging file used by all functionalities
        self.log_file: str = (
            os.path.abspath(log_file) if not os.path.isabs(log_file) else log_file
        )

        # --- Shared timing parameters ---
        self.check_interval_seconds: float = check_interval_seconds
        self.delay_seconds: float = delay_seconds
        self.duration_minutes: float = duration_minutes

        # For reference, store script's own directory
        self.script_dir: str = os.path.dirname(os.path.abspath(__file__))

        # Initialize thread references
        self.run_continuously_thread: threading.Thread = None
        self.run_monitor_thread: threading.Thread = None
        self.run_script3_thread: threading.Thread = None  # New thread for script3

        # Track monitoring start time
        self.monitor_start_time: datetime = None

    ###########################################################################
    #                   Combined Logging Setup (for all parts)                #
    ###########################################################################
    def setup_logging(self) -> int:
        """
        Sets up logging to both console and a single log file.

        Returns:
            int: 0 if successful, 1 if an error occurred.
        """
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # Log everything internally

        # Clear out any existing handlers (avoid duplicates if called multiple times)
        if logger.hasHandlers():
            logger.handlers.clear()

        # Common formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File Handler
        try:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)   # Everything goes to the file
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up file handler: {e}")
            return 1

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)  # Console shows INFO level and above
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logging.info("Logging has been set up successfully.")
        return 0

    ###########################################################################
    #                        Logic From QCCheck (scripts)                     #
    ###########################################################################
    def run_script(self, script_name: str) -> int:
        """
        Executes a Python script using subprocess.

        Args:
            script_name (str): The path to the script to execute.

        Returns:
            int: 0 if the script ran successfully, 1 if execution failed.
        """
        logging.info(f"Starting execution of {script_name}...")
        try:
            # Ensure the script exists
            if not os.path.isfile(script_name):
                logging.error(f"Script {script_name} does not exist.")
                print(f"[run_script] ERROR: Script {script_name} does not exist.")
                return 1

            # Execute the script
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                check=True
            )
            logging.info(f"Successfully executed {script_name}.")
            logging.debug(f"Output of {script_name}:\n{result.stdout}")
            if result.stderr:
                logging.warning(f"Error output from {script_name}:\n{result.stderr}")
            return 0
        except subprocess.CalledProcessError as e:
            logging.error(f"Execution of {script_name} failed with return code {e.returncode}.")
            logging.error(f"stdout: {e.stdout}")
            logging.error(f"stderr: {e.stderr}")
            print(f"[run_script] ERROR: Execution of {script_name} failed with return code {e.returncode}.")
            return 1
        except Exception as e:
            logging.exception(f"An unexpected error occurred while executing {script_name}: {e}")
            print(f"[run_script] ERROR: An unexpected error occurred while executing {script_name}: {e}")
            return 1

    def run_continuously(self, delay_seconds: float = None, duration_minutes: float = None) -> int:
        """
        Continuously runs two scripts (script1, script2) in sequence, with a delay between them,
        for a specified duration.

        Args:
            delay_seconds (float): The delay (in seconds) between the two scripts.
            duration_minutes (float): The total duration (in minutes) to run the scripts.

        Returns:
            int: The number of completed cycles.
        """
        if delay_seconds is None:
            delay_seconds = self.delay_seconds
        if duration_minutes is None:
            duration_minutes = self.duration_minutes

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        cycle = 0

        print(f"[run_continuously] Starting scripts every {delay_seconds}s for {duration_minutes} minutes.")
        logging.info(f"Starting continuous execution of scripts every {delay_seconds} seconds "
                     f"for {duration_minutes} minutes.")

        while datetime.now() < end_time:
            cycle += 1
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[run_continuously] --- Cycle {cycle} --- at {current_time}")
            logging.info(f"--- Cycle {cycle} --- at {current_time}")

            # Run script1
            print(f"[run_continuously] Running script1: {self.script1}")
            logging.info(f"Running script1: {self.script1}")
            result1 = self.run_script(self.script1)
            if result1 != 0:
                print("[run_continuously] ERROR: Script1 failed. Aborting.")
                logging.error(f"Cycle {cycle}: Aborting due to failure in {self.script1}.")
                break

            # Wait for the specified delay
            print(f"[run_continuously] Sleeping for {delay_seconds}s before running script2...")
            logging.info(f"Cycle {cycle}: Waiting {delay_seconds} seconds before running {self.script2}...")
            time.sleep(delay_seconds)

            # Run script2
            print(f"[run_continuously] Running script2: {self.script2}")
            logging.info(f"Running script2: {self.script2}")
            result2 = self.run_script(self.script2)
            if result2 != 0:
                print("[run_continuously] ERROR: Script2 failed. Aborting.")
                logging.error(f"Cycle {cycle}: Aborting due to failure in {self.script2}.")
                break

        print("[run_continuously] Continuous execution phase completed.")
        logging.info("Continuous execution phase completed.")
        return cycle

    ###########################################################################
    #                        Logic From Script3 (scripts)                     #
    ###########################################################################
    def run_script3(self) -> int:
        """
        Executes the third Python script using subprocess.

        This method can be designed to run script3 either once or in a loop,
        depending on the intended functionality.

        Returns:
            int: 0 if the script ran successfully, 1 if execution failed.
        """
        logging.info(f"Starting execution of {self.script3}...")
        try:
            # Ensure the script exists
            if not os.path.isfile(self.script3):
                logging.error(f"Script {self.script3} does not exist.")
                print(f"[run_script3] ERROR: Script {self.script3} does not exist.")
                return 1

            # Execute the script
            result = subprocess.run(
                [sys.executable, self.script3],
                capture_output=True,
                text=True,
                check=True
            )
            logging.info(f"Successfully executed {self.script3}.")
            logging.debug(f"Output of {self.script3}:\n{result.stdout}")
            if result.stderr:
                logging.warning(f"Error output from {self.script3}:\n{result.stderr}")
            return 0
        except subprocess.CalledProcessError as e:
            logging.error(f"Execution of {self.script3} failed with return code {e.returncode}.")
            logging.error(f"stdout: {e.stdout}")
            logging.error(f"stderr: {e.stderr}")
            print(f"[run_script3] ERROR: Execution of {self.script3} failed with return code {e.returncode}.")
            return 1
        except Exception as e:
            logging.exception(f"An unexpected error occurred while executing {self.script3}: {e}")
            print(f"[run_script3] ERROR: An unexpected error occurred while executing {self.script3}: {e}")
            return 1

    ###########################################################################
    #                    Logic From QCRSD (Directory Watch)                   #
    ###########################################################################
    def _notify_size_increase(self, new_count: int, newest_file: str) -> int:
        """
        Logs and prints a message when new files are detected.

        Args:
            new_count (int): The total number of files now in the directory.
            newest_file (str): The newest file name detected.

        Returns:
            int: 0 if the notification was handled successfully.
        """
        msg = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
               f"The file size has increased to {new_count} and the newest file name is {newest_file}")
        print(msg)
        logging.info(msg)
        return 0

    def _process_new_file(self, new_file: str) -> int:
        """
        Runs the parse script on the new file and searches for TIC_RSD in the output.

        Args:
            new_file (str): The name of the new file to process.

        Returns:
            int: 0 if processing succeeded, 1 otherwise.
        """
        file_path: str = os.path.join(self.input_dir, new_file)
        cmd = [sys.executable, self.parse_script, file_path]
        print(f"[_process_new_file] Running parse script: {' '.join(cmd)}")
        logging.debug(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logging.debug(f"Subprocess STDOUT: {result.stdout}")
            logging.debug(f"Subprocess STDERR: {result.stderr}")

            tic_rsd_found = False
            for line in result.stdout.splitlines():
                if line.startswith("TIC_RSD:"):
                    tic_rsd_str = line.split(":", 1)[1].strip().rstrip('%')
                    try:
                        tic_rsd = float(tic_rsd_str)
                    except ValueError:
                        tic_rsd = -1.0  # Indicate invalid float parse

                    msg = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                           f"TIC_RSD value for '{new_file}': {tic_rsd_str}%")
                    print(msg)
                    logging.info(f"TIC_RSD for '{new_file}': {tic_rsd_str}%")
                    tic_rsd_found = True

                    # Additional summary if RSD too high
                    if tic_rsd > 25:
                        warning_msg = (f"Please rerun '{new_file}', RSD too high to quantify accurately.")
                        print('_________________________')
                        print(warning_msg)
                        print('_________________________')
                        logging.warning(warning_msg)
                    break

            if not tic_rsd_found:
                msg = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                       "TIC_RSD value not found in parsing script output.")
                print(msg)
                logging.warning("TIC_RSD value not found in parsing script output.")

            return 0

        except subprocess.CalledProcessError as e:
            error_msg = f"Error running parse script on '{new_file}': {e}"
            print(f"[_process_new_file] {error_msg}")
            logging.error(error_msg)
            logging.error(f"Subprocess STDOUT: {e.stdout}")
            logging.error(f"Subprocess STDERR: {e.stderr}")
            return 1
        except Exception as e:
            error_msg = f"Unexpected error while processing '{new_file}': {e}"
            print(f"[_process_new_file] {error_msg}")
            logging.error(error_msg)
            return 1

    def check_input_dir(self) -> int:
        """
        Checks the input directory for new files, processes them if found,
        and updates the count file.

        Returns:
            int: 0 if the check was successful, 1 if an error occurred.
        """
        print(f"[check_input_dir] Checking directory: {self.input_dir}")
        logging.info(f"Checking directory: {self.input_dir}")

        if not os.path.isdir(self.input_dir):
            msg = f"Directory '{self.input_dir}' does not exist."
            print(f"[check_input_dir] {msg}")
            logging.error(msg)
            return 1

        current_files = [
            f for f in os.listdir(self.input_dir)
            if os.path.isfile(os.path.join(self.input_dir, f))
        ]
        current_count = len(current_files)
        print(f"[check_input_dir] Current number of files: {current_count}")
        logging.info(f"Current number of files in '{self.input_dir}': {current_count}")

        # Load previous files from count_file
        previous_files = set()
        if os.path.exists(self.count_file):
            try:
                with open(self.count_file, 'r') as f:
                    for line in f:
                        line_str = line.strip()
                        if line_str:
                            previous_files.add(line_str)
                logging.debug(f"Loaded {len(previous_files)} previous files from '{self.count_file}'.")
            except (ValueError, IOError) as e:
                msg = f"Error reading '{self.count_file}': {e}"
                print(f"[check_input_dir] {msg}")
                logging.error(msg)

        # Identify new files
        current_files_set = set(current_files)
        new_files = list(current_files_set - previous_files)

        if new_files:
            msg = f"Number of new files detected: {len(new_files)}"
            print(f"[check_input_dir] {msg}")
            logging.info(msg)

            for new_file in new_files:
                print(f"[check_input_dir] Processing new file: {new_file}")
                logging.info(f"Processing new file: {new_file}")
                process_code = self._process_new_file(new_file)
                if process_code != 0:
                    logging.warning(f"Failed to process file '{new_file}' properly.")

            # Find the newest file among the new ones
            try:
                newest_file = max(
                    new_files,
                    key=lambda f: os.path.getmtime(os.path.join(self.input_dir, f))
                )
                self._notify_size_increase(current_count, newest_file)
            except Exception as e:
                logging.error(f"Error determining the newest file: {e}")
                print(f"[check_input_dir] ERROR: Could not determine the newest file: {e}")
        else:
            msg = "No new files detected since the last check."
            print(f"[check_input_dir] {msg}")
            logging.info(msg)

        # Update the count_file with the current file list
        try:
            with open(self.count_file, 'w') as f:
                for file_name in current_files:
                    f.write(f"{file_name}\n")
            logging.debug(f"Updated '{self.count_file}' with the current file list.")
        except IOError as e:
            msg = f"Error writing to '{self.count_file}': {e}"
            print(f"[check_input_dir] {msg}")
            logging.error(msg)
            return 1

        return 0

    def run_monitor(self) -> int:
        """
        Continuously monitors the directory for new files, checking every
        self.check_interval_seconds. Monitoring can be paused by creating the
        pause_flag file and resumed by removing it.

        Returns:
            int: 0 if the monitor stopped normally, 1 on unexpected error.
        """
        try:
            self.monitor_start_time = datetime.now()
            iteration_count = 0
            while True:
                iteration_count += 1
                current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                elapsed_time = datetime.now() - self.monitor_start_time
                elapsed_minutes, elapsed_seconds = divmod(int(elapsed_time.total_seconds()), 60)
                print(f"[run_monitor] Iteration {iteration_count}: Checking directory '{self.input_dir}' at {current_time_str} (Elapsed: {elapsed_minutes}m {elapsed_seconds}s)...")
                logging.info(f"Iteration {iteration_count}: Checking directory '{self.input_dir}' at {current_time_str} (Elapsed: {elapsed_minutes}m {elapsed_seconds}s).")

                # Check if pause flag exists
                if os.path.exists(self.pause_flag):
                    msg = f"Monitoring paused. Remove '{os.path.basename(self.pause_flag)}' to resume."
                    print(f"[run_monitor] {msg}")
                    logging.info(msg)

                    while os.path.exists(self.pause_flag):
                        print(f"[run_monitor] Monitoring is paused. Waiting to resume...")
                        logging.info("Monitoring is paused. Waiting to resume...")
                        time.sleep(5)  # Check the pause flag every 5 seconds

                    resumed_msg = "Monitoring resumed."
                    print(f"[run_monitor] {resumed_msg}")
                    logging.info(resumed_msg)

                check_code = self.check_input_dir()
                if check_code != 0:
                    print("[run_monitor] WARNING: An error occurred during directory check. Continuing monitoring...")
                    logging.warning("An error occurred during directory check. Continuing monitoring...")

                time.sleep(self.check_interval_seconds)

        except KeyboardInterrupt:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Monitoring stopped by user.")
            logging.info("Monitoring stopped by user.")
            return 0
        except Exception as e:
            logging.error(f"Unexpected error in run_monitor: {e}")
            print(f"[run_monitor] ERROR: Unexpected error in run_monitor: {e}")
            return 1

    ###########################################################################
    #                          Logic From Script3 (scripts)                  #
    ###########################################################################
    def run_script3(self) -> int:
        """
        Executes the third Python script using subprocess.

        This method can be designed to run script3 either once or in a loop,
        depending on the intended functionality.

        Returns:
            int: 0 if the script ran successfully, 1 if execution failed.
        """
        logging.info(f"Starting execution of {self.script3}...")
        try:
            # Ensure the script exists
            if not os.path.isfile(self.script3):
                logging.error(f"Script {self.script3} does not exist.")
                print(f"[run_script3] ERROR: Script {self.script3} does not exist.")
                return 1

            # Execute the script
            result = subprocess.run(
                [sys.executable, self.script3],
                capture_output=True,
                text=True,
                check=True
            )
            logging.info(f"Successfully executed {self.script3}.")
            logging.debug(f"Output of {self.script3}:\n{result.stdout}")
            if result.stderr:
                logging.warning(f"Error output from {self.script3}:\n{result.stderr}")
            return 0
        except subprocess.CalledProcessError as e:
            logging.error(f"Execution of {self.script3} failed with return code {e.returncode}.")
            logging.error(f"stdout: {e.stdout}")
            logging.error(f"stderr: {e.stderr}")
            print(f"[run_script3] ERROR: Execution of {self.script3} failed with return code {e.returncode}.")
            return 1
        except Exception as e:
            logging.exception(f"An unexpected error occurred while executing {self.script3}: {e}")
            print(f"[run_script3] ERROR: An unexpected error occurred while executing {self.script3}: {e}")
            return 1

    ###########################################################################
    #                          One-Stop "Run All" Method                      #
    ###########################################################################
    def run_all(self) -> int:
        """
        Executes QCCheck, QCRSD, and the third script functionalities in parallel:
          1) Runs the two scripts in a continuous loop (QCCheck logic).
          2) Starts the directory monitoring loop (QCRSD logic).
          3) Runs the third script in parallel for extended monitoring or additional tasks.

        Returns:
            int:
                0 if all threads started successfully,
                1 if there was an issue starting any thread.
        """
        print("[run_all] Starting combined QCMonitorRSD workflow...")
        logging.info("Starting combined QCMonitorRSD workflow...")

        # Check if input directory exists before starting; create if not
        if not os.path.isdir(self.input_dir):
            msg = f"Monitored directory '{self.input_dir}' does not exist. Creating it."
            print(f"[run_all] {msg}")
            logging.warning(msg)
            try:
                os.makedirs(self.input_dir)
                print(f"[run_all] Directory '{self.input_dir}' created successfully.")
                logging.info(f"Directory '{self.input_dir}' created successfully.")
            except Exception as e:
                error_msg = f"Failed to create directory '{self.input_dir}': {e}"
                print(f"[run_all] {error_msg}")
                logging.error(error_msg)
                return 1

        # Start run_continuously in a separate thread
        try:
            self.run_continuously_thread = threading.Thread(
                target=self.run_continuously,
                name="RunContinuouslyThread",
                daemon=True  # Daemonize thread to exit when main thread exits
            )
            self.run_continuously_thread.start()
            print("[run_all] Started run_continuously in a separate thread.")
            logging.info("Started run_continuously in a separate thread.")
        except Exception as e:
            logging.error(f"Failed to start run_continuously thread: {e}")
            print(f"[run_all] Failed to start run_continuously thread: {e}")
            return 1

        # Start run_monitor in a separate thread
        try:
            self.run_monitor_thread = threading.Thread(
                target=self.run_monitor,
                name="RunMonitorThread",
                daemon=True  # Daemonize thread to exit when main thread exits
            )
            self.run_monitor_thread.start()
            print("[run_all] Started run_monitor in a separate thread.")
            logging.info("Started run_monitor in a separate thread.")
        except Exception as e:
            logging.error(f"Failed to start run_monitor thread: {e}")
            print(f"[run_all] Failed to start run_monitor thread: {e}")
            return 1

        # Start run_script3 in a separate thread
        try:
            self.run_script3_thread = threading.Thread(
                target=self.run_script3,
                name="RunScript3Thread",
                daemon=True  # Daemonize thread to exit when main thread exits
            )
            self.run_script3_thread.start()
            print("[run_all] Started run_script3 in a separate thread.")
            logging.info("Started run_script3 in a separate thread.")
        except Exception as e:
            logging.error(f"Failed to start run_script3 thread: {e}")
            print(f"[run_all] Failed to start run_script3 thread: {e}")
            return 1

        print("[run_all] All threads started successfully.")
        logging.info("All threads started successfully.")
        return 0

    def main(self) -> int:
        """
        Main entry point for the combined class.

        Returns:
            int: 0 if the entire workflow started successfully, or 1 if there was an error.
        """
        setup_code = self.setup_logging()
        if setup_code != 0:
            print("[main] Failed to set up logging.")
            return 1

        logging.info("Starting combined QCMonitorRSD workflow...")
        result = self.run_all()
        if result != 0:
            logging.error("QCMonitorRSD encountered an issue during startup.")
            print("[main] QCMonitorRSD encountered an issue during startup.")
            return 1

        # Keep the main thread alive to allow daemon threads to run
        try:
            total_start_time = datetime.now()
            print(f"[main] Monitoring started at {total_start_time.strftime('%Y-%m-%d %H:%M:%S')}. Press Ctrl+C to stop.")
            logging.info(f"Monitoring started at {total_start_time.strftime('%Y-%m-%d %H:%M:%S')}.")
            while True:
                time.sleep(60)  # Main thread sleeps; threads run in the background
        except KeyboardInterrupt:
            stop_time = datetime.now()
            elapsed_time = stop_time - total_start_time
            elapsed_minutes, elapsed_seconds = divmod(int(elapsed_time.total_seconds()), 60)
            print(f"\n[{stop_time.strftime('%Y-%m-%d %H:%M:%S')}] QCMonitorRSD stopped by user after {elapsed_minutes} minutes and {elapsed_seconds} seconds.")
            logging.info(f"QCMonitorRSD stopped by user after {elapsed_minutes} minutes and {elapsed_seconds} seconds.")
            return 0
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            print(f"[main] ERROR: Unexpected error in main loop: {e}")
            return 1
