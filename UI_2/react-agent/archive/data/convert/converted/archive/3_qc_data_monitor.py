#!/usr/bin/env python3
"""
QCMonitorRSD - A simplified script to run three processes in parallel:
1) Run script1 to convert .raw files to text
2) Run script2 to move converted files to the 'converted' directory
3) Run script3 for additional monitoring
"""

import os
import sys
import time
import logging
import subprocess
import threading
from datetime import datetime

class QCMonitorRSD:
    def __init__(self,
                 script1='1_msc_run_convert_files.py',
                 script2='2_msc_move_converted_files.py',
                 script3='3_qc_data_monitor.py',
                 output_dir='/home/sanjay/QTRAP_memory/sciborg_dev/UI_2/react-agent/data/convert/converted',
                 log_file='qc_monitor_rsd.log',
                 delay_seconds=10.0):
        """
        Initialize the QCMonitorRSD with the specified parameters.
        """
        # Get the absolute paths for all scripts
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Resolve script paths relative to base_dir
        self.script1 = os.path.join(self.base_dir, os.path.basename(script1))
        self.script2 = os.path.join(self.base_dir, os.path.basename(script2))
        self.script3 = os.path.join(self.base_dir, os.path.basename(script3))
        
        # Use the exact output_dir path specified (not relative)
        self.output_dir = output_dir
        self.log_file = os.path.join(self.base_dir, os.path.basename(log_file))
        
        self.delay_seconds = delay_seconds
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"Using output directory: {self.output_dir}")
        
    def run_script(self, script_name):
        """Run a Python script and return 0 if successful, 1 if failed."""
        try:
            logging.info(f"Starting {script_name}")
            
            # Set environment variable for output directory to fix path doubling issue
            env = os.environ.copy()
            env['OUTPUT_DIR'] = self.output_dir
            
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                check=True,
                env=env  # Pass the environment variables
            )
            
            logging.info(f"Successfully executed {script_name}")
            if result.stdout:
                logging.debug(f"Output from {script_name}:\n{result.stdout}")
            if result.stderr:
                logging.warning(f"Warnings from {script_name}:\n{result.stderr}")
            return 0
        except Exception as e:
            logging.error(f"Error executing {script_name}: {e}")
            return 1
    
    def run_file_processing(self):
        """Run script1 and script2 in sequence continuously."""
        while True:
            try:
                # Run file conversion script (convert .raw to text)
                result1 = self.run_script(self.script1)
                
                # Wait before moving files
                time.sleep(self.delay_seconds)
                
                # Run file moving script (move converted files to output_dir)
                result2 = self.run_script(self.script2)
                
                # Short pause before next cycle
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error in file processing cycle: {e}")
                time.sleep(30)  # Wait a bit longer if there was an error
    
    def run_monitoring(self):
        """Run script3 for continuous monitoring."""
        while True:
            try:
                self.run_script(self.script3)
                time.sleep(60)  # Run monitoring once per minute
            except Exception as e:
                logging.error(f"Error in monitoring cycle: {e}")
                time.sleep(60)  # Continue trying even if there are errors
    
    def start(self):
        """Start all processes in separate threads."""
        logging.info(f"Starting QCMonitorRSD from {self.base_dir}")
        logging.info(f"Script1: {self.script1}")
        logging.info(f"Script2: {self.script2}")
        logging.info(f"Script3: {self.script3}")
        logging.info(f"Output directory: {self.output_dir}")
        
        # Start file processing thread
        processing_thread = threading.Thread(
            target=self.run_file_processing,
            daemon=True,
            name="ProcessingThread"
        )
        processing_thread.start()
        logging.info("Started file processing thread")
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(
            target=self.run_monitoring,
            daemon=True,
            name="MonitoringThread"
        )
        monitoring_thread.start()
        logging.info("Started monitoring thread")
        
        logging.info("All processes started successfully")
        print(f"QCMonitorRSD started successfully. Files will be moved to {self.output_dir}")
        print("Press Ctrl+C to stop.")
        
        # Keep main thread alive to allow daemon threads to run
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logging.info("Stopping by user request")
            print("\nQCMonitorRSD stopped by user request.")
            return 0

def main():
    """Main entry point."""
    # Hardcoded exact path to avoid any path manipulation issues
    target_dir = "/home/sanjay/QTRAP_memory/sciborg_dev/UI_2/react-agent/data/convert/converted"
    
    monitor = QCMonitorRSD(
        script1='1_msc_run_convert_files.py',
        script2='2_msc_move_converted_files.py',
        script3='3_qc_data_monitor.py',
        output_dir=target_dir,
        log_file='qc_monitor_rsd.log',
        delay_seconds=10.0
    )
    return monitor.start()

if __name__ == "__main__":
    sys.exit(main())