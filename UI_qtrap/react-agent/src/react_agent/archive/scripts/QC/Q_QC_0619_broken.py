import asyncio
import logging
import shutil
import subprocess
import traceback
from typing import Any, Dict, List, Literal, Optional, TypedDict, Tuple
from pathlib import Path
from datetime import datetime
import os
import re

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# ----------------------------------------------------------------------------
# Date Configuration
# ----------------------------------------------------------------------------
DATE_STR = datetime.now().strftime("%Y%m%d")

# ----------------------------------------------------------------------------
# Directory Configuration for QC Text, CSV, and Logs
# ----------------------------------------------------------------------------
QC_TEXT_BASE_DIR = Path(
    "/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/text"
)
QC_TEXT_TARGET_DIR = QC_TEXT_BASE_DIR / DATE_STR

QC_CSV_BASE_DIR = Path(
    "/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/csv"
)
QC_CSV_TARGET_DIR = QC_CSV_BASE_DIR / DATE_STR

LOG_BASE_DIR = Path(
    "/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/logs/qc"
)
LOG_DIR = LOG_BASE_DIR / DATE_STR
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"QC_{DATE_STR}.log"

# ----------------------------------------------------------------------------
# Logger Setup
# ----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.info(f"Logging initialized. Writing to {LOG_FILE}")

# ----------------------------------------------------------------------------
# Conversion Directory Configuration (Windows and WSL)
# ----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
WINDOWS_MSCONVERT_DIR = r"C:\Users\ChopraLab\Desktop\laptop\convert_raw"
WSL_MOUNT_DIR = Path("/mnt/c/Users/ChopraLab/Desktop/laptop/convert_raw")
WSL_SOURCE_DIR = WSL_MOUNT_DIR / "converted_files"
WSL_TARGET_DIR = QC_TEXT_TARGET_DIR

# ----------------------------------------------------------------------------
# Convert Class
# ----------------------------------------------------------------------------
class Convert:
    async def setup_directories(self):
        await asyncio.to_thread(lambda: WSL_TARGET_DIR.mkdir(parents=True, exist_ok=True))

    async def run_conversion(self) -> None:
        batch_wsl = WSL_MOUNT_DIR / "convert_files.bat"
        logger.info("Checking for batch file at: %s", batch_wsl)
        if not batch_wsl.exists():
            raise FileNotFoundError(f"Batch file not found: {batch_wsl}")

        cmd = [
            "cmd.exe",
            "/c",
            f"cd /d {WINDOWS_MSCONVERT_DIR} && convert_files.bat"
        ]
        logger.info("Executing command: %s", ' '.join(cmd))
        await asyncio.to_thread(subprocess.run, cmd, check=True)
        logger.info("Conversion completed successfully.")

    async def move_converted(self) -> List[str]:
        files = await asyncio.to_thread(lambda: list(WSL_SOURCE_DIR.glob("*.txt")))
        logger.info("Found %d .txt files to move.", len(files))

        moved_files = []
        for src in files:
            dest = WSL_TARGET_DIR / src.name
            await asyncio.to_thread(shutil.copy2, src, dest)
            moved_files.append(src.name)
            logger.info("Moved %s to %s", src.name, dest)

        return moved_files

# ----------------------------------------------------------------------------
# QTRAP Parsing Logic (TIC RSD and CSV Export)
# ----------------------------------------------------------------------------
class QTRAP_Parse:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        window_size: int = 7,
        threshold_factor: float = 0.1,
        top_group_count: int = 6
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.top_group_count = top_group_count

        self.filenames: List[str] = []
        self.q1_values: List[float] = []
        self.q3_values: List[float] = []
        self.lipids: List[str] = []
        self.dates: List[str] = []
        self.sample_names: List[str] = []
        self.samples: List[str] = []
        self.summed_intensities: List[int] = []

    def TIC_RSD(self, lines: List[str]) -> Tuple[
        Optional[float], Optional[float], List[float], Optional[float], Optional[float]
    ]:
        tic_time = None
        time_points = []
        if len(lines) > 95:
            m = re.search(r"binary:\s+\[\d+\]\s+(.*)", lines[95].strip())
            if m:
                time_points = [float(v) for v in m.group(1).split()]
                tic_time = time_points[-1] if time_points else None

        intensity_floats = []
        if len(lines) > 98:
            m = re.search(r"binary:\s+\[\d+\]\s+(.*)", lines[98].strip())
            if m:
                try:
                    intensity_floats = [float(v) for v in m.group(1).split()]
                except ValueError:
                    intensity_floats = []

        if intensity_floats:
            mean_tot = sum(intensity_floats) / len(intensity_floats)
            if mean_tot and len(intensity_floats) > 1:
                var_tot = sum((x - mean_tot)**2 for x in intensity_floats) / (len(intensity_floats) - 1)
                std_tot = var_tot**0.5
                tic_rsd_total = (std_tot / mean_tot) * 100
            else:
                tic_rsd_total = 0.0
        else:
            tic_rsd_total = None

        window_rsds = []
        threshold = None
        if intensity_floats:
            top5 = sorted(intensity_floats, reverse=True)[:5]
            threshold = (sum(top5) / len(top5)) * self.threshold_factor

        if threshold is not None and len(intensity_floats) >= self.window_size:
            for i in range(len(intensity_floats) - self.window_size + 1):
                win = intensity_floats[i:i+self.window_size]
                if min(win) < threshold:
                    continue
                mean_w = sum(win) / len(win)
                if mean_w:
                    var_w = sum((x - mean_w)**2 for x in win) / (len(win) - 1)
                    std_w = var_w**0.5
                    window_rsds.append((std_w / mean_w) * 100)

        tic_best = min(window_rsds) if window_rsds else None
        tic_topgroup = sum(sorted(window_rsds)[:self.top_group_count]) / len(sorted(window_rsds)[:self.top_group_count]) if window_rsds else None

        return tic_time, tic_rsd_total, window_rsds, tic_best, tic_topgroup

    def Summed_TIC(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Summed_TIC'] = df.groupby('Sample_Name')['Summed_Intensity'].transform('sum')
        return df

    def parse_file(self) -> bool:
        try:
            with open(self.input_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading file {self.input_file}: {e}")
            return False

        base = os.path.splitext(os.path.basename(self.input_file))[0]
        parts = base.split('_', 1)
        date_str, sample_name = (parts + ['', ''])[:2]
        sample = 'Blank' if 'Blank' in sample_name else 'Sample'

        parsing = False
        current_q1 = current_q3 = None
        current_filename = current_lipid = ""

        for line in lines:
            if 'name:' in line:
                m = re.search(r'name:\s+([\w.]+)', line)
                if m:
                    current_filename = m.group(1)

            if 'id: SRM SIC Q1=' in line:
                m = re.search(r'Q1=(\d+\.\d+).*Q3=(\d+\.\d+).*name=([^\s]+)', line)
                if m:
                    current_q1 = float(m.group(1))
                    current_q3 = float(m.group(2))
                    current_lipid = m.group(3)

            if 'cvParam: intensity array' in line:
                parsing = True
            elif parsing and 'binary: [' in line:
                m = re.search(r'binary:\s+\[\d+\]\s+([\d\s]+)', line)
                if m and current_filename and current_q1 is not None and current_q3 is not None:
                    ints = list(map(int, m.group(1).split()))
                    self.filenames.append(current_filename)
                    self.q1_values.append(current_q1)
                    self.q3_values.append(current_q3)
                    self.lipids.append(current_lipid)
                    self.dates.append(date_str)
                    self.sample_names.append(sample_name)
                    self.samples.append(sample)
                    self.summed_intensities.append(sum(ints))
                parsing = False
            elif parsing:
                parsing = False

        if not self.filenames:
            return False

        tic_time, tic_rsd_total, tic_rsds, tic_best, tic_topgroup = self.TIC_RSD(lines)

        df = pd.DataFrame({
            'Filename':               self.filenames,
            'Q1':                     self.q1_values,
            'Q3':                     self.q3_values,
            'Lipid':                  self.lipids,
            'Date':                   self.dates,
            'Sample_Name':            self.sample_names,
            'Sample':                 self.samples,
            'Summed_Intensity':       self.summed_intensities,
            'TIC_Time':               [tic_time] * len(self.filenames),
            'TIC_RSD_Total':          [tic_rsd_total] * len(self.filenames),
            'TIC_RSD_Window':         [tic_rsds] * len(self.filenames),
            'TIC_RSD_WindowBest':     [tic_best] * len(self.filenames),
            'TIC_RSD_TopGroupWindow': [tic_topgroup] * len(self.filenames),
        })

        df = self.Summed_TIC(df)

        try:
            df.to_csv(self.output_file, index=False)
            self.parsed_df = df
            return True
        except Exception as e:
            logger.error(f"Error saving CSV to {self.output_file}: {e}")
            return False

    def run(self) -> Optional[pd.DataFrame]:
        if self.parse_file():
            try:
                return pd.read_csv(self.output_file)
            except Exception as e:
                logger.error(f"Error reading CSV back in: {e}")
        return None



# ----------------------------------------------------------------------------
# Move WIFF/SCAN Pairs from Server
# ----------------------------------------------------------------------------
import shutil
from datetime import datetime
from pathlib import Path
import asyncio

async def move_wiff_pairs(
    server_dir='/mnt/d_drive/Analyst Data/Projects/API Instrument/Data',
    raw_data_dir='/mnt/c/Users/ChopraLab/Desktop/laptop/convert_raw/raw_data',
    date_str=None
):
    debug_log = []  # Capture debug info to return
    
    # Convert network path to WSL-accessible path if needed
    try:
        server = Path(server_dir)
        debug_log.append(f"✓ Server path accessible: {server_dir}")
    except Exception as e:
        error_msg = f"✗ Error accessing server directory {server_dir}: {e}"
        debug_log.append(error_msg)
        print(error_msg)
        return [], debug_log
    
    dest = Path(raw_data_dir)
    # Always use non-blocking mkdir
    await asyncio.to_thread(dest.mkdir, parents=True, exist_ok=True)
    debug_log.append(f"✓ Destination directory ready: {raw_data_dir}")

    # List all files in server dir
    try:
        debug_log.append(f"\U0001F4C1 Scanning directory: {server_dir}")
        # Use asyncio.to_thread to make the blocking glob operation non-blocking
        all_files = await asyncio.to_thread(lambda: list(server.glob('*.wiff')))
        debug_log.append(f"\U0001F4C4 Found {len(all_files)} .wiff files total")
        
        # Debug: log first few files found - also make this non-blocking
        if all_files:
            sample_files = all_files[:10]  # First 10 files
            for i, f in enumerate(sample_files):
                # Make stat() call non-blocking too
                stat_info = await asyncio.to_thread(f.stat)
                mod_time = datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M')
                debug_log.append(f"   {i+1}. {f.name} (modified: {mod_time})")
            if len(all_files) > 10:
                debug_log.append(f"   ... and {len(all_files) - 10} more files")
        else:
            debug_log.append("   No .wiff files found")
            
    except Exception as e:
        error_msg = f"✗ Error listing files in {server_dir}: {e}"
        debug_log.append(error_msg)
        print(error_msg)
        return [], debug_log

    pairs_to_copy = []

    if date_str:
        # NEW: Filter by filename prefix instead of modification date
        if not re.match(r'^\d{8}$', date_str):
            debug_log.append(f" Invalid date format '{date_str}'. Expected YYYYMMDD format.")
            return [], debug_log
        debug_log.append(f" Looking for files with prefix: {date_str}")
        matching_files = []
        for wiff_file in all_files:
            if wiff_file.name.startswith(date_str):
                matching_files.append(wiff_file.name)
                scan_file = wiff_file.with_suffix(wiff_file.suffix + '.scan')
                scan_exists = await asyncio.to_thread(scan_file.exists)
                if scan_exists:
                    pairs_to_copy.append((wiff_file, scan_file))
                    debug_log.append(f" Found complete pair: {wiff_file.name}")
                else:
                    debug_log.append(f" Found .wiff but missing .scan: {wiff_file.name}")
        debug_log.append(f" Files with prefix {date_str}: {len(matching_files)}")
        debug_log.append(f" Complete pairs found: {len(pairs_to_copy)}")
        if not pairs_to_copy:
            debug_log.append(f" No complete file pairs found for date '{date_str}'")
    else:
        # No date provided: Find most recent .wiff, ensure .scan exists
        debug_log.append(" No date specified, looking for most recent files...")
        if not all_files:
            debug_log.append(" No .wiff files found in server directory")
            return [], debug_log
            
        # Sort by modified time, descending (most recent first) - make non-blocking
        file_times = []
        for f in all_files:
            stat_info = await asyncio.to_thread(f.stat)
            file_times.append((f, stat_info.st_mtime))
        
        sorted_files = sorted(file_times, key=lambda x: x[1], reverse=True)
        
        for wiff_file, mtime in sorted_files:
            scan_file = wiff_file.with_suffix(wiff_file.suffix + '.scan')
            scan_exists = await asyncio.to_thread(scan_file.exists)
            if scan_exists:
                pairs_to_copy.append((wiff_file, scan_file))
                mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                debug_log.append(f"✓ Selected most recent pair: {wiff_file.name} ({mod_time})")
                break  # Only take the most recent complete pair

    if not pairs_to_copy:
        debug_log.append("❌ No file pairs found to copy")
        return [], debug_log

    # Copy pairs
    debug_log.append(f"\U0001F4E4 Starting to copy {len(pairs_to_copy)} file pairs...")
    copied_files = []
    
    for wiff_file, scan_file in pairs_to_copy:
        for src_file in (wiff_file, scan_file):
            dest_file = dest / src_file.name
            await asyncio.to_thread(shutil.copy2, src_file, dest_file)
            debug_log.append(f"✓ Copied: {src_file.name}")
        copied_files.append(wiff_file.name)

    debug_log.append(f"\U0001F389 Successfully copied {len(copied_files)} file pairs")
    return copied_files, debug_log

# ----------------------------------------------------------------------------
# QC Results Generation
# ----------------------------------------------------------------------------
def qc_results(date_str: str = DATE_STR) -> None:
    """
    Process all CSV files in the qc/csv/{date} directory and create a single results file
    with QC pass/fail results based on TIC_RSD_TopGroupWindow values.
    Uses the actual CSV filename instead of the Filename column from the CSV content.
    Columns ordered as: QC_Result | Filename | TIC_RSD_TopGroupWindow | TIC_RSD_WindowBest | 
    Summed_TIC | TIC_Time | TIC_RSD_Window
    All numeric values are rounded to 2 decimal places.
    
    Args:
        date_str: Date string in YYYYMMDD format (defaults to current date)
    """
    # Set up source and target directories
    source_dir = QC_CSV_BASE_DIR / date_str
    
    # Create results directory if it doesn't exist
    results_base_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results")
    results_dir = results_base_dir / date_str
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files in the source directory
    csv_files = list(source_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {source_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process in {source_dir}")
    
    # List to store the results for each file
    results_data = []
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Use the actual CSV filename without extension
            actual_filename = csv_file.stem
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Get the first row for the required values
            if not df.empty:
                first_row = df.iloc[0]
                
                # Round the TIC_RSD_TopGroupWindow value before the pass/fail test
                tic_rsd_topgroup = round(first_row['TIC_RSD_TopGroupWindow'], 2)
                
                # Create a dictionary with the file data and rounded numeric values
                file_data = {
                    'QC_Result': 'fail' if pd.isna(tic_rsd_topgroup) or tic_rsd_topgroup > 25 else 'pass',
                    'Filename': actual_filename,
                    'TIC_RSD_TopGroupWindow': tic_rsd_topgroup,
                    'TIC_RSD_WindowBest': round(first_row['TIC_RSD_WindowBest'], 2),
                    'Summed_TIC': round(first_row['Summed_TIC'], 2),
                    'TIC_Time': round(first_row['TIC_Time'], 2),
                    'TIC_RSD_Window': first_row['TIC_RSD_Window']  # This is a list and will be handled separately
                }
                
                results_data.append(file_data)
                logger.info(f"Processed {csv_file.name}")
            else:
                logger.warning(f"File {csv_file.name} is empty")
                
        except Exception as e:
            logger.error(f"Error processing {csv_file.name}: {e}")
    
    if not results_data:
        logger.warning("No data processed successfully")
        return
    
    # Convert list of dictionaries to dataframe
    results_df = pd.DataFrame(results_data)
    
    # Define the output columns order explicitly
    output_columns = [
        'QC_Result',
        'Filename',
        'TIC_RSD_TopGroupWindow',
        'TIC_RSD_WindowBest',
        'Summed_TIC',
        'TIC_Time',
        'TIC_RSD_Window'
    ]
    
    # Ensure the DataFrame has all the required columns in the specified order
    results_df = results_df[output_columns]
    
    # Round all numeric columns to 2 decimal places
    # This will apply to all numeric columns except TIC_RSD_Window which is a list
    for col in results_df.select_dtypes(include=['float64', 'int64']).columns:
        results_df[col] = results_df[col].round(2)
    
    # Generate results filename based on the date
    results_file = results_dir / f"QC_{date_str}_RESULTS.csv"
    
    # Save the results
    try:
        results_df.to_csv(results_file, index=False)
        logger.info(f"✅ QC results saved to {results_file} ({len(results_df)} rows with values rounded to 2 decimal places)")
    except Exception as e:
        logger.error(f"Error saving QC results to {results_file}: {e}")


##########
        #######

def qc_validated_move(date_str: str = DATE_STR) -> None:
    """
    Move text files based on QC validation results:
    - Pass: move to production directory
    - Fail: move to QC fail worklist directory
    
    This function reads the QC results file, identifies files that passed or failed QC,
    and moves their corresponding text files to the appropriate directories.
    
    Args:
        date_str: Date string in YYYYMMDD format (defaults to current date)
    """
    # Set up source directories
    qc_results_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results") / date_str
    qc_text_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/text") / date_str
    
    # Set up target directories
    # For passed files
    prod_text_base_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/text")
    prod_text_dir = prod_text_base_dir / date_str
    prod_text_dir.mkdir(parents=True, exist_ok=True)
    
    # For failed files
    fail_text_base_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/worklist/qc_fail")
    fail_text_dir = fail_text_base_dir / date_str
    fail_text_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to the QC results file
    results_file = qc_results_dir / f"QC_{date_str}_RESULTS.csv"
    
    # Check if results file exists
    if not results_file.exists():
        logger.error(f"QC results file not found: {results_file}")
        return
    
    try:
        # Read the QC results
        results_df = pd.read_csv(results_file)
        
        # Get list of files that passed and failed QC
        passed_files = results_df[results_df['QC_Result'] == 'pass']['Filename'].tolist()
        failed_files = results_df[results_df['QC_Result'] == 'fail']['Filename'].tolist()
        
        logger.info(f"Found {len(passed_files)} files that passed QC and {len(failed_files)} files that failed QC")
        
        # Track successful moves
        moved_passed_files = []
        moved_failed_files = []
        
        # Move each passed file to production directory
        for filename in passed_files:
            # Source text file (need to add .txt extension since the Filename in results doesn't have it)
            src_file = qc_text_dir / f"{filename}.txt"
            
            # Check if source file exists
            if not src_file.exists():
                logger.warning(f"Source file not found: {src_file}")
                continue
            
            # Target file path
            dst_file = prod_text_dir / f"{filename}.txt"
            
            try:
                # Copy the file to the production directory
                shutil.copy2(src_file, dst_file)
                moved_passed_files.append(filename)
                logger.info(f"Moved passed file: {filename}.txt to {prod_text_dir}")
            except Exception as e:
                logger.error(f"Error moving passed file {filename}.txt: {e}")
        
        # Move each failed file to the QC fail worklist directory
        for filename in failed_files:
            # Source text file
            src_file = qc_text_dir / f"{filename}.txt"
            
            # Check if source file exists
            if not src_file.exists():
                logger.warning(f"Source file not found: {src_file}")
                continue
            
            # Target file path
            dst_file = fail_text_dir / f"{filename}.txt"
            
            try:
                # Copy the file to the fail directory
                shutil.copy2(src_file, dst_file)
                moved_failed_files.append(filename)
                logger.info(f"Moved failed file: {filename}.txt to {fail_text_dir}")
            except Exception as e:
                logger.error(f"Error moving failed file {filename}.txt: {e}")
        
        # Log summary
        if moved_passed_files:
            logger.info(f"✅ Successfully moved {len(moved_passed_files)} files that passed QC to production directory")
        else:
            logger.warning("No files were moved to production directory")
            
        if moved_failed_files:
            logger.info(f"✅ Successfully moved {len(moved_failed_files)} files that failed QC to the worklist/qc_fail directory")
        else:
            logger.warning("No files were moved to the worklist/qc_fail directory")
            
    except Exception as e:
        logger.error(f"Error processing QC results: {e}")
# ----------------------------------------------------------------------------
# Conversion + Parsing Node
# ----------------------------------------------------------------------------
class QCState(TypedDict):
    messages: List[BaseMessage]
    converted_files: Optional[List[str]]
    parsing_result: Optional[str]
    agent_state: Dict[str, Any]
    date: Optional[str]  # Optional date field for LangGraph UI or API

async def convert_and_parse_node(state: QCState, config: RunnableConfig) -> QCState:
    # Default missing keys for robustness (LangGraph, CLI, etc.)
    state = {
        'messages': [],
        'converted_files': None,
        'parsing_result': None,
        'agent_state': {},
        **state
    }
    # --- LangGraph UI Robustness Check ---
    # Ensure 'messages' key exists and is a non-empty list with at least one user/human message
    messages = state.get('messages', [])
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError(
            "Initial state must include a 'messages' key with at least one user message, e.g. "
            "{'messages': [{'type': 'human', 'content': 'do QC'}]}\n"
            "If using LangChain, pass a list of HumanMessage objects."
        )
    # Optionally, check that the first message is a user/human message
    first_msg = messages[0]
    if isinstance(first_msg, dict):
        if first_msg.get('type') not in ('human', 'user'):
            raise ValueError(
                "The first message in 'messages' must have type 'human' or 'user'. "
                "Got: {}".format(first_msg.get('type'))
            )
    # If using LangChain BaseMessage objects, check class name
    elif hasattr(first_msg, '__class__'):
        if not (first_msg.__class__.__name__.lower().startswith('human') or first_msg.__class__.__name__.lower().startswith('user')):
            raise ValueError(
                f"The first message in 'messages' must be a HumanMessage or UserMessage, got: {first_msg.__class__.__name__}"
            )
    # Continue with pipeline logic
    # Step 1: Move WIFF/SCAN pairs from server for user-specified, UI-supplied, or today's date
    # 1. Prefer date from state if present and valid
    user_date = state.get('date', None)
    if user_date and re.match(r'^20\d{6}$', str(user_date)):
        logger.info(f"Date from state: {user_date}")
    else:
        # 2. Try to extract from first user message
        first_msg_text = getattr(first_msg, 'content', '')
        date_match = re.search(r'(20\d{6})', first_msg_text)
        if date_match:
            user_date = date_match.group(1)
            logger.info(f"Date from user message: {user_date}")
        else:
            # 3. Fallback to today's date
            user_date = DATE_STR
            logger.info(f"No date found, using today's date: {DATE_STR}")
    move_files, move_debug = await move_wiff_pairs(date_str=user_date)

    move_summary = f"✅ move_wiff_pairs: {len(move_files)} file pairs moved." if move_files else "❌ move_wiff_pairs: No file pairs moved."
    logger.info(move_summary)
    logger.debug("\n".join(move_debug))
    
    converter = Convert()
    try:
        await converter.setup_directories()
        await converter.run_conversion()
        moved = await converter.move_converted()
        msg1 = f"✅ Conversion successful. Files moved: {moved}"
        logger.info(msg1)
        # prepare parsing
        # ensure CSV target dir exists
        await asyncio.to_thread(lambda: QC_CSV_TARGET_DIR.mkdir(parents=True, exist_ok=True))
        # list txt files from QC_TEXT_TARGET_DIR
        files = await asyncio.to_thread(lambda: list(QC_TEXT_TARGET_DIR.glob("*.txt")))
        results: List[str] = []
        for src in files:
            base = src.stem
            out_csv = QC_CSV_TARGET_DIR / f"{base}.csv"
            parser = QTRAP_Parse(str(src), str(out_csv))
            df = await asyncio.to_thread(parser.run)
            if df is not None:
                msg = f"✅ Parsed {src.name} → {out_csv.name} ({len(df)} rows)"
                logger.info(msg)
                results.append(msg)
            else:
                msg = f"❌ Failed to parse {src.name}"
                logger.error(msg)
                results.append(msg)
        summary = "\n".join(results) if results else "No files parsed."
        
        # Generate QC results after parsing is complete
        try:
            await asyncio.to_thread(qc_results, DATE_STR)
            qc_msg = f"✅ QC results generated for date {DATE_STR}"
            logger.info(qc_msg)
            summary += f"\n{qc_msg}"
            
            # Move validated files to production directory
            try:
                await asyncio.to_thread(qc_validated_move, DATE_STR)
                move_msg = f"✅ QC validated files moved to production directory"
                logger.info(move_msg)
                summary += f"\n{move_msg}"
            except Exception as e:
                move_msg = f"❌ Failed to move QC validated files: {e}"
                logger.error(move_msg)
                summary += f"\n{move_msg}"
                
        except Exception as e:
            qc_msg = f"❌ Failed to generate QC results: {e}"
            logger.error(qc_msg)
            summary += f"\n{qc_msg}"
            
        return {
            **state,
            'move_wiff_pairs_result': move_files,
            'move_wiff_pairs_debug': move_debug,
            'converted_files': moved,
            'parsing_result': summary,
            'messages': state['messages'] + [
                AIMessage(content=move_summary),
                AIMessage(content="\n".join(move_debug)),
                AIMessage(content=msg1),
                AIMessage(content=summary)
            ]
        }
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error during convert+parse: {e}\n{tb}")
        return {**state, 'move_wiff_pairs_result': move_files, 'move_wiff_pairs_debug': move_debug, 'converted_files': None, 'parsing_result': f"Failed: {e}", 'messages': state['messages'] + [AIMessage(content=str(e))]}
# ----------------------------------------------------------------------------
# Graph Construction
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    print("Starting QC pipeline for date:", DATE_STR)
    # Set up an initial state for the pipeline
    initial_state = {
        'messages': [],
        'converted_files': None,
        'parsing_result': None,
        'agent_state': {},
    }
    # Run the pipeline
    result = asyncio.run(convert_and_parse_node(initial_state, {}))
    print("\nQC Pipeline Complete!")
    print("---------------------")
    if result.get('move_wiff_pairs_result') is not None:
        print(f"[move_wiff_pairs] Files moved: {result['move_wiff_pairs_result']}")
    if result.get('converted_files') is not None:
        print(f"[Conversion] Files moved: {result['converted_files']}")
    if result.get('parsing_result'):
        print(f"[Parsing/QC] {result['parsing_result']}")
    print("\nSee log file for detailed output:")
    print(LOG_FILE)

builder = StateGraph(QCState)
builder.add_node('convert_and_parse', convert_and_parse_node)
builder.add_node('tools', ToolNode([]))  # no external tools
builder.set_entry_point('convert_and_parse')
builder.add_edge('convert_and_parse', 'tools')

graph = builder.compile()
graph.name = 'QC Convert and Parse Agent'