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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# ----------------------------------------------------------------------------
# Date Configuration
# ----------------------------------------------------------------------------
DEFAULT_DATE_STR = datetime.now().strftime("%Y%m%d")

# ----------------------------------------------------------------------------
# Date Extraction from Messages
# ----------------------------------------------------------------------------
def extract_date_from_messages(messages: List[BaseMessage]) -> Optional[str]:
    """
    Extract date from user messages. Supports multiple formats:
    - YYYYMMDD (20241219)
    - YYYY-MM-DD (2024-12-19)
    - MM/DD/YYYY (12/19/2024)
    - MM-DD-YYYY (12-19-2024)
    - Natural language like "today", "yesterday"
    """
    if not messages:
        return None
    
    # Get the most recent human/user message
    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get('type') in ('human', 'user'):
                user_message = msg.get('content', '')
                break
        elif hasattr(msg, 'content'):
            if msg.__class__.__name__.lower().startswith(('human', 'user')):
                user_message = msg.content
                break
    
    if not user_message:
        return None
    
    content = user_message.lower().strip()
    
    # Handle natural language dates
    if 'today' in content:
        return datetime.now().strftime("%Y%m%d")
    elif 'yesterday' in content:
        from datetime import timedelta
        yesterday = datetime.now() - timedelta(days=1)
        return yesterday.strftime("%Y%m%d")
    
    # Pattern matching for various date formats
    date_patterns = [
        # YYYYMMDD format
        r'\b(\d{4})(\d{2})(\d{2})\b',
        # YYYY-MM-DD format
        r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',
        # MM/DD/YYYY format
        r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',
        # MM-DD-YYYY format
        r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b',
    ]
    
    for i, pattern in enumerate(date_patterns):
        match = re.search(pattern, content)
        if match:
            if i == 0:  # YYYYMMDD
                year, month, day = match.groups()
            elif i == 1:  # YYYY-MM-DD
                year, month, day = match.groups()
            else:  # MM/DD/YYYY or MM-DD-YYYY
                month, day, year = match.groups()
            
            # Validate and format
            try:
                # Pad month and day with zeros if needed
                month = month.zfill(2)
                day = day.zfill(2)
                
                # Validate date
                datetime.strptime(f"{year}{month}{day}", "%Y%m%d")
                return f"{year}{month}{day}"
            except ValueError:
                continue
    
    return None

def get_dynamic_date_str(messages: List[BaseMessage]) -> str:
    """
    Get date string from messages or use current date as default
    """
    extracted_date = extract_date_from_messages(messages)
    return extracted_date if extracted_date else DEFAULT_DATE_STR

# ----------------------------------------------------------------------------
# Directory Configuration Functions (Now Dynamic)
# ----------------------------------------------------------------------------
def get_directories(date_str: str) -> Dict[str, Path]:
    """Get all directory paths for a given date"""
    qc_text_base_dir = Path(
        "/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/text"
    )
    qc_csv_base_dir = Path(
        "/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/csv"
    )
    log_base_dir = Path(
        "/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/logs/qc"
    )
    
    return {
        'qc_text_target': qc_text_base_dir / date_str,
        'qc_csv_target': qc_csv_base_dir / date_str,
        'log_dir': log_base_dir / date_str,
        'wsl_target': qc_text_base_dir / date_str,  # Same as qc_text_target
    }

async def setup_logging(date_str: str) -> logging.Logger:
    """Setup logging for a specific date"""
    dirs = get_directories(date_str)
    log_dir = dirs['log_dir']
    await asyncio.to_thread(log_dir.mkdir, parents=True, exist_ok=True)
    log_file = log_dir / f"QC_{date_str}.log"
    
    # Create a new logger instance for this date
    logger = logging.getLogger(f"qc_pipeline_{date_str}")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized for date {date_str}. Writing to {log_file}")
    return logger

# ----------------------------------------------------------------------------
# Updated Convert Class (Now Takes Date Parameter)
# ----------------------------------------------------------------------------
class Convert:
    def __init__(self, date_str: str):
        self.date_str = date_str
        self.dirs = get_directories(date_str)
        self.logger = logging.getLogger(f"qc_pipeline_{date_str}")
        
        # WSL paths
        self.wsl_mount_dir = Path("/mnt/c/Users/ChopraLab/Desktop/laptop/convert_raw")
        self.wsl_source_dir = self.wsl_mount_dir / "converted_files"
        self.wsl_target_dir = self.dirs['wsl_target']

    async def setup_directories(self):
        await asyncio.to_thread(lambda: self.wsl_target_dir.mkdir(parents=True, exist_ok=True))

    async def run_conversion(self) -> None:
        batch_wsl = self.wsl_mount_dir / "convert_files.bat"
        self.logger.info("Checking for batch file at: %s", batch_wsl)
        if not batch_wsl.exists():
            raise FileNotFoundError(f"Batch file not found: {batch_wsl}")

        windows_msconvert_dir = r"C:\Users\ChopraLab\Desktop\laptop\convert_raw"
        cmd = [
            "cmd.exe",
            "/c",
            f"cd /d {windows_msconvert_dir} && convert_files.bat"
        ]
        self.logger.info("Executing command: %s", ' '.join(cmd))
        await asyncio.to_thread(subprocess.run, cmd, check=True)
        self.logger.info("Conversion completed successfully.")

    async def move_converted(self) -> List[str]:
        files = await asyncio.to_thread(lambda: list(self.wsl_source_dir.glob("*.txt")))
        self.logger.info("Found %d .txt files to move.", len(files))

        moved_files = []
        for src in files:
            dest = self.wsl_target_dir / src.name
            await asyncio.to_thread(shutil.copy2, src, dest)
            moved_files.append(src.name)
            self.logger.info("Moved %s to %s", src.name, dest)

        return moved_files

# ----------------------------------------------------------------------------
# Updated QTRAP_Parse Class (Unchanged but included for completeness)
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
            logger = logging.getLogger()
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
            logger = logging.getLogger()
            logger.error(f"Error saving CSV to {self.output_file}: {e}")
            return False

    def run(self) -> Optional[pd.DataFrame]:
        if self.parse_file():
            try:
                return pd.read_csv(self.output_file)
            except Exception as e:
                logger = logging.getLogger()
                logger.error(f"Error reading CSV back in: {e}")
        return None

# ----------------------------------------------------------------------------
# Updated move_wiff_pairs Function
# ----------------------------------------------------------------------------
async def move_wiff_pairs(
    date_str: str,
    server_dir='/mnt/d_drive/Analyst Data/Projects/API Instrument/Data',
    raw_data_dir='/mnt/c/Users/ChopraLab/Desktop/laptop/convert_raw/raw_data'
):
    """Updated to use the provided date_str parameter"""
    debug_log = []
    logger = logging.getLogger(f"qc_pipeline_{date_str}")
    
    try:
        server = Path(server_dir)
        debug_log.append(f"✓ Server path accessible: {server_dir}")
    except Exception as e:
        error_msg = f"✗ Error accessing server directory {server_dir}: {e}"
        debug_log.append(error_msg)
        logger.error(error_msg)
        return [], debug_log
    
    dest = Path(raw_data_dir)
    await asyncio.to_thread(dest.mkdir, parents=True, exist_ok=True)
    debug_log.append(f"✓ Destination directory ready: {raw_data_dir}")

    try:
        debug_log.append(f"🔍 Scanning directory for date {date_str}: {server_dir}")
        all_files = await asyncio.to_thread(lambda: list(server.glob('*.wiff')))
        debug_log.append(f"📄 Found {len(all_files)} .wiff files total")
    except Exception as e:
        error_msg = f"✗ Error listing files in {server_dir}: {e}"
        debug_log.append(error_msg)
        logger.error(error_msg)
        return [], debug_log

    pairs_to_copy = []

    # Parse the date string (YYYYMMDD format)
    try:
        target_date = datetime.strptime(date_str, "%Y%m%d").date()
        debug_log.append(f"📅 Looking for files modified on: {target_date}")
    except ValueError:
        debug_log.append(f"❌ Invalid date format '{date_str}'. Expected YYYYMMDD format.")
        return [], debug_log
    
    matching_files = []
    
    for wiff_file in all_files:
        file_stat = await asyncio.to_thread(wiff_file.stat)
        file_mod_time = datetime.fromtimestamp(file_stat.st_mtime)
        file_mod_date = file_mod_time.date()
        
        if file_mod_date == target_date:
            matching_files.append(wiff_file.name)
            scan_file = wiff_file.with_suffix(wiff_file.suffix + '.scan')
            scan_exists = await asyncio.to_thread(scan_file.exists)
            if scan_exists:
                pairs_to_copy.append((wiff_file, scan_file))
                debug_log.append(f"✓ Found complete pair: {wiff_file.name} (modified: {file_mod_time.strftime('%Y-%m-%d %H:%M')})")
            else:
                debug_log.append(f"⚠ Found .wiff but missing .scan: {wiff_file.name}")
    
    debug_log.append(f"📊 Files modified on {target_date}: {len(matching_files)}")
    debug_log.append(f"📦 Complete pairs found: {len(pairs_to_copy)}")
    
    if not pairs_to_copy:
        debug_log.append(f"❌ No complete file pairs found for date '{date_str}'")
        # Show recent files with their modification dates
        debug_log.append("📅 Recent files and their modification dates:")
        
        file_times = []
        for f in all_files:
            stat_info = await asyncio.to_thread(f.stat)
            file_times.append((f, stat_info.st_mtime))
        
        recent_files = sorted(file_times, key=lambda x: x[1], reverse=True)[:10]
        for f, mtime in recent_files:
            mod_time = datetime.fromtimestamp(mtime)
            debug_log.append(f"   • {f.name} - {mod_time.strftime('%Y-%m-%d %H:%M')}")

    if not pairs_to_copy:
        debug_log.append("❌ No file pairs found to copy")
        return [], debug_log

    # Copy pairs
    debug_log.append(f"📤 Starting to copy {len(pairs_to_copy)} file pairs...")
    copied_files = []
    
    for wiff_file, scan_file in pairs_to_copy:
        for src_file in (wiff_file, scan_file):
            dest_file = dest / src_file.name
            await asyncio.to_thread(shutil.copy2, src_file, dest_file)
            debug_log.append(f"✓ Copied: {src_file.name}")
        copied_files.append(wiff_file.name)

    debug_log.append(f"🎉 Successfully copied {len(copied_files)} file pairs")
    return copied_files, debug_log

# ----------------------------------------------------------------------------
# Updated QC Results Functions
# ----------------------------------------------------------------------------
async def qc_results(date_str: str) -> None:
    """Updated to use the provided date_str parameter"""
    logger = logging.getLogger(f"qc_pipeline_{date_str}")
    dirs = get_directories(date_str)
    
    source_dir = dirs['qc_csv_target']
    results_base_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results")
    results_dir = results_base_dir / date_str
    await asyncio.to_thread(results_dir.mkdir, parents=True, exist_ok=True)
    
    csv_files = await asyncio.to_thread(lambda: list(source_dir.glob("*.csv")))
    if not csv_files:
        logger.warning(f"No CSV files found in {source_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process in {source_dir}")
    
    results_data = []
    
    for csv_file in csv_files:
        try:
            actual_filename = csv_file.stem
            df = await asyncio.to_thread(pd.read_csv, csv_file)
            
            if not df.empty:
                first_row = df.iloc[0]
                tic_rsd_topgroup = round(first_row['TIC_RSD_TopGroupWindow'], 2)
                
                file_data = {
                    'QC_Result': 'fail' if pd.isna(tic_rsd_topgroup) or tic_rsd_topgroup > 25 else 'pass',
                    'Filename': actual_filename,
                    'TIC_RSD_TopGroupWindow': tic_rsd_topgroup,
                    'TIC_RSD_WindowBest': round(first_row['TIC_RSD_WindowBest'], 2),
                    'Summed_TIC': round(first_row['Summed_TIC'], 2),
                    'TIC_Time': round(first_row['TIC_Time'], 2),
                    'TIC_RSD_Window': first_row['TIC_RSD_Window']
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
    
    results_df = pd.DataFrame(results_data)
    
    output_columns = [
        'QC_Result', 'Filename', 'TIC_RSD_TopGroupWindow',
        'TIC_RSD_WindowBest', 'Summed_TIC', 'TIC_Time', 'TIC_RSD_Window'
    ]
    
    results_df = results_df[output_columns]
    
    for col in results_df.select_dtypes(include=['float64', 'int64']).columns:
        results_df[col] = results_df[col].round(2)
    
    results_file = results_dir / f"QC_{date_str}_RESULTS.csv"
    
    try:
        await asyncio.to_thread(results_df.to_csv, results_file, index=False)
        logger.info(f"✅ QC results saved to {results_file} ({len(results_df)} rows)")
    except Exception as e:
        logger.error(f"Error saving QC results to {results_file}: {e}")

async def qc_validated_move(date_str: str) -> None:
    """Updated to use the provided date_str parameter"""
    logger = logging.getLogger(f"qc_pipeline_{date_str}")
    dirs = get_directories(date_str)
    
    qc_results_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results") / date_str
    qc_text_dir = dirs['qc_text_target']
    
    prod_text_base_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/text")
    prod_text_dir = prod_text_base_dir / date_str
    await asyncio.to_thread(prod_text_dir.mkdir, parents=True, exist_ok=True)
    
    fail_text_base_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/worklist/qc_fail")
    fail_text_dir = fail_text_base_dir / date_str
    await asyncio.to_thread(fail_text_dir.mkdir, parents=True, exist_ok=True)
    
    results_file = qc_results_dir / f"QC_{date_str}_RESULTS.csv"
    
    if not results_file.exists():
        logger.error(f"QC results file not found: {results_file}")
        return
    
    try:
        results_df = await asyncio.to_thread(pd.read_csv, results_file)
        passed_files = results_df[results_df['QC_Result'] == 'pass']['Filename'].tolist()
        failed_files = results_df[results_df['QC_Result'] == 'fail']['Filename'].tolist()
        
        logger.info(f"Found {len(passed_files)} files that passed QC and {len(failed_files)} files that failed QC")
        
        moved_passed_files = []
        moved_failed_files = []
        
        for filename in passed_files:
            src_file = qc_text_dir / f"{filename}.txt"
            src_exists = await asyncio.to_thread(src_file.exists)
            if not src_exists:
                logger.warning(f"Source file not found: {src_file}")
                continue
            
            dst_file = prod_text_dir / f"{filename}.txt"
            try:
                await asyncio.to_thread(shutil.copy2, src_file, dst_file)
                moved_passed_files.append(filename)
                logger.info(f"Moved passed file: {filename}.txt to {prod_text_dir}")
            except Exception as e:
                logger.error(f"Error moving passed file {filename}.txt: {e}")
        
        for filename in failed_files:
            src_file = qc_text_dir / f"{filename}.txt"
            src_exists = await asyncio.to_thread(src_file.exists)
            if not src_exists:
                logger.warning(f"Source file not found: {src_file}")
                continue
            
            dst_file = fail_text_dir / f"{filename}.txt"
            try:
                await asyncio.to_thread(shutil.copy2, src_file, dst_file)
                moved_failed_files.append(filename)
                logger.info(f"Moved failed file: {filename}.txt to {fail_text_dir}")
            except Exception as e:
                logger.error(f"Error moving failed file {filename}.txt: {e}")
        
        if moved_passed_files:
            logger.info(f"✅ Successfully moved {len(moved_passed_files)} files that passed QC to production directory")
        if moved_failed_files:
            logger.info(f"✅ Successfully moved {len(moved_failed_files)} files that failed QC to the worklist/qc_fail directory")
            
    except Exception as e:
        logger.error(f"Error processing QC results: {e}")

# ----------------------------------------------------------------------------
# Updated State and Node
# ----------------------------------------------------------------------------
class QCState(TypedDict):
    messages: List[BaseMessage]
    converted_files: Optional[List[str]]
    parsing_result: Optional[str]
    agent_state: Dict[str, Any]
    date_str: Optional[str]  # Add date_str to state

async def convert_and_parse_node(state: QCState, config: RunnableConfig) -> QCState:
    # Default missing keys for robustness
    state = {
        'messages': [],
        'converted_files': None,
        'parsing_result': None,
        'agent_state': {},
        'date_str': None,
        **state
    }
    
    # Validate messages
    messages = state.get('messages', [])
    if not isinstance(messages, list):
        messages = []
    
    # If no messages, create a default one
    if len(messages) == 0:
        messages = [{'type': 'human', 'content': 'run QC'}]
        state['messages'] = messages
    
    # Extract date from messages or use current date
    date_str = get_dynamic_date_str(messages)
    logger = await setup_logging(date_str)
    
    logger.info(f"🚀 Starting QC pipeline for date: {date_str}")
    
    # Add date confirmation to messages
    date_msg = f"📅 Processing QC for date: {date_str}"
    
    # Step 1: Move WIFF/SCAN pairs from server
    move_files, move_debug = await move_wiff_pairs(date_str=date_str)
    move_summary = f"✅ move_wiff_pairs: {len(move_files)} file pairs moved." if move_files else "❌ move_wiff_pairs: No file pairs moved."
    logger.info(move_summary)
    logger.debug("\n".join(move_debug))
    
    # Step 2: Convert and parse files
    converter = Convert(date_str)
    try:
        await converter.setup_directories()
        await converter.run_conversion()
        moved = await converter.move_converted()
        msg1 = f"✅ Conversion successful. Files moved: {moved}"
        logger.info(msg1)
        
        # Step 3: Parse files and generate CSV
        dirs = get_directories(date_str)
        await asyncio.to_thread(lambda: dirs['qc_csv_target'].mkdir(parents=True, exist_ok=True))
        
        files = await asyncio.to_thread(lambda: list(dirs['qc_text_target'].glob("*.txt")))
        results: List[str] = []
        
        for src in files:
            base = src.stem
            out_csv = dirs['qc_csv_target'] / f"{base}.csv"
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
        
        # Step 4: Generate QC results after parsing is complete
        try:
            await qc_results(date_str)
            qc_msg = f"✅ QC results generated for date {date_str}"
            logger.info(qc_msg)
            summary += f"\n{qc_msg}"
            
            # Step 5: Move validated files to production directory
            try:
                await qc_validated_move(date_str)
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
            'date_str': date_str,
            'move_wiff_pairs_result': move_files,
            'move_wiff_pairs_debug': move_debug,
            'converted_files': moved,
            'parsing_result': summary,
            'messages': state['messages'] + [
                AIMessage(content=date_msg),
                AIMessage(content=move_summary),
                AIMessage(content="\n".join(move_debug)),
                AIMessage(content=msg1),
                AIMessage(content=summary)
            ]
        }
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error during convert+parse: {e}\n{tb}")
        return {
            **state,
            'date_str': date_str,
            'move_wiff_pairs_result': move_files,
            'move_wiff_pairs_debug': move_debug,
            'converted_files': None,
            'parsing_result': f"Failed: {e}",
            'messages': state['messages'] + [
                AIMessage(content=date_msg),
                AIMessage(content=str(e))
            ]
        }

# ----------------------------------------------------------------------------
# Simple End Node to Complete the Graph
# ----------------------------------------------------------------------------
async def end_node(state: QCState, config: RunnableConfig) -> QCState:
    """Simple end node that doesn't modify state"""
    return state

# ----------------------------------------------------------------------------
# Graph Construction
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    
    # Example usage with different date formats
    test_messages = [
        # Test with specific date
        {'type': 'human', 'content': 'run QC for 20241219'},
        # Test with different date format
        # {'type': 'human', 'content': 'process QC for 2024-12-19'},
        # Test with natural language
        # {'type': 'human', 'content': 'run QC for today'},
        # Test with no date (uses current date)
        # {'type': 'human', 'content': 'run QC'},
    ]
    
    date_str = get_dynamic_date_str(test_messages)
    print(f"🚀 Starting QC pipeline for date: {date_str}")
    
    # Set up an initial state for the pipeline
    initial_state = {
        'messages': test_messages,
        'converted_files': None,
        'parsing_result': None,
        'agent_state': {},
        'date_str': None,
    }
    
    # Run the pipeline
    result = asyncio.run(convert_and_parse_node(initial_state, {}))
    
    print("\n🎉 QC Pipeline Complete!")
    print("=" * 50)
    print(f"📅 Date processed: {result.get('date_str')}")
    if result.get('move_wiff_pairs_result') is not None:
        print(f"📁 [move_wiff_pairs] Files moved: {result['move_wiff_pairs_result']}")
    if result.get('converted_files') is not None:
        print(f"🔄 [Conversion] Files moved: {result['converted_files']}")
    if result.get('parsing_result'):
        print(f"📊 [Parsing/QC] {result['parsing_result']}")
    
    # Show log file location
    dirs = get_directories(result.get('date_str', DEFAULT_DATE_STR))
    log_file = dirs['log_dir'] / f"QC_{result.get('date_str', DEFAULT_DATE_STR)}.log"
    print(f"\n📝 See log file for detailed output: {log_file}")

# Build the graph
builder = StateGraph(QCState)
builder.add_node('convert_and_parse', convert_and_parse_node)
builder.add_node('end', end_node)
builder.set_entry_point('convert_and_parse')
builder.add_edge('convert_and_parse', 'end')

graph = builder.compile()
graph.name = 'QC Convert and Parse Agent'