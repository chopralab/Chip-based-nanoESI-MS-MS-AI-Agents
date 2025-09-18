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
# Project Extraction and Filtering Utilities
# ----------------------------------------------------------------------------
def extract_project_from_filename(filename: str) -> Optional[str]:
    """
    Extract project name from Proj-ProjectName component in filename.
    Example: 20250916_21MeOHACN_BrainLipidExtract_LC-PC_R-1_Op-Tom_Proj-Solvent01_PC_withSPLASH.wiff
    Should return: "Solvent01"
    """
    # Split filename by underscores and search for Proj- component
    for part in filename.split('_'):
        if part.startswith('Proj-') and len(part) > 5:
            proj = part[5:]
            # Validate: alphanumeric or hyphens only
            if re.fullmatch(r'[\w-]+', proj):
                return proj
    return None

# Date extraction from messages removed.

def validate_project_files(files: List[Path], project_name: str) -> List[Path]:
    """
    Filter files list to only include files matching the specified project.
    Return only files that contain Proj-{project_name} in their filename.
    """
    filtered = [f for f in files if f"Proj-{project_name}" in f.name]
    return filtered

def extract_project_from_messages(messages: List[BaseMessage]) -> Optional[str]:
    """
    Extract project name from user messages.
    Supports patterns like:
    - "Solvent01"
    - "process project Solvent01"
    - "Proj-Solvent01"
    - "project Solvent01"
    """
    if not messages:
        return None
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
    # Try common patterns
    patterns = [
        r"Proj-([\w-]+)",
        r"project ([\w-]+)",
        r"process project ([\w-]+)",
        r"process ([\w-]+)",
        r"([\w-]+) project",
        r"\b([A-Za-z0-9\-]{4,})\b"  # catch single words like Solvent01
    ]
    for pat in patterns:
        m = re.search(pat, user_message, re.IGNORECASE)
        if m:
            proj = m.group(1)
            if re.fullmatch(r'[A-Za-z0-9\-]+', proj):
                return proj
    return None

def get_dynamic_project_str(messages: List[BaseMessage]) -> Optional[str]:
    """
    Try to extract project from messages, otherwise prompt user.
    """
    project = extract_project_from_messages(messages)
    if project:
        print(f"Detected project from input: {project}")
        return project
    print("\nEnter project name for QC processing:")
    print("Example: Solvent01 (will process files containing 'Proj-Solvent01')")
    user_input = input("Project: ").strip()
    if user_input and re.match(r'^[A-Za-z0-9\-]+$', user_input):
        return user_input
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
        # Check WSL mount accessibility first
        self.logger.info("🔍 Checking WSL mount accessibility...")
        
        # Check if /mnt/c is accessible
        mnt_c = Path("/mnt/c")
        try:
            mnt_c_exists = await asyncio.to_thread(mnt_c.exists)
            self.logger.info(f"📂 /mnt/c exists: {mnt_c_exists}")
            
            if mnt_c_exists:
                mnt_c_contents = await asyncio.to_thread(lambda: list(mnt_c.iterdir())[:5])  # First 5 items
                self.logger.info(f"📋 /mnt/c contents (first 5): {[f.name for f in mnt_c_contents]}")
            else:
                self.logger.error("❌ /mnt/c is not accessible - WSL mount may not be working")
                
        except Exception as e:
            self.logger.error(f"❌ Error checking /mnt/c: {e}")
        
        # Check the specific conversion directories
        self.logger.info(f"🔍 Checking conversion directories...")
        self.logger.info(f"📁 WSL mount dir: {self.wsl_mount_dir}")
        self.logger.info(f"📁 WSL source dir: {self.wsl_source_dir}")
        
        try:
            mount_dir_exists = await asyncio.to_thread(self.wsl_mount_dir.exists)
            self.logger.info(f"📂 Mount directory exists: {mount_dir_exists}")
            
            if mount_dir_exists:
                mount_contents = await asyncio.to_thread(lambda: list(self.wsl_mount_dir.iterdir()))
                self.logger.info(f"📋 Mount directory contents: {[f.name for f in mount_contents]}")
            
        except Exception as e:
            self.logger.error(f"❌ Error checking mount directory: {e}")
        
        # Check for batch file
        batch_wsl = self.wsl_mount_dir / "convert_files.bat"
        self.logger.info("🔍 Checking for batch file at: %s", batch_wsl)
        
        try:
            batch_exists = await asyncio.to_thread(batch_wsl.exists)
            if not batch_exists:
                raise FileNotFoundError(f"Batch file not found: {batch_wsl}")
            else:
                self.logger.info("✅ Batch file found")
                
        except Exception as e:
            self.logger.error(f"❌ Batch file error: {e}")
            raise

        windows_msconvert_dir = r"C:\Users\ChopraLab\Desktop\laptop\convert_raw"
        cmd = [
            "cmd.exe",
            "/c",
            f"cd /d {windows_msconvert_dir} && convert_files.bat"
        ]
        self.logger.info("🚀 Executing command: %s", ' '.join(cmd))
        
        try:
            result = await asyncio.to_thread(subprocess.run, cmd, check=True, capture_output=True, text=True)
            self.logger.info("✅ Conversion completed successfully.")
            self.logger.info(f"📤 Command output: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"⚠️ Command stderr: {result.stderr}")
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Conversion failed with return code {e.returncode}")
            self.logger.error(f"❌ Command stdout: {e.stdout}")
            self.logger.error(f"❌ Command stderr: {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Conversion error: {e}")
            raise

    async def move_converted(self) -> List[str]:
        # Add detailed debugging for the move process
        self.logger.info(f"🔍 Starting move_converted process...")
        self.logger.info(f"📁 WSL source directory: {self.wsl_source_dir}")
        self.logger.info(f"📁 WSL target directory: {self.wsl_target_dir}")
        
        # Check if source directory exists and is accessible
        try:
            source_exists = await asyncio.to_thread(self.wsl_source_dir.exists)
            self.logger.info(f"📂 Source directory exists: {source_exists}")
            
            if source_exists:
                # Check if we can list the directory
                source_contents = await asyncio.to_thread(lambda: list(self.wsl_source_dir.iterdir()))
                self.logger.info(f"📋 Source directory contents: {[f.name for f in source_contents]}")
            else:
                self.logger.error(f"❌ Source directory does not exist: {self.wsl_source_dir}")
                return []
                
        except Exception as e:
            self.logger.error(f"❌ Error accessing source directory {self.wsl_source_dir}: {e}")
            return []
        
        # Check if target directory exists, create if needed
        try:
            target_exists = await asyncio.to_thread(self.wsl_target_dir.exists)
            self.logger.info(f"📂 Target directory exists: {target_exists}")
            
            if not target_exists:
                await asyncio.to_thread(self.wsl_target_dir.mkdir, parents=True, exist_ok=True)
                self.logger.info(f"✅ Created target directory: {self.wsl_target_dir}")
                
        except Exception as e:
            self.logger.error(f"❌ Error with target directory {self.wsl_target_dir}: {e}")
            return []
        
        # Look for .txt files specifically
        try:
            files = await asyncio.to_thread(lambda: list(self.wsl_source_dir.glob("*.txt")))
            self.logger.info(f"🔍 Found {len(files)} .txt files to move.")
            
            if len(files) == 0:
                # Check for other file types to debug
                all_files = await asyncio.to_thread(lambda: list(self.wsl_source_dir.glob("*")))
                self.logger.info(f"📋 All files in source directory: {[f.name for f in all_files]}")
                self.logger.warning("⚠️ No .txt files found in source directory")
                return []
                
        except Exception as e:
            self.logger.error(f"❌ Error listing .txt files: {e}")
            return []

        moved_files = []
        for src in files:
            try:
                dest = self.wsl_target_dir / src.name
                self.logger.info(f"📤 Attempting to move: {src} → {dest}")
                
                # Check if source file is readable
                src_size = await asyncio.to_thread(lambda: src.stat().st_size)
                self.logger.info(f"📏 Source file size: {src_size} bytes")
                
                await asyncio.to_thread(shutil.copy2, src, dest)
                moved_files.append(src.name)
                self.logger.info(f"✅ Successfully moved {src.name} to {dest}")
                
                # Verify the file was copied
                dest_exists = await asyncio.to_thread(dest.exists)
                if dest_exists:
                    dest_size = await asyncio.to_thread(lambda: dest.stat().st_size)
                    self.logger.info(f"✅ Verified: {dest.name} ({dest_size} bytes)")
                else:
                    self.logger.error(f"❌ File copy failed - destination file doesn't exist: {dest}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error moving {src.name}: {e}")

        self.logger.info(f"📊 Move summary: {len(moved_files)} files successfully moved")
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
    project_name: Optional[str] = None,
    server_dir='/mnt/d_drive/Analyst Data/Projects/API Instrument/Data',
    raw_data_dir='/mnt/c/Users/ChopraLab/Desktop/laptop/convert_raw/raw_data'
):
    """
    Move WIFF/SCAN pairs for a given date, optionally filtering by project.
    """
    debug_log = []
    date_str = datetime.now().strftime("%Y%m%d")
    logger = logging.getLogger(f"qc_pipeline_{date_str}{'_' + project_name if project_name else ''}")
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
    # Project filtering
    if project_name:
        filtered_files = validate_project_files(all_files, project_name)
        debug_log.append(f"🔬 Project filter: Only keeping files with Proj-{project_name} in filename ({len(filtered_files)} matched)")
        nonmatching = [f.name for f in all_files if f not in filtered_files]
        if nonmatching:
            debug_log.append(f"🚫 Files not matching project: {nonmatching}")
        all_files = filtered_files
        if not all_files:
            debug_log.append(f"❌ No files found matching project 'Proj-{project_name}'")
            return [], debug_log
    pairs_to_copy = []
    target_date = datetime.now().date()
    debug_log.append(f"📅 Looking for files modified on: {target_date}")
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
    # Prompt for project name (replaces date extraction)
    messages = state.get('messages', []) if isinstance(state, dict) else []
    project_name = get_dynamic_project_str(messages)
    date_str = datetime.now().strftime("%Y%m%d")
    logger = await setup_logging(f"{date_str}{'_' + project_name if project_name else ''}")
    logger.info(f"🚀 Starting QC pipeline for date: {date_str}{' | project: ' + (project_name or 'ALL')}")
    if project_name:
        logger.info(f"Processing project: {project_name}")
    else:
        logger.info("Processing all projects (no specific project selected)")
    date_msg = f"📅 Processing QC for date: {date_str}{' | project: ' + project_name if project_name else ''}"
    logger.info("🔧 Running file flow diagnostics...")
    diagnostic_info = await diagnose_file_flow(date_str)
    move_files, move_debug = await move_wiff_pairs(project_name=project_name)
    move_summary = f"✅ move_wiff_pairs: {len(move_files)} file pairs moved." if move_files else f"❌ move_wiff_pairs: No file pairs moved for project {project_name or '[ALL]'}."
    logger.info(move_summary)
    logger.debug("\n".join(move_debug))
    converter = Convert(date_str)
    try:
        await converter.setup_directories()
        await converter.run_conversion()
        moved = await converter.move_converted()
        msg1 = f"✅ Conversion successful. Files moved: {moved}"
        logger.info(msg1)
        dirs = get_directories(date_str)
        await asyncio.to_thread(lambda: dirs['qc_csv_target'].mkdir(parents=True, exist_ok=True))
        files = await asyncio.to_thread(lambda: list(dirs['qc_text_target'].glob("*.txt")))
        if project_name:
            files = validate_project_files(files, project_name)
            logger.info(f"🔬 Project filter: Only parsing files with Proj-{project_name} in filename ({len(files)} matched)")
            if not files:
                logger.error(f"❌ No .txt files found for project {project_name} in {dirs['qc_text_target']}")
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
        try:
            await qc_results(date_str)
            qc_msg = f"✅ QC results generated for date {date_str}"
            logger.info(qc_msg)
            summary += f"\n{qc_msg}"
            try:
                await qc_validated_move(date_str)
                move_msg = f"✅ QC validated files moved to production and fail directories"
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
# Diagnostic Functions
# ----------------------------------------------------------------------------
async def diagnose_file_flow(date_str: str) -> Dict[str, Any]:
    """
    Diagnose the entire file flow to identify where files might be getting stuck
    """
    logger = logging.getLogger(f"qc_pipeline_{date_str}")
    dirs = get_directories(date_str)
    
    diagnostic_info = {
        'wsl_mount_accessible': False,
        'conversion_dirs': {},
        'qc_dirs': {},
        'file_counts': {}
    }
    
    # Check WSL mount
    try:
        mnt_c = Path("/mnt/c")
        diagnostic_info['wsl_mount_accessible'] = await asyncio.to_thread(mnt_c.exists)
        if diagnostic_info['wsl_mount_accessible']:
            logger.info("✅ WSL mount /mnt/c is accessible")
        else:
            logger.error("❌ WSL mount /mnt/c is NOT accessible")
    except Exception as e:
        logger.error(f"❌ Error checking WSL mount: {e}")
    
    # Check conversion directories
    conversion_dirs = {
        'wsl_mount': Path("/mnt/c/Users/ChopraLab/Desktop/laptop/convert_raw"),
        'raw_data': Path("/mnt/c/Users/ChopraLab/Desktop/laptop/convert_raw/raw_data"),
        'converted_files': Path("/mnt/c/Users/ChopraLab/Desktop/laptop/convert_raw/converted_files"),
    }
    
    for name, path in conversion_dirs.items():
        try:
            exists = await asyncio.to_thread(path.exists)
            diagnostic_info['conversion_dirs'][name] = {
                'path': str(path),
                'exists': exists,
                'files': []
            }
            
            if exists:
                files = await asyncio.to_thread(lambda: list(path.glob("*")))
                diagnostic_info['conversion_dirs'][name]['files'] = [f.name for f in files[:10]]  # First 10 files
                logger.info(f"📁 {name}: {len(files)} files")
            else:
                logger.warning(f"⚠️ {name} does not exist: {path}")
                
        except Exception as e:
            logger.error(f"❌ Error checking {name}: {e}")
            diagnostic_info['conversion_dirs'][name] = {'error': str(e)}
    
    # Check QC directories
    qc_dir_names = ['qc_text_target', 'qc_csv_target']
    for name in qc_dir_names:
        path = dirs[name]
        try:
            exists = await asyncio.to_thread(path.exists)
            diagnostic_info['qc_dirs'][name] = {
                'path': str(path),
                'exists': exists,
                'files': []
            }
            
            if exists:
                files = await asyncio.to_thread(lambda: list(path.glob("*")))
                diagnostic_info['qc_dirs'][name]['files'] = [f.name for f in files]
                logger.info(f"📁 {name}: {len(files)} files")
            else:
                logger.info(f"📁 {name} does not exist yet: {path}")
                
        except Exception as e:
            logger.error(f"❌ Error checking {name}: {e}")
            diagnostic_info['qc_dirs'][name] = {'error': str(e)}
    
    return diagnostic_info

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