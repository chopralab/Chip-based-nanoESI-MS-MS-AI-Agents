import asyncio
import logging
import shutil
import subprocess
import traceback
import time
import signal
from typing import Any, Dict, List, Literal, Optional, TypedDict, Tuple
from pathlib import Path
from datetime import datetime, timedelta
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
# QC Thresholds
# ----------------------------------------------------------------------------
# Maximum acceptable TIC RSD value for passing QC
TIC_RSD_THRESHOLD = 25.0

# ----------------------------------------------------------------------------
# Duration Parser and Monitoring Configuration
# ----------------------------------------------------------------------------
def parse_duration(duration_str: str) -> int:
    """
    Parse duration string into total seconds with explicit unit handling.
    
    Supported formats:
    - 's' or 'seconds': multiply by 1 (e.g., '30s' = 30 seconds)
    - 'm' or 'minutes': multiply by 60 (e.g., '5m' = 5 * 60 = 300 seconds)  
    - 'h' or 'hours': multiply by 3600 (e.g., '2h' = 2 * 3600 = 7200 seconds)
    - 'd' or 'days': multiply by 86400 (e.g., '1d' = 1 * 86400 = 86400 seconds)
    
    Returns: Total duration in seconds
    """
    if not duration_str:
        raise ValueError("Duration string cannot be empty")
    
    duration_str = duration_str.strip().lower()
    
    # Constants for clarity
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 60 * 60  # 3600 seconds
    SECONDS_PER_DAY = 24 * 60 * 60  # 86400 seconds
    
    if duration_str.endswith('s') or duration_str.endswith('seconds'):
        # Extract number and convert to seconds (multiply by 1)
        number = int(duration_str.rstrip('seconds').rstrip('s'))
        total_seconds = number * 1
        return total_seconds
        
    elif duration_str.endswith('m') or duration_str.endswith('minutes'):
        # Extract number and convert minutes to seconds (multiply by 60)
        number = int(duration_str.rstrip('minutes').rstrip('m'))
        total_seconds = number * SECONDS_PER_MINUTE
        return total_seconds
        
    elif duration_str.endswith('h') or duration_str.endswith('hours'):
        # Extract number and convert hours to seconds (multiply by 3600)
        number = int(duration_str.rstrip('hours').rstrip('h'))
        total_seconds = number * SECONDS_PER_HOUR
        return total_seconds
        
    elif duration_str.endswith('d') or duration_str.endswith('days'):
        # Extract number and convert days to seconds (multiply by 86400)
        number = int(duration_str.rstrip('days').rstrip('d'))
        total_seconds = number * SECONDS_PER_DAY
        return total_seconds
        
    elif duration_str.isdigit():
        # Default: treat bare numbers as minutes
        number = int(duration_str)
        total_seconds = number * SECONDS_PER_MINUTE  # Default to minutes
        return total_seconds
        
    else:
        raise ValueError(f"Invalid duration format: '{duration_str}'. Use formats like: 5s, 5m, 2h, 1d")

def format_duration(total_seconds: float) -> str:
    """
    Format total seconds back into human readable duration string.
    
    Args:
        total_seconds: Duration in seconds to format
        
    Returns:
        Human readable string (e.g., "5m 30s", "2h 15m", "1d 3h")
    """
    seconds = int(total_seconds)
    
    if seconds < 60:
        # Less than 1 minute: show seconds only
        return f"{seconds}s"
        
    elif seconds < 3600:  # Less than 1 hour
        # Show minutes and remaining seconds
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds}s"
        else:
            return f"{minutes}m"
            
    elif seconds < 86400:  # Less than 1 day
        # Show hours and remaining minutes
        hours = seconds // 3600
        remaining_seconds = seconds % 3600
        minutes = remaining_seconds // 60
        if minutes > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{hours}h"
            
    else:  # 1 day or more
        # Show days and remaining hours
        days = seconds // 86400
        remaining_seconds = seconds % 86400
        hours = remaining_seconds // 3600
        if hours > 0:
            return f"{days}d {hours}h"
        else:
            return f"{days}d"

def extract_duration_from_messages(messages) -> str:
    """Extract duration from messages like 'monitor for 2 minutes'"""
    for msg in messages:
        content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
        if 'minute' in content.lower():
            import re
            match = re.search(r'(\d+)\s*minute', content.lower())
            if match:
                return f"{match.group(1)}m"
    return "5m"  # default

async def check_file_stability(file_path: Path, stability_minutes: int = 1) -> bool:
    """
    Check if a file is stable (not growing) by comparing file size over time.
    
    Args:
        file_path: Path to the file to check
        stability_minutes: Minutes to wait before checking again (default: 1)
    
    Returns:
        True if file size unchanged, False if still growing
    """
    logger = logging.getLogger()
    
    try:
        # Check if file exists
        if not await asyncio.to_thread(file_path.exists):
            logger.warning(f"File does not exist: {file_path}")
            return False
        
        # Get initial file size
        initial_stat = await asyncio.to_thread(file_path.stat)
        initial_size = initial_stat.st_size
        initial_time = datetime.now()
        
        logger.info(f"File {file_path.name} size: {initial_size} bytes at {initial_time.strftime('%H:%M:%S')}")
        
        # Wait for specified minutes
        wait_seconds = stability_minutes * 60
        await asyncio.sleep(wait_seconds)
        
        # Check file size again
        final_stat = await asyncio.to_thread(file_path.stat)
        final_size = final_stat.st_size
        final_time = datetime.now()
        
        logger.info(f"File {file_path.name} size: {final_size} bytes at {final_time.strftime('%H:%M:%S')}")
        
        # Compare sizes
        if initial_size == final_size:
            logger.info(f"File {file_path.name} has not changed in past {stability_minutes} minute(s), proceeding with QC")
            return True
        else:
            logger.info(f"File {file_path.name} is still growing, skipping this iteration")
            return False
            
    except Exception as e:
        logger.error(f"Error checking file stability for {file_path}: {e}")
        return False

def format_time_remaining(seconds: int) -> str:
    """Format seconds into human-readable time remaining string."""
    if seconds <= 0:
        return "0s"
    
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 and not parts:  # Only show seconds if no larger units
        parts.append(f"{secs}s")
    
    return " ".join(parts) if parts else "0s"

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
    
    # Add debug logging
    logger = logging.getLogger()
    logger.debug(f"validate_project_files: Input {len(files)} files, output {len(filtered)} files")
    logger.debug(f"Project filter: 'Proj-{project_name}'")
    
    # Log matched files
    for f in filtered:
        logger.debug(f"✅ Matched: {f.name}")
    
    # Log non-matching files (first 5 for brevity)
    non_matching = [f for f in files if f"Proj-{project_name}" not in f.name]
    if non_matching:
        logger.debug(f"❌ Non-matching files (first 5): {[f.name for f in non_matching[:5]]}")
    
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
def get_directories(project_name: str) -> Dict[str, Path]:
    """
    Get all directory paths using project name as the folder structure.
    Project name should be extracted from Proj-ProjectName (e.g., "Solvent01" from "Proj-Solvent01")
    """
    if not project_name:
        raise ValueError("Project name is required for directory structure.")
    qc_text_base_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/text")
    qc_csv_base_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/csv")
    log_base_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/logs/qc")
    results_base_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results")
    return {
        'qc_text_target': qc_text_base_dir / project_name,
        'qc_csv_target': qc_csv_base_dir / project_name,
        'log_dir': log_base_dir / project_name,
        'results_dir': results_base_dir / project_name,
        'wsl_target': qc_text_base_dir / project_name,
    }


async def setup_logging(project_name: str) -> logging.Logger:
    """Setup logging for a specific project"""
    dirs = get_directories(project_name)
    log_dir = dirs['log_dir']
    await asyncio.to_thread(log_dir.mkdir, parents=True, exist_ok=True)
    log_file = log_dir / f"QC_{project_name}.log"
    
    # Create a new logger instance for this project
    logger = logging.getLogger(f"qc_pipeline_{project_name}")
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
    
    logger.info(f"Logging initialized for project {project_name}. Writing to {log_file}")
    return logger

# ----------------------------------------------------------------------------
# Updated Convert Class (Now Takes Project Parameter)
# ----------------------------------------------------------------------------
class Convert:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.dirs = get_directories(project_name)
        self.logger = logging.getLogger(f"qc_pipeline_{project_name}")
        
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
            all_txt_files = await asyncio.to_thread(lambda: list(self.wsl_source_dir.glob("*.txt")))
            self.logger.info(f"🔍 Found {len(all_txt_files)} total .txt files in source directory.")
            
            # Filter files by project name
            files = [f for f in all_txt_files if f"Proj-{self.project_name}" in f.name]
            self.logger.info(f"🔬 After project filter 'Proj-{self.project_name}': {len(files)} files")
            
            # Debug logging for each file that will be moved
            for f in files:
                self.logger.info(f"📄 Will move: {f.name}")
            
            if len(files) == 0:
                self.logger.warning(f"⚠️ No .txt files found matching project {self.project_name}")
                if all_txt_files:
                    self.logger.info(f"📋 Available files (first 5): {[f.name for f in all_txt_files[:5]]}")
                    # Show which files don't match the project filter
                    non_matching = [f.name for f in all_txt_files if f"Proj-{self.project_name}" not in f.name]
                    if non_matching:
                        self.logger.info(f"🚫 Files not matching 'Proj-{self.project_name}': {non_matching[:5]}")
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
    raw_data_dir='/mnt/c/Users/ChopraLab/Desktop/laptop/convert_raw/raw_data',
    check_stability: bool = False,
    stability_minutes: int = 1
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
                debug_log.append(f"✓ Found complete pair: {wiff_file.name} (modified: {file_mod_time.strftime('%Y-%m-%d %H:%M')})")
                
                # Check file stability if requested
                if check_stability:
                    debug_log.append(f"🔍 Checking file stability for {wiff_file.name}...")
                    wiff_stable = await check_file_stability(wiff_file, stability_minutes)
                    scan_stable = await check_file_stability(scan_file, stability_minutes)
                    
                    if wiff_stable and scan_stable:
                        pairs_to_copy.append((wiff_file, scan_file))
                        debug_log.append(f"✅ File pair {wiff_file.name} is stable, adding to copy queue")
                    else:
                        debug_log.append(f"⏳ File pair {wiff_file.name} is still growing, skipping this iteration")
                else:
                    pairs_to_copy.append((wiff_file, scan_file))
            else:
                debug_log.append(f"⚠ Found .wiff but missing .scan: {wiff_file.name}")
    
    debug_log.append(f"📊 Files modified on {target_date}: {len(matching_files)}")
    debug_log.append(f"📦 Complete pairs found: {len(pairs_to_copy)}")
    if check_stability:
        debug_log.append(f"🔍 File stability checking: {'enabled' if check_stability else 'disabled'} ({stability_minutes} min wait)")
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
async def qc_results(project_name: str) -> None:
    """Updated to use the provided project_name parameter"""
    logger = logging.getLogger(f"qc_pipeline_{project_name}")
    dirs = get_directories(project_name)
    
    source_dir = dirs['qc_csv_target']
    results_dir = dirs['results_dir']
    await asyncio.to_thread(results_dir.mkdir, parents=True, exist_ok=True)
    
    csv_files = await asyncio.to_thread(lambda: list(source_dir.glob("*.csv")))
    logger.info(f"🔍 Found {len(csv_files)} CSV files in {source_dir}")
    
    # Debug logging for each CSV file
    for f in csv_files:
        logger.info(f"📄 Processing CSV: {f.name}")
    
    if not csv_files:
        logger.warning(f"No CSV files found in {source_dir}")
        return
    
    results_data = []
    
    for csv_file in csv_files:
        try:
            actual_filename = csv_file.stem
            df = await asyncio.to_thread(pd.read_csv, csv_file)
            
            if not df.empty:
                first_row = df.iloc[0]
                tic_rsd_topgroup = round(first_row['TIC_RSD_TopGroupWindow'], 2)
                tic_rsd_best = round(first_row['TIC_RSD_WindowBest'], 2) if not pd.isna(first_row['TIC_RSD_WindowBest']) else None
                summed_tic = round(first_row['Summed_TIC'], 2) if not pd.isna(first_row['Summed_TIC']) else None
                tic_time = round(first_row['TIC_Time'], 2) if not pd.isna(first_row['TIC_Time']) else None

                # Threshold decision and logging
                pass_check = (not pd.isna(tic_rsd_topgroup)) and (tic_rsd_topgroup <= TIC_RSD_THRESHOLD)
                logger.info(
                    f"Threshold Check: {actual_filename} TIC_RSD={tic_rsd_topgroup} ≤ {TIC_RSD_THRESHOLD} = {pass_check}"
                )
                logger.info(
                    f"QC Decision: {actual_filename} → {'PASS' if pass_check else 'FAIL'} (TIC_RSD_TopGroup: {tic_rsd_topgroup} vs threshold: {TIC_RSD_THRESHOLD})"
                )
                
                file_data = {
                    'QC_Result': 'fail' if not pass_check else 'pass',
                    'Filename': actual_filename,
                    'TIC_RSD_TopGroupWindow': tic_rsd_topgroup,
                    'TIC_RSD_WindowBest': tic_rsd_best,
                    'Summed_TIC': summed_tic,
                    'TIC_Time': tic_time,
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
    
    results_file = results_dir / f"QC_{project_name}_RESULTS.csv"
    
    try:
        await asyncio.to_thread(results_df.to_csv, results_file, index=False)
        logger.info(f" QC results saved to {results_file} ({len(results_df)} rows)")
    except Exception as e:
        logger.error(f"Error saving QC results to {results_file}: {e}")

async def qc_validated_move(project_name: str) -> None:
    """Updated to use the provided project_name parameter"""
    logger = logging.getLogger(f"qc_pipeline_{project_name}")
    dirs = get_directories(project_name)
    
    qc_results_dir = dirs['results_dir']
    qc_text_dir = dirs['qc_text_target']
    
    prod_text_base_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/text")
    prod_text_dir = prod_text_base_dir / project_name
    await asyncio.to_thread(prod_text_dir.mkdir, parents=True, exist_ok=True)
    
    fail_text_base_dir = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/worklist/qc_fail")
    fail_text_dir = fail_text_base_dir / project_name
    await asyncio.to_thread(fail_text_dir.mkdir, parents=True, exist_ok=True)
    
    results_file = qc_results_dir / f"QC_{project_name}_RESULTS.csv"
    
    if not results_file.exists():
        logger.error(f"QC results file not found: {results_file}")
        return
    
    try:
        # Read QC results and build lookup
        logger.info("Processing QC results for file movement...")
        results_df = await asyncio.to_thread(pd.read_csv, results_file)
        logger.info(f"🔍 QC Results file contains {len(results_df)} rows")
        logger.info(f"📄 Files in QC results: {results_df['Filename'].tolist()}")
        
        qc_lookup: Dict[str, Dict[str, Any]] = {}
        for _, row in results_df.iterrows():
            name = str(row['Filename'])
            qc_lookup[name] = {
                'result': row.get('QC_Result', 'fail'),
                'tic_rsd': float(row['TIC_RSD_TopGroupWindow']) if not pd.isna(row.get('TIC_RSD_TopGroupWindow')) else None,
                'tic_rsd_best': float(row['TIC_RSD_WindowBest']) if not pd.isna(row.get('TIC_RSD_WindowBest')) else None,
                'summed_tic': float(row['Summed_TIC']) if not pd.isna(row.get('Summed_TIC')) else None,
                'tic_time': float(row['TIC_Time']) if not pd.isna(row.get('TIC_Time')) else None,
            }

        passed_files = [fn for fn, v in qc_lookup.items() if v['result'] == 'pass']
        failed_files = [fn for fn, v in qc_lookup.items() if v['result'] == 'fail']

        logger.info(f"Found {len(passed_files)} files that passed QC and {len(failed_files)} files that failed QC")

        moved_passed_files: List[str] = []
        moved_failed_files: List[str] = []

        # Move PASSED files with enhanced logging
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
                meta = qc_lookup.get(filename, {})
                tic_rsd = meta.get('tic_rsd')
                tic_rsd_best = meta.get('tic_rsd_best')
                summed_tic = meta.get('summed_tic')
                tic_time = meta.get('tic_time')
                if tic_rsd is not None:
                    logger.info(
                        f"PASS QC: {filename} - TIC_RSD_TopGroupWindow: {tic_rsd:.2f} (threshold: ≤{TIC_RSD_THRESHOLD}) → moved to production"
                    )
                    logger.info(
                        f"QC Metrics for {filename}: TIC_RSD_TopGroup={tic_rsd:.2f}, TIC_RSD_Best={(tic_rsd_best if tic_rsd_best is not None else float('nan')):.2f}, Summed_TIC={(summed_tic if summed_tic is not None else float('nan')):.2f}, TIC_Time={(tic_time if tic_time is not None else float('nan')):.2f}"
                    )
                else:
                    logger.info(f"PASS QC: {filename} - TIC_RSD_TopGroupWindow: N/A (threshold: ≤{TIC_RSD_THRESHOLD}) → moved to production")
            except Exception as e:
                logger.error(f"Error moving passed file {filename}.txt: {e}")

        # Move FAILED files with enhanced logging (including missing QC data)
        # Apply project filtering to text files to only move files for this project
        all_qc_text_files = await asyncio.to_thread(lambda: list(qc_text_dir.glob('*.txt')))
        project_text_files = [f for f in all_qc_text_files if f"Proj-{project_name}" in f.name]
        qc_text_files = [p.stem for p in project_text_files]
        
        logger.info(f"🔍 Found {len(all_qc_text_files)} total .txt files in QC text directory")
        logger.info(f"🔬 After project filter 'Proj-{project_name}': {len(qc_text_files)} files")
        logger.info(f"📄 Project text files found: {qc_text_files}")
        
        missing_qc_files = [stem for stem in qc_text_files if stem not in qc_lookup]
        if missing_qc_files:
            logger.warning(f"⚠️ Project files in text directory but not in QC results: {missing_qc_files}")
        
        all_failed = list(set(failed_files + missing_qc_files))
        logger.info(f"📊 Total failed files to move: {len(all_failed)}")

        for filename in all_failed:
            src_file = qc_text_dir / f"{filename}.txt"
            src_exists = await asyncio.to_thread(src_file.exists)
            if not src_exists:
                logger.warning(f"Source file not found: {src_file}")
                continue

            dst_file = fail_text_dir / f"{filename}.txt"
            try:
                await asyncio.to_thread(shutil.copy2, src_file, dst_file)
                moved_failed_files.append(filename)
                meta = qc_lookup.get(filename)
                if meta is None:
                    logger.warning(f"WARNING: No QC results found for {filename} - moving to qc_fail directory")
                    continue
                tic_rsd = meta.get('tic_rsd')
                tic_rsd_best = meta.get('tic_rsd_best')
                summed_tic = meta.get('summed_tic')
                tic_time = meta.get('tic_time')
                if tic_rsd is not None:
                    logger.info(
                        f"FAIL QC: {filename} - TIC_RSD_TopGroupWindow: {tic_rsd:.2f} (threshold: ≤{TIC_RSD_THRESHOLD}) → moved to qc_fail directory"
                    )
                    logger.info(
                        f"QC Metrics for {filename}: TIC_RSD_TopGroup={tic_rsd:.2f}, TIC_RSD_Best={(tic_rsd_best if tic_rsd_best is not None else float('nan')):.2f}, Summed_TIC={(summed_tic if summed_tic is not None else float('nan')):.2f}, TIC_Time={(tic_time if tic_time is not None else float('nan')):.2f}"
                    )
                else:
                    logger.info(f"FAIL QC: {filename} - TIC_RSD_TopGroupWindow: N/A (threshold: ≤{TIC_RSD_THRESHOLD}) → moved to qc_fail directory")
            except Exception as e:
                logger.error(f"Error moving failed file {filename}.txt: {e}")

        total_qc_data_files = len(qc_lookup)
        total_files_moved = len(moved_passed_files) + len(moved_failed_files)
        files_with_qc_data = len([f for f in moved_passed_files + moved_failed_files if f in qc_lookup])
        files_without_qc_data = total_files_moved - files_with_qc_data
        
        logger.info(
            f"QC Results Summary: {len(moved_passed_files)} files PASSED QC, {len([f for f in moved_failed_files if f in qc_lookup])} files FAILED QC, {files_without_qc_data} files moved without QC data"
        )

        # QC Processing Report block
        logger.info("\n=== QC PROCESSING REPORT ===")
        logger.info(f"Project: {project_name}")
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"QC Data Files: {total_qc_data_files}")
        logger.info(f"Total Files Moved: {total_files_moved}")
        logger.info(f"Files Passed QC: {len(moved_passed_files)}")
        logger.info(f"Files Failed QC: {len([f for f in moved_failed_files if f in qc_lookup])}")
        logger.info(f"Files Moved Without QC Data: {files_without_qc_data}")
        logger.info(f"Threshold Used: TIC_RSD_TopGroupWindow ≤ {TIC_RSD_THRESHOLD}")
        logger.info("============================")
            
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
    project_name: Optional[str]  # Add project_name to state
    runtime_duration: Optional[str]  # Duration string like "1h", "2d"
    file_stability_check_minutes: Optional[int]  # Minutes to wait for file stability
    monitoring_start_time: Optional[float]  # Start time for monitoring
    monitoring_end_time: Optional[float]  # End time for monitoring
    loop_iteration: Optional[int]  # Current loop iteration
    files_processed: Optional[int]  # Count of files processed
    files_skipped: Optional[int]  # Count of files skipped due to instability
    errors_encountered: Optional[int]  # Count of errors

async def run_single_qc_check(project_name: str, logger) -> int:
    """Run complete QC pipeline iteration - find files, convert, parse, analyze, move"""
    try:
        logger.info(f"Starting QC check for project: {project_name}")
        
        # Step 1: Check for new WIFF files and move them
        logger.info("Step 1: Checking for new WIFF files...")
        move_files, move_debug = await move_wiff_pairs(project_name=project_name)
        
        if not move_files:
            logger.info("No new WIFF files found")
            return 0
            
        logger.info(f"Found {len(move_files)} new WIFF file pairs")
        for debug_line in move_debug:
            logger.debug(debug_line)
        
        # Step 2: Run conversion (WIFF to TXT)
        logger.info("Step 2: Converting WIFF files to TXT...")
        dirs = get_directories(project_name)
        converter = Convert(project_name)
        await converter.setup_directories()
        await converter.run_conversion()
        moved_txt_files = await converter.move_converted()
        
        if not moved_txt_files:
            logger.info("No TXT files were converted/moved")
            return 0
            
        logger.info(f"Converted and moved {len(moved_txt_files)} TXT files")
        
        # Step 3: Parse TXT files to CSV
        logger.info("Step 3: Parsing TXT files to CSV...")
        await asyncio.to_thread(lambda: dirs['qc_csv_target'].mkdir(parents=True, exist_ok=True))
        
        txt_files = await asyncio.to_thread(lambda: list(dirs['qc_text_target'].glob("*.txt")))
        txt_files = validate_project_files(txt_files, project_name)
        logger.info(f"Found {len(txt_files)} TXT files to parse")
        
        parsed_count = 0
        for txt_file in txt_files:
            base_name = txt_file.stem
            csv_output = dirs['qc_csv_target'] / f"{base_name}.csv"
            
            parser = QTRAP_Parse(str(txt_file), str(csv_output))
            df = await asyncio.to_thread(parser.run)
            
            if df is not None:
                parsed_count += 1
                logger.info(f"Parsed {txt_file.name} -> {csv_output.name}")
            else:
                logger.error(f"Failed to parse {txt_file.name}")
                
        if parsed_count == 0:
            logger.info("No files successfully parsed")
            return 0
            
        logger.info(f"Successfully parsed {parsed_count} files")
        
        # Step 4: Generate QC results
        logger.info("Step 4: Generating QC results...")
        await qc_results(project_name)
        logger.info("QC results generated")
        
        # Step 5: Move files based on QC pass/fail
        logger.info("Step 5: Moving files based on QC results...")
        await qc_validated_move(project_name)
        logger.info("Files moved based on QC validation")
        
        logger.info(f"QC check completed - processed {parsed_count} files")
        return parsed_count
        
    except Exception as e:
        logger.error(f"Error in QC check: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0

async def run_parsing_and_qc(project_name: str, moved_files: List[str]) -> None:
    """Run parsing and QC for the moved files"""
    logger = logging.getLogger(f"qc_pipeline_{project_name}")
    dirs = get_directories(project_name)
    
    # Setup CSV directory
    await asyncio.to_thread(lambda: dirs['qc_csv_target'].mkdir(parents=True, exist_ok=True))
    
    # Parse files
    all_files = await asyncio.to_thread(lambda: list(dirs['qc_text_target'].glob("*.txt")))
    files = validate_project_files(all_files, project_name)
    
    # Parse each file
    for src in files:
        base = src.stem
        out_csv = dirs['qc_csv_target'] / f"{base}.csv"
        parser = QTRAP_Parse(str(src), str(out_csv))
        df = await asyncio.to_thread(parser.run)
        if df is not None:
            logger.info(f"✅ Parsed {src.name} → {out_csv.name} ({len(df)} rows)")
    
    # Generate QC results
    await qc_results(project_name)
    
    # Move files based on QC results
    await qc_validated_move(project_name)

async def continuous_qc_monitoring(project_name: str, duration_str: str) -> Dict[str, Any]:
    """
    Main continuous monitoring function that runs for the specified duration.
    Checks for new files every 60 seconds throughout the entire duration.
    """
    logger = await setup_logging(project_name)
    
    # Parse duration and calculate end time
    total_seconds = parse_duration(duration_str)
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=total_seconds)
    
    logger.info(f"🚀 Starting continuous QC monitoring for project: {project_name}")
    logger.info(f"⏰ Duration: {duration_str} ({total_seconds} seconds)")
    logger.info(f"📅 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🏁 End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🔄 Will check for new files every 1 minute")
    
    check_interval = 60  # 1 minute in seconds
    check_count = 0
    total_files_processed = 0
    total_expected_checks = max(1, total_seconds // check_interval)
    
    while datetime.now() < end_time:
        check_count += 1
        current_time = datetime.now()
        elapsed = current_time - start_time
        remaining = end_time - current_time
        
        logger.info(f"🔍 Check #{check_count} - Elapsed: {format_duration(elapsed.total_seconds())}, Remaining: {format_duration(remaining.total_seconds())}")
        logger.info(f"🔄 Starting QC check #{check_count} of approximately {total_expected_checks}")
        logger.info(f"⏰ Current time: {current_time.strftime('%H:%M:%S')}")
        logger.info(f"📊 Progress: {elapsed.total_seconds():.0f}s/{total_seconds}s ({(elapsed.total_seconds()/total_seconds)*100:.1f}%)")
        logger.info(f"🔍 Scanning for new files with pattern: Proj-{project_name}")
        
        # Run QC check for this iteration
        files_processed = await run_single_qc_check(project_name)
        total_files_processed += files_processed
        
        if files_processed > 0:
            logger.info(f"✅ Check #{check_count} complete - processed {files_processed} files")
        else:
            logger.info(f"⏸️ Check #{check_count}: No new files found")
        
        # Wait for next check if time remaining
        if datetime.now() < end_time:
            next_check_time = min(check_interval, remaining.total_seconds())
            if next_check_time > 0:
                next_time = (current_time + timedelta(seconds=check_interval)).strftime('%H:%M:%S')
                logger.info(f"⏸️ Next check in {next_check_time:.0f} seconds (at {next_time})")
                logger.info("─" * 50)  # Visual separator between checks
                await asyncio.sleep(next_check_time)
    
    final_elapsed = datetime.now() - start_time
    logger.info(f"🏁 Continuous monitoring completed!")
    logger.info(f"⏰ Total runtime: {format_duration(final_elapsed.total_seconds())}")
    logger.info(f"🔍 Total checks performed: {check_count}")
    logger.info(f"📁 Total files processed: {total_files_processed}")
    
    return {
        'success': True,
        'duration': duration_str,
        'total_runtime': final_elapsed.total_seconds(),
        'checks_performed': check_count,
        'files_processed': total_files_processed,
        'start_time': start_time,
        'end_time': datetime.now()
    }

async def continuous_qc_monitoring_node(state: QCState, config: RunnableConfig) -> QCState:
    """
    Continuous QC monitoring node that runs for a specified duration.
    Uses the new continuous monitoring implementation.
    """
    messages = state.get('messages', []) if isinstance(state, dict) else []
    project_name = extract_project_from_messages(messages)
    runtime_duration = state.get('runtime_duration')
    
    if not project_name:
        raise ValueError("Project name is required for QC monitoring")
    
    if not runtime_duration:
        # Default to 5 minutes if no duration specified
        runtime_duration = "5m"
    
    try:
        # Run continuous monitoring
        result = await continuous_qc_monitoring(project_name, runtime_duration)
        
        return {
            **state,
            'project_name': project_name,
            'runtime_duration': runtime_duration,
            'monitoring_start_time': result['start_time'],
            'monitoring_end_time': result['end_time'],
            'loop_iteration': result['checks_performed'],
            'files_processed': result['files_processed'],
            'files_skipped': 0,
            'errors_encountered': 0,
            'parsing_result': f"Continuous monitoring completed: {result['checks_performed']} checks, {result['files_processed']} files processed"
        }
        
    except Exception as e:
        logger = logging.getLogger(f"qc_pipeline_{project_name}")
        logger.error(f"❌ Continuous monitoring failed: {e}")
        
        return {
            **state,
            'project_name': project_name,
            'runtime_duration': runtime_duration,
            'monitoring_start_time': datetime.now(),
            'monitoring_end_time': datetime.now(),
            'loop_iteration': 0,
            'files_processed': 0,
            'files_skipped': 0,
            'errors_encountered': 1,
            'parsing_result': f"Continuous monitoring failed: {e}"
        }

async def run_single_qc_iteration(project_name: str, logger) -> Dict[str, Any]:
    """
    Run a single iteration of the QC pipeline for the given project.
    Returns success status and any error messages.
    """
    try:
        dirs = get_directories(project_name)
        converter = Convert(project_name)
        
        # Setup directories
        await converter.setup_directories()
        
        # Run conversion
        await converter.run_conversion()
        
        # Move converted files
        moved = await converter.move_converted()
        logger.info(f"✅ Conversion successful. Files moved: {moved}")
        
        # Setup CSV directory
        await asyncio.to_thread(lambda: dirs['qc_csv_target'].mkdir(parents=True, exist_ok=True))
        
        # Parse files
        all_files = await asyncio.to_thread(lambda: list(dirs['qc_text_target'].glob("*.txt")))
        logger.info(f"🔍 Found {len(all_files)} total .txt files in {dirs['qc_text_target']}")
        
        if project_name:
            files = validate_project_files(all_files, project_name)
            logger.info(f"🔬 After project filter 'Proj-{project_name}': {len(files)} files")
            
            # Debug logging for each file that will be parsed
            for f in files:
                logger.info(f"📄 Will parse: {f.name}")
        else:
            files = all_files
            logger.warning(f"⚠️ No project filtering applied - processing all {len(files)} files")
        
        if not files:
            logger.warning(f"⚠️ No .txt files found for project {project_name} in {dirs['qc_text_target']}")
            if all_files:
                logger.warning(f"📋 Available files (first 5): {[f.name for f in all_files[:5]]}")
            return {'success': True, 'files_processed': 0}
        
        # Parse each file
        parsed_count = 0
        for src in files:
            base = src.stem
            out_csv = dirs['qc_csv_target'] / f"{base}.csv"
            parser = QTRAP_Parse(str(src), str(out_csv))
            df = await asyncio.to_thread(parser.run)
            if df is not None:
                logger.info(f"✅ Parsed {src.name} → {out_csv.name} ({len(df)} rows)")
                parsed_count += 1
            else:
                logger.error(f"❌ Failed to parse {src.name}")
        
        # Generate QC results
        await qc_results(project_name)
        logger.info(f"✅ QC results generated for {project_name}")
        
        # Move validated files
        await qc_validated_move(project_name)
        logger.info(f"✅ QC validated files moved to production and fail directories")
        
        return {'success': True, 'files_processed': parsed_count}
        
    except Exception as e:
        logger.error(f"❌ Error in QC iteration: {e}")
        return {'success': False, 'error': str(e)}

async def convert_and_parse_node(state: QCState, config: RunnableConfig) -> QCState:
    # Extract project name and duration
    messages = state.get('messages', [])
    project_name = get_dynamic_project_str(messages)
    duration = extract_duration_from_messages(messages) or "5m"
    
    if not project_name:
        raise ValueError("Project name is required")
    
    # Setup logging
    logger = await setup_logging(project_name)
    
    # Parse duration to seconds
    total_seconds = parse_duration(duration)
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=total_seconds)
    
    logger.info(f"Starting continuous QC monitoring for project: {project_name}")
    logger.info(f"Duration: {duration} ({total_seconds} seconds)")
    logger.info(f"Start: {start_time.strftime('%H:%M:%S')}, End: {end_time.strftime('%H:%M:%S')}")
    
    check_count = 0
    total_files_processed = 0
    
    # CONTINUOUS MONITORING LOOP
    while datetime.now() < end_time:
        check_count += 1
        current_time = datetime.now()
        elapsed = current_time - start_time
        remaining = end_time - current_time
        
        logger.info(f"Check #{check_count} - Elapsed: {elapsed.total_seconds():.0f}s, Remaining: {remaining.total_seconds():.0f}s")
        
        # Run one QC check iteration with enhanced logging
        logger.info(f"=== QC CHECK #{check_count} STARTING ===")
        files_processed = await run_single_qc_check(project_name, logger)
        total_files_processed += files_processed
        logger.info(f"=== QC CHECK #{check_count} COMPLETED: {files_processed} files processed ===")
        
        if files_processed > 0:
            logger.info(f"✅ Check #{check_count}: Successfully processed {files_processed} files")
        else:
            logger.info(f"⏸️ Check #{check_count}: No new files found or processed")
        
        # Wait 60 seconds before next check (if time remaining)
        if datetime.now() < end_time:
            sleep_time = min(60, remaining.total_seconds())
            if sleep_time > 0:
                logger.info(f"Waiting {sleep_time:.0f} seconds until next check...")
                await asyncio.sleep(sleep_time)
    
    final_elapsed = datetime.now() - start_time
    logger.info(f"Continuous monitoring completed after {final_elapsed.total_seconds():.0f} seconds")
    logger.info(f"Total checks: {check_count}, Total files processed: {total_files_processed}")
    
    return {
        **state,
        'project_name': project_name,
        'duration': duration,
        'total_checks': check_count,
        'total_files_processed': total_files_processed,
        'messages': state['messages'] + [
            AIMessage(content=f"Continuous QC monitoring completed for {project_name} ({duration})")
        ]
    }

async def diagnose_file_flow(project_name: str) -> Dict[str, Any]:
    """
    Diagnose the entire file flow to identify where files might be getting stuck
    """
    logger = logging.getLogger(f"qc_pipeline_{project_name}")
    dirs = get_directories(project_name)
    
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
# Utility Functions for State Management
# ----------------------------------------------------------------------------

def create_qc_state(
    messages: List[BaseMessage], 
    project_name: Optional[str] = None,
    runtime_duration: Optional[str] = None,
    file_stability_check_minutes: int = 1
) -> QCState:
    """
    Create a properly initialized QCState for the QC pipeline.
    
    Args:
        messages: List of messages from user
        project_name: Optional project name (will be extracted from messages if not provided)
        runtime_duration: Optional duration string (will be extracted from messages if not provided)
        file_stability_check_minutes: Minutes to wait for file stability checking
    
    Returns:
        Properly initialized QCState
    """
    if not project_name:
        project_name = extract_project_from_messages(messages)
    
    if not runtime_duration:
        runtime_duration = extract_duration_from_messages(messages)
    
    return {
        'messages': messages,
        'converted_files': None,
        'parsing_result': None,
        'agent_state': {},
        'project_name': project_name,
        'runtime_duration': runtime_duration,
        'file_stability_check_minutes': file_stability_check_minutes,
        'monitoring_start_time': None,
        'monitoring_end_time': None,
        'loop_iteration': None,
        'files_processed': None,
        'files_skipped': None,
        'errors_encountered': None,
    }

def should_use_continuous_monitoring(state: QCState) -> bool:
    """
    Determine if continuous monitoring should be used based on state.
    
    Returns True if runtime_duration is specified, False otherwise.
    """
    return state.get('runtime_duration') is not None

# ----------------------------------------------------------------------------
# Graph Construction
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    
    # Example usage with continuous monitoring
    test_messages = [
        # Short run - 1 hour
        {'type': 'human', 'content': 'run QC for project Solvent01 for 1h'},
        # Long run - 2 days  
        # {'type': 'human', 'content': 'monitor QC for project Solvent01 for 2d'},
        # Single execution (legacy)
        # {'type': 'human', 'content': 'run QC for project Solvent01'},
    ]
    
    # Create initial state using utility function
    initial_state = create_qc_state(test_messages)
    
    project_name = initial_state['project_name']
    duration_str = initial_state['runtime_duration']
    
    print(f"🚀 Starting QC pipeline for project: {project_name}")
    
    if should_use_continuous_monitoring(initial_state):
        print(f"⏱️ Monitoring duration: {duration_str}")
        
        # Run continuous monitoring directly
        result = asyncio.run(continuous_qc_monitoring(project_name, duration_str))
        
        print("\n🎉 QC Continuous Monitoring Complete!")
        print("=" * 60)
        print(f"📊 Project: {project_name}")
        print(f"⏱️ Duration: {duration_str}")
        print(f"🔄 Checks performed: {result.get('checks_performed')}")
        print(f"✅ Files processed: {result.get('files_processed')}")
        print(f"⏰ Total runtime: {format_duration(result.get('total_runtime', 0))}")
        
        if result.get('success'):
            print(f"\n📋 Monitoring completed successfully!")
        else:
            print(f"\n❌ Monitoring encountered issues")
    else:
        print("🔄 Running single QC execution")
        
        # Run single execution
        result = asyncio.run(convert_and_parse_node(initial_state, {}))
        
        print("\n🎉 QC Pipeline Complete!")
        print("=" * 50)
        print(f"📊 Project processed: {result.get('project_name')}")
        if result.get('move_wiff_pairs_result') is not None:
            print(f"📁 [move_wiff_pairs] Files moved: {result['move_wiff_pairs_result']}")
        if result.get('converted_files') is not None:
            print(f"🔄 [Conversion] Files moved: {result['converted_files']}")
        if result.get('parsing_result'):
            print(f"📊 [Parsing/QC] {result['parsing_result']}")
    
    # Show log file location
    if result.get('project_name'):
        dirs = get_directories(result['project_name'])
        log_file = dirs['log_dir'] / f"QC_{result['project_name']}.log"
        print(f"\n📝 See log file for detailed output: {log_file}")

# Build the graph with both single and continuous execution options
builder = StateGraph(QCState)
builder.add_node('convert_and_parse', convert_and_parse_node)
builder.add_node('continuous_monitoring', continuous_qc_monitoring_node)
builder.add_node('end', end_node)

# Set entry point based on whether duration is specified
builder.set_entry_point('convert_and_parse')  # Default entry point
builder.add_edge('convert_and_parse', 'end')
builder.add_edge('continuous_monitoring', 'end')

graph = builder.compile()
graph.name = 'QC Convert and Parse Agent'

# Alternative graph for continuous monitoring
continuous_builder = StateGraph(QCState)
continuous_builder.add_node('continuous_monitoring', continuous_qc_monitoring_node)
continuous_builder.add_node('end', end_node)
continuous_builder.set_entry_point('continuous_monitoring')
continuous_builder.add_edge('continuous_monitoring', 'end')

continuous_graph = continuous_builder.compile()
continuous_graph.name = 'QC Continuous Monitoring Agent'