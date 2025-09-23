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

# Import QC Worklist Generator - Conditional Import Solution
try:
    # Try relative import first (works in package context)
    from .qc_worklist_generator import generate_worklist_for_project
except ImportError:
    try:
        # Try absolute import (works when executed standalone)
        from qc_worklist_generator import generate_worklist_for_project
    except ImportError:
        # Fallback: add current directory to path and import
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        from qc_worklist_generator import generate_worklist_for_project

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
    Parse duration string into total seconds.
    Supports formats: "30s", "1h", "2h", "30m", "1d", "2d"
    Returns total seconds as integer.
    """
    if not duration_str:
        raise ValueError("Duration string cannot be empty")
    
    duration_str = duration_str.strip().lower()
    
    # Extract number and unit
    import re
    match = re.match(r'^(\d+)([shmd])$', duration_str)
    if not match:
        raise ValueError(f"Invalid duration format: '{duration_str}'. Use formats like '30s', '1h', '30m', '2d'")
    
    number = int(match.group(1))
    unit = match.group(2)
    
    if unit == 's':  # seconds
        return number
    elif unit == 'm':  # minutes
        return number * 60
    elif unit == 'h':  # hours
        return number * 3600
    elif unit == 'd':  # days
        return number * 86400
    else:
        raise ValueError(f"Unsupported time unit: '{unit}'. Use 's' (seconds), 'm' (minutes), 'h' (hours), or 'd' (days)")

def extract_duration_from_messages(messages: List[BaseMessage]) -> Optional[str]:
    """
    Extract runtime duration from user messages.
    Supports patterns like: "for 1h", "for 2d", "1h", "2d"
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
    
    # Try patterns for duration extraction
    patterns = [
        r'for\s+(\d+[hmd])',  # "for 1h", "for 2d"
        r'(\d+[hmd])\s*$',    # "1h", "2d" at end
        r'\s(\d+[hmd])\s',    # " 1h " in middle
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    return None

def extract_interval_and_duration_from_messages(messages: List[BaseMessage]) -> tuple[Optional[str], Optional[str]]:
    """
    Extract both interval and duration from user messages for interval-based monitoring.
    
    Supports patterns like:
    - "run QC for project Solvent01 every 1 minute for 10 minutes"
    - "monitor Solvent01 QC every 30 seconds for 2 hours"
    - "check QC for project Test01 every 2m for 1h"
    - "QC monitoring every 1 minute for 10 minutes" (NEW)
    
    Returns:
        tuple: (interval_str, duration_str) or (None, None) if not found
    """
    if not messages:
        return None, None
    
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
        return None, None
    
    # Pattern for "every X for Y" format
    interval_duration_patterns = [
        r'every\s+(\d+)\s+(?:minute|minutes|min|m)\s+for\s+(\d+)\s+(?:minute|minutes|min|m)',  # every 1 minute for 10 minutes
        r'every\s+(\d+)\s+(?:second|seconds|sec|s)\s+for\s+(\d+)\s+(?:minute|minutes|min|m)',  # every 30 seconds for 10 minutes
        r'every\s+(\d+)\s+(?:second|seconds|sec|s)\s+for\s+(\d+)\s+(?:hour|hours|hr|h)',      # every 30 seconds for 2 hours
        r'every\s+(\d+)\s+(?:minute|minutes|min|m)\s+for\s+(\d+)\s+(?:hour|hours|hr|h)',      # every 2 minutes for 1 hour
        r'every\s+(\d+[sm])\s+for\s+(\d+[hm])',  # every 2m for 1h
        r'every\s+(\d+)\s*([sm])\s+for\s+(\d+)\s*([hm])',  # every 2 m for 1 h
        r'monitoring\s+every\s+(\d+)\s+(?:minute|minutes|min|m)\s+for\s+(\d+)\s+(?:minute|minutes|min|m)',  # QC monitoring every 1 minute for 10 minutes
    ]
    
    for pattern in interval_duration_patterns:
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 4:  # Pattern with separate number and unit
                interval_str = f"{groups[0]}{groups[1].lower()}"
                duration_str = f"{groups[2]}{groups[3].lower()}"
            elif len(groups) == 2:  # Pattern with combined number+unit
                if groups[0].isdigit():  # First pattern type
                    # Determine units from the pattern matched
                    if 'second' in match.group(0).lower():
                        interval_str = f"{groups[0]}s"
                    elif 'minute' in match.group(0).lower():
                        interval_str = f"{groups[0]}m"
                    else:
                        interval_str = f"{groups[0]}m"  # default to minutes
                    
                    if 'hour' in match.group(0).lower():
                        duration_str = f"{groups[1]}h"
                    else:
                        duration_str = f"{groups[1]}m"  # default to minutes
                else:  # Combined format like "2m"
                    interval_str = groups[0].lower()
                    duration_str = groups[1].lower()
            
            return interval_str, duration_str
    
    # If no interval pattern found, check for regular duration only
    duration_only = extract_duration_from_messages(messages)
    return None, duration_only

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

async def check_file_stability_tracked(file_path: Path, file_tracker: Dict, stability_seconds: int = 20) -> bool:
    """
    Check if file size has been stable for specified duration using tracking dictionary.
    
    Args:
        file_path: Path to the file to check
        file_tracker: Dictionary to track file sizes and timestamps
        stability_seconds: Seconds to wait for stability (default: 20)
    
    Returns:
        True if file is stable, False if still growing or not enough time passed
    """
    import time
    
    try:
        if not file_path.exists():
            return False
        
        current_size = file_path.stat().st_size
        current_time = time.time()
        file_key = str(file_path)
        
        if file_key not in file_tracker:
            # First time seeing this file
            file_tracker[file_key] = {
                'size': current_size, 
                'timestamp': current_time,
                'checks': 1
            }
            return False  # Wait for stability
        
        last_size = file_tracker[file_key]['size']
        last_time = file_tracker[file_key]['timestamp']
        checks = file_tracker[file_key].get('checks', 1)
        
        if current_size != last_size:
            # File has grown, update tracker
            file_tracker[file_key] = {
                'size': current_size, 
                'timestamp': current_time,
                'checks': checks + 1
            }
            return False  # File is still growing
        
        # File size unchanged, check if enough time has passed
        time_elapsed = current_time - last_time
        if time_elapsed >= stability_seconds:
            # File is stable, update check count
            file_tracker[file_key]['checks'] = checks + 1
            return True
        
        # Not enough time has passed, update check count
        file_tracker[file_key]['checks'] = checks + 1
        return False
        
    except Exception as e:
        logger = logging.getLogger()
        logger.error(f"Error checking file stability for {file_path}: {e}")
        return False

async def get_stable_files(all_files: List[Path], file_tracker: Dict, stability_seconds: int = 20) -> tuple[List[Path], List[Path]]:
    """
    Separate files into stable and unstable lists.
    
    Args:
        all_files: List of all files found
        file_tracker: Dictionary to track file stability
        stability_seconds: Seconds required for stability
    
    Returns:
        tuple: (stable_files, unstable_files)
    """
    stable_files = []
    unstable_files = []
    
    for file_path in all_files:
        is_stable = await check_file_stability_tracked(file_path, file_tracker, stability_seconds)
        if is_stable:
            stable_files.append(file_path)
        else:
            unstable_files.append(file_path)
    
    return stable_files, unstable_files

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
        if secs > 0:
            parts.append(f"{secs}s")
    elif hours > 0:
        parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0:
            parts.append(f"{secs}s")
    elif minutes > 0:
        parts.append(f"{minutes}m")
        if secs > 0:
            parts.append(f"{secs}s")
    else:
        parts.append(f"{secs}s")
    
    return " ".join(parts)

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
    Extract project from messages. If not found, return None instead of prompting.
    This version is compatible with LangGraph server environment.
    """
    project = extract_project_from_messages(messages)
    if project:
        # Project successfully extracted from messages
        return project
    
    # Instead of prompting with input(), return None and let the calling function handle it
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
    wiff_source_dir = Path("/mnt/c/Users/ChopraLab/Desktop/laptop/convert_raw/raw_data")
    return {
        'qc_text_target': qc_text_base_dir / project_name,
        'qc_csv_target': qc_csv_base_dir / project_name,
        'log_dir': log_base_dir / project_name,
        'results_dir': results_base_dir / project_name,
        'wsl_target': qc_text_base_dir / project_name,
        'wiff': wiff_source_dir,  # Source directory for WIFF files
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
    # New interval monitoring parameters
    loop_interval_seconds: Optional[int]  # How often to check (e.g., 60 for 1 minute)
    total_duration_seconds: Optional[int]  # How long to run total (e.g., 600 for 10 minutes)
    processed_files: Optional[set]  # Track files already processed
    file_size_tracker: Optional[Dict[str, Dict[str, Any]]]  # Track file sizes and timestamps
    stability_check_interval: Optional[int]  # Check file size every N seconds (default 20)

async def continuous_qc_monitoring_node(state: QCState, config: RunnableConfig) -> QCState:
    """
    Enhanced QC monitoring node with interval-based processing.
    Supports both traditional duration-based monitoring and new interval-based monitoring.
    """
    import time
    import signal
    import traceback
    
    # Extract parameters from state and messages
    messages = state.get('messages', []) if isinstance(state, dict) else []
    project_name = get_dynamic_project_str(messages)
    
    if not project_name:
        error_msg = (
            "No project name found in message. Please include project name in your request. "
            "Examples: 'monitor QC every 1 minute for 10 minutes for project Solvent01' or 'run QC for project TestProject01 for 2h'"
        )
        return {
            **state,
            'parsing_result': error_msg,
            'messages': state['messages'] + [
                AIMessage(content=error_msg)
            ]
        }
    
    # Check for interval-based monitoring first
    interval_str, duration_str = extract_interval_and_duration_from_messages(messages)
    
    # If no interval found, fall back to traditional duration monitoring
    if not interval_str:
        duration_str = duration_str or extract_duration_from_messages(messages) or state.get('runtime_duration', '1h')
        return await traditional_continuous_monitoring(state, config, project_name, duration_str)
    
    # Setup logging
    logger = await setup_logging(project_name)
    
    # Parse interval and duration
    try:
        interval_seconds = parse_duration(interval_str)
        total_seconds = parse_duration(duration_str)
    except ValueError as e:
        raise ValueError(f"Invalid interval/duration format: {e}")
    
    # SPECIAL CASE: 1-minute intervals for 10 minutes - use comprehensive monitoring
    if interval_seconds == 60 and total_seconds == 600:
        logger.info("🎯 Detected request for comprehensive 10-minute monitoring with 1-minute intervals")
        logger.info("🚀 Switching to comprehensive monitoring function with detailed logging")
        
        # Call the comprehensive monitoring function
        monitoring_result = await run_qc_monitoring_every_minute_for_10_minutes(project_name)
        
        # Convert results to QCState format and return
        if monitoring_result['success']:
            success_msg = f"✅ Completed comprehensive QC monitoring: {monitoring_result.get('files_processed_total', 0)} files processed, {monitoring_result.get('completed_loops', 0)} loops completed"
            
            return {
                **state,
                'project_name': project_name,
                'runtime_duration': duration_str,
                'loop_interval_seconds': interval_seconds,
                'total_duration_seconds': total_seconds,
                'monitoring_start_time': time.time() - monitoring_result.get('total_duration_seconds', 600),
                'monitoring_end_time': time.time(),
                'loop_iteration': monitoring_result.get('completed_loops', 0),
                'files_processed': monitoring_result.get('files_processed_total', 0),
                'files_skipped': monitoring_result.get('files_skipped_total', 0),
                'errors_encountered': monitoring_result.get('errors_encountered', 0),
                'parsing_result': success_msg,
                'messages': state['messages'] + [
                    AIMessage(content=f"🎯 Started comprehensive QC monitoring: {project_name}, every 1 minute for 10 minutes"),
                    AIMessage(content=success_msg),
                    AIMessage(content=f"📊 Final Stats: {monitoring_result.get('success_rate', 0):.1f}% success rate, {monitoring_result.get('unique_files_tracked', 0)} unique files tracked")
                ]
            }
        else:
            error_msg = f"❌ Comprehensive QC monitoring failed: {monitoring_result.get('error', 'Unknown error')}"
            
            return {
                **state,
                'project_name': project_name,
                'runtime_duration': duration_str,
                'parsing_result': error_msg,
                'errors_encountered': monitoring_result.get('errors_encountered', 1),
                'messages': state['messages'] + [
                    AIMessage(content=f"🎯 Started comprehensive QC monitoring: {project_name}, every 1 minute for 10 minutes"),
                    AIMessage(content=error_msg)
                ]
            }
    
    # Initialize monitoring state
    start_time = time.time()
    end_time = start_time + total_seconds
    loop_iteration = 0
    files_processed_total = 0
    files_skipped_total = 0
    errors_encountered = 0
    
    # Initialize tracking structures
    processed_files = state.get('processed_files', set()) or set()
    file_size_tracker = state.get('file_size_tracker', {}) or {}
    stability_check_interval = state.get('stability_check_interval', 20)
    
    logger.info(f"🚀 Starting QC monitoring: {project_name}, every {interval_str} for {duration_str}")
    logger.info(f"📅 Monitoring will run until: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"⏱️ Check interval: {interval_seconds}s, File stability: {stability_check_interval}s")
    
    # Graceful shutdown handler
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.info(f"🛑 Shutdown signal received (signal {signum})")
        shutdown_requested = True
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    monitoring_results = []
    
    try:
        # Main interval-based monitoring loop
        while time.time() < end_time and not shutdown_requested:
            loop_iteration += 1
            current_time = time.time()
            remaining_seconds = int(end_time - current_time)
            time_remaining = format_time_remaining(remaining_seconds)
            
            logger.info(f"🔄 Loop {loop_iteration}, {time_remaining} remaining")
            
            try:
                # Find all potential files for this project
                dirs = get_directories(project_name)
                wiff_source_dir = dirs['wiff']
                
                # Get all WIFF files for this project from the raw data directory
                all_wiff_files = []
                if await asyncio.to_thread(wiff_source_dir.exists):
                    all_wiff_patterns = await asyncio.to_thread(lambda: list(wiff_source_dir.glob("*.wiff")))
                    for wiff_file in all_wiff_patterns:
                        if f"Proj-{project_name}" in wiff_file.name:
                            all_wiff_files.append(wiff_file)
                
                logger.info(f"🔍 Found {len(all_wiff_files)} total files for project {project_name}")
                
                if all_wiff_files:
                    # Check file stability
                    stable_files, unstable_files = await get_stable_files(
                        all_wiff_files, file_size_tracker, stability_check_interval
                    )
                    
                    # Filter out already processed files
                    new_stable_files = [f for f in stable_files if str(f) not in processed_files]
                    already_processed = [f for f in stable_files if str(f) in processed_files]
                    
                    logger.info(f"📊 Files status: {len(stable_files)} stable, {len(unstable_files)} still growing")
                    logger.info(f"📊 Processing status: {len(new_stable_files)} new, {len(already_processed)} already done")
                    
                    # Log stability details
                    if unstable_files:
                        logger.info(f"⏳ Files still growing: {[f.name for f in unstable_files[:3]]}")
                    if already_processed:
                        logger.info(f"✅ Files already processed: {[f.name for f in already_processed[:3]]}")
                    
                    # Process new stable files
                    if new_stable_files:
                        logger.info(f"🔄 Processing {len(new_stable_files)} new stable files...")
                        
                        # Run the complete QC pipeline for new files
                        result = await run_single_qc_check(project_name, logger, loop_iteration)
                        
                        if result['success']:
                            files_processed_this_loop = result.get('files_processed', len(new_stable_files))
                            files_processed_total += files_processed_this_loop
                            
                            # Add processed files to tracking set
                            for file_path in new_stable_files:
                                processed_files.add(str(file_path))
                            
                            logger.info(f"✅ Loop {loop_iteration}: Processed {files_processed_this_loop} new files")
                            monitoring_results.append(f"Loop {loop_iteration}: Processed {files_processed_this_loop} new files")
                        else:
                            errors_encountered += 1
                            logger.error(f"❌ Loop {loop_iteration}: Error - {result['error']}")
                            monitoring_results.append(f"Loop {loop_iteration}: Error - {result['error']}")
                    else:
                        logger.info(f"📭 Loop {loop_iteration}: No new files to process")
                        monitoring_results.append(f"Loop {loop_iteration}: No new files")
                    
                    files_skipped_total += len(unstable_files)
                    
                else:
                    logger.info(f"📭 Loop {loop_iteration}: No files found for project {project_name}")
                    monitoring_results.append(f"Loop {loop_iteration}: No files found")
                
            except Exception as e:
                errors_encountered += 1
                logger.error(f"❌ Error in monitoring loop {loop_iteration}: {e}")
                logger.error(f"🔍 Traceback: {traceback.format_exc()}")
                monitoring_results.append(f"Loop {loop_iteration}: Exception - {str(e)}")
            
            # Status reporting every hour for long runs
            total_loops = int(total_seconds / interval_seconds)
            if total_loops > 10 and loop_iteration % max(1, total_loops // 10) == 0:
                logger.info(f"📊 Status Report - Files processed: {files_processed_total}, Files skipped: {files_skipped_total}, Errors: {errors_encountered}")
            
            # Sleep for the specified interval
            if time.time() < end_time and not shutdown_requested:
                sleep_time = min(interval_seconds, remaining_seconds)
                logger.debug(f"😴 Sleeping for {sleep_time} seconds before next check")
                await asyncio.sleep(sleep_time)
        
        # Final status
        total_time = time.time() - start_time
        if shutdown_requested:
            logger.info(f"🛑 QC monitoring stopped by user request after {format_time_remaining(int(total_time))}")
        else:
            logger.info(f"✅ QC monitoring completed after {duration_str}")
        
        logger.info(f"📊 Final Summary:")
        logger.info(f"   • Total loops: {loop_iteration}")
        logger.info(f"   • Files processed: {files_processed_total}")
        logger.info(f"   • Files skipped (unstable): {files_skipped_total}")
        logger.info(f"   • Errors encountered: {errors_encountered}")
        logger.info(f"   • Unique files tracked: {len(processed_files)}")
        
        # Clean up file tracker for processed files to prevent memory buildup
        processed_file_keys = [str(f) for f in processed_files]
        for key in list(file_size_tracker.keys()):
            if key in processed_file_keys:
                del file_size_tracker[key]
        
        # Update state with monitoring results
        updated_state = {
            **state,
            'project_name': project_name,
            'runtime_duration': duration_str,
            'loop_interval_seconds': interval_seconds,
            'total_duration_seconds': total_seconds,
            'processed_files': processed_files,
            'file_size_tracker': file_size_tracker,
            'stability_check_interval': stability_check_interval,
            'monitoring_start_time': start_time,
            'monitoring_end_time': time.time(),
            'loop_iteration': loop_iteration,
            'files_processed': files_processed_total,
            'files_skipped': files_skipped_total,
            'errors_encountered': errors_encountered,
            'parsing_result': '\n'.join(monitoring_results),
            'messages': state['messages'] + [
                AIMessage(content=f"🚀 Started interval QC monitoring: {project_name}, every {interval_str} for {duration_str}"),
                AIMessage(content=f"📊 Monitoring completed: {files_processed_total} files processed, {errors_encountered} errors"),
                AIMessage(content='\n'.join(monitoring_results[-5:]))  # Last 5 results
            ]
        }
        
        return updated_state
        
    except Exception as e:
        logger.error(f"❌ Fatal error in interval monitoring loop: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        
        return {
            **state,
            'project_name': project_name,
            'runtime_duration': duration_str,
            'parsing_result': f"Fatal error: {str(e)}",
            'errors_encountered': errors_encountered + 1,
            'messages': state['messages'] + [
                AIMessage(content=f"❌ QC monitoring failed: {str(e)}")
            ]
        }

async def traditional_continuous_monitoring(state: QCState, config: RunnableConfig, project_name: str, duration_str: str) -> QCState:
    """
    Traditional continuous monitoring (backward compatibility).
    This maintains the existing behavior for non-interval monitoring.
    """
    import time
    import signal
    import traceback
    
    # Validate project name (should already be validated, but double-check)
    if not project_name:
        error_msg = "Project name is required for traditional continuous monitoring"
        return {
            **state,
            'parsing_result': error_msg,
            'messages': state['messages'] + [
                AIMessage(content=error_msg)
            ]
        }
    
    file_stability_minutes = state.get('file_stability_check_minutes', 1)
    
    # Parse duration
    try:
        total_seconds = parse_duration(duration_str)
    except ValueError as e:
        raise ValueError(f"Invalid duration format: {e}")
    
    # Setup logging
    logger = await setup_logging(project_name)
    
    # Initialize monitoring state
    start_time = time.time()
    end_time = start_time + total_seconds
    loop_iteration = 0
    files_processed = 0
    files_skipped = 0
    errors_encountered = 0
    
    logger.info(f"🚀 Running QC monitoring for {duration_str} for project: {project_name}")
    logger.info(f"📅 Monitoring will run until: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Graceful shutdown handler
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.info(f"🛑 Shutdown signal received (signal {signum})")
        shutdown_requested = True
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    monitoring_results = []
    
    try:
        # Main monitoring loop (traditional 5-minute intervals)
        while time.time() < end_time and not shutdown_requested:
            loop_iteration += 1
            current_time = time.time()
            remaining_seconds = int(end_time - current_time)
            time_remaining = format_time_remaining(remaining_seconds)
            
            logger.info(f"🔄 Monitoring loop {loop_iteration}, {time_remaining} remaining")
            logger.info(f"🔍 Checking project: {project_name}")
            
            try:
                # Run the complete QC pipeline
                result = await run_single_qc_check(project_name, logger, loop_iteration)
                
                if result['success']:
                    files_processed_this_loop = result.get('files_processed', 0)
                    files_processed += files_processed_this_loop
                    logger.info(f"✅ Successfully processed {files_processed_this_loop} files")
                    monitoring_results.append(f"Loop {loop_iteration}: Processed {files_processed_this_loop} files")
                else:
                    errors_encountered += 1
                    logger.error(f"❌ Error processing files: {result['error']}")
                    monitoring_results.append(f"Loop {loop_iteration}: Error - {result['error']}")
                
            except Exception as e:
                errors_encountered += 1
                logger.error(f"❌ Error in monitoring loop {loop_iteration}: {e}")
                monitoring_results.append(f"Loop {loop_iteration}: Exception - {str(e)}")
            
            # Status reporting every hour for long runs
            if loop_iteration % 12 == 0:  # Every 12 iterations (assuming 5-min intervals = 1 hour)
                logger.info(f"📊 Status Report - Files processed: {files_processed}, Files skipped: {files_skipped}, Errors: {errors_encountered}")
            
            # Sleep between checks (5 minutes)
            if time.time() < end_time and not shutdown_requested:
                sleep_time = min(300, remaining_seconds)  # 5 minutes or remaining time
                logger.debug(f"😴 Sleeping for {sleep_time} seconds before next check")
                await asyncio.sleep(sleep_time)
        
        # Final status
        total_time = time.time() - start_time
        if shutdown_requested:
            logger.info(f"🛑 QC monitoring stopped by user request after {format_time_remaining(int(total_time))}")
        else:
            logger.info(f"✅ QC monitoring completed after {duration_str}")
        
        logger.info(f"📊 Final Summary - Files processed: {files_processed}, Files skipped: {files_skipped}, Errors: {errors_encountered}")
        
        # Update state with monitoring results
        updated_state = {
            **state,
            'project_name': project_name,
            'runtime_duration': duration_str,
            'file_stability_check_minutes': file_stability_minutes,
            'monitoring_start_time': start_time,
            'monitoring_end_time': time.time(),
            'loop_iteration': loop_iteration,
            'files_processed': files_processed,
            'files_skipped': files_skipped,
            'errors_encountered': errors_encountered,
            'parsing_result': '\n'.join(monitoring_results),
            'messages': state['messages'] + [
                AIMessage(content=f"🚀 Started QC monitoring for {duration_str} for project: {project_name}"),
                AIMessage(content=f"📊 Monitoring completed: {files_processed} files processed, {errors_encountered} errors"),
                AIMessage(content='\n'.join(monitoring_results))
            ]
        }
        
        return updated_state
        
    except Exception as e:
        logger.error(f"❌ Fatal error in monitoring loop: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        
        return {
            **state,
            'project_name': project_name,
            'runtime_duration': duration_str,
            'parsing_result': f"Fatal error: {str(e)}",
            'errors_encountered': errors_encountered + 1,
            'messages': state['messages'] + [
                AIMessage(content=f"❌ QC monitoring failed: {str(e)}")
            ]
        }

async def run_qc_monitoring_every_minute_for_10_minutes(project_name: str) -> Dict[str, Any]:
    """
    QC Monitoring that runs every 1 minute for exactly 10 minutes.
    Provides comprehensive logging of all activities, file processing status, and detailed timestamps.
    
    Args:
        project_name: Name of the project to monitor (e.g., "Solvent01")
    
    Returns:
        Dictionary with monitoring results and statistics
    """
    import time
    import signal
    import traceback
    from datetime import datetime
    
    # Initialize monitoring parameters
    interval_seconds = 60  # 1 minute
    total_duration_seconds = 600  # 10 minutes
    start_time = time.time()
    end_time = start_time + total_duration_seconds
    loop_iteration = 0
    files_processed_total = 0
    files_skipped_total = 0
    errors_encountered = 0
    processed_files = set()  # Avoid reprocessing same files
    file_size_tracker = {}  # Track file stability
    
    # Setup logging
    logger = await setup_logging(project_name)
    
    # Session Start Logging
    start_datetime = datetime.fromtimestamp(start_time)
    end_datetime = datetime.fromtimestamp(end_time)
    
    logger.info("=" * 80)
    logger.info("QC MONITORING SESSION STARTED")
    logger.info("=" * 80)
    logger.info(f"Project: {project_name}")
    logger.info(f"Interval: Every 60 seconds (1 minute)")
    logger.info(f"Total Duration: 600 seconds (10 minutes)")
    logger.info(f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"File Stability Check: 20 seconds")
    logger.info("=" * 80)
    
    # Graceful shutdown handler
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.info(f"🛑 Shutdown signal received (signal {signum})")
        shutdown_requested = True
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    monitoring_results = []
    
    try:
        # Main monitoring loop - runs every 1 minute for 10 minutes
        while time.time() < end_time and not shutdown_requested:
            loop_start_time = time.time()
            loop_iteration += 1
            current_time = time.time()
            remaining_seconds = int(end_time - current_time)
            time_remaining_formatted = f"{remaining_seconds // 60:02d}:{remaining_seconds % 60:02d}"
            
            # Loop iteration logging
            logger.info("-" * 60)
            logger.info(f"LOOP {loop_iteration} STARTING")
            logger.info(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"Remaining: {remaining_seconds} seconds ({time_remaining_formatted})")
            logger.info("-" * 60)
            
            try:
                # Step 1: Scanning for WIFF files
                logger.info("Step 1: Scanning for WIFF files...")
                dirs = get_directories(project_name)
                
                # Find all WIFF files in the source directory
                wiff_source_dir = Path("/mnt/c/Users/ChopraLab/Desktop/laptop/convert_raw/raw_data")
                all_wiff_files = []
                total_wiff_files = 0
                
                if await asyncio.to_thread(wiff_source_dir.exists):
                    all_wiff_patterns = await asyncio.to_thread(lambda: list(wiff_source_dir.glob("*.wiff")))
                    total_wiff_files = len(all_wiff_patterns)
                    
                    # Filter for project-specific files
                    for wiff_file in all_wiff_patterns:
                        if f"Proj-{project_name}" in wiff_file.name:
                            all_wiff_files.append(wiff_file)
                
                logger.info(f"Found {total_wiff_files} total WIFF files")
                logger.info(f"Found {len(all_wiff_files)} WIFF files for project {project_name}")
                
                # Step 2: Checking file stability
                logger.info("Step 2: Checking file stability...")
                stable_files, unstable_files = await get_stable_files(
                    all_wiff_files, file_size_tracker, stability_seconds=20
                )
                
                # Filter out already processed files
                new_stable_files = [f for f in stable_files if str(f) not in processed_files]
                already_processed_files = [f for f in stable_files if str(f) in processed_files]
                
                # File Status Summary
                logger.info("File Status Summary:")
                logger.info(f"  - Total files found: {len(all_wiff_files)}")
                logger.info(f"  - Stable files: {len(stable_files)}")
                logger.info(f"  - Unstable files: {len(unstable_files)}")
                logger.info(f"  - New stable files: {len(new_stable_files)}")
                logger.info(f"  - Already processed: {len(already_processed_files)}")
                
                # Step 3: Processing files (if any new stable files)
                if new_stable_files:
                    logger.info(f"Step 3: Processing {len(new_stable_files)} new stable files...")
                    
                    for file_path in new_stable_files:
                        logger.info(f"Processing: {file_path.name}")
                    
                    # Run the complete QC pipeline for new files
                    result = await run_single_qc_check(project_name, logger, loop_iteration)
                    
                    if result['success']:
                        files_processed_this_loop = result.get('files_processed', len(new_stable_files))
                        files_processed_total += files_processed_this_loop
                        
                        # Add processed files to tracking set
                        for file_path in new_stable_files:
                            processed_files.add(str(file_path))
                            logger.info(f"  ✓ Successfully processed: {file_path.name}")
                        
                        monitoring_results.append(f"Loop {loop_iteration}: Processed {files_processed_this_loop} files")
                    else:
                        errors_encountered += 1
                        logger.error(f"❌ Error processing files: {result['error']}")
                        monitoring_results.append(f"Loop {loop_iteration}: Error - {result['error']}")
                else:
                    logger.info("Step 3: No new stable files to process")
                    monitoring_results.append(f"Loop {loop_iteration}: No new files")
                
                # Update skipped files count
                files_skipped_total += len(unstable_files)
                
                # Loop completion logging
                loop_duration = int(time.time() - loop_start_time)
                logger.info(f"LOOP {loop_iteration} COMPLETED")
                logger.info(f"Loop Duration: {loop_duration} seconds")
                logger.info(f"Files Processed This Loop: {len(new_stable_files) if new_stable_files else 0}")
                logger.info("Cumulative Stats:")
                logger.info(f"  - Total Files Processed: {files_processed_total}")
                logger.info(f"  - Total Files Skipped: {files_skipped_total}")
                logger.info(f"  - Total Errors: {errors_encountered}")
                logger.info(f"  - Unique Files Tracked: {len(processed_files)}")
                
            except Exception as e:
                errors_encountered += 1
                logger.error(f"❌ Error in monitoring loop {loop_iteration}: {e}")
                logger.error(f"🔍 Traceback: {traceback.format_exc()}")
                monitoring_results.append(f"Loop {loop_iteration}: Exception - {str(e)}")
            
            # Sleep until next iteration (exactly 60 seconds from loop start)
            if time.time() < end_time and not shutdown_requested:
                elapsed_time = time.time() - loop_start_time
                sleep_time = max(0, interval_seconds - elapsed_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        # Session end logging
        end_time_actual = time.time()
        total_monitoring_time = int(end_time_actual - start_time)
        total_minutes = total_monitoring_time // 60
        success_rate = (files_processed_total / max(1, files_processed_total + errors_encountered)) * 100
        
        logger.info("=" * 80)
        logger.info("QC MONITORING SESSION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Project: {project_name}")
        logger.info(f"End Time: {datetime.fromtimestamp(end_time_actual).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total Monitoring Time: {total_monitoring_time} seconds ({total_minutes} minutes)")
        logger.info(f"Completed Loops: {loop_iteration}")
        logger.info("Final Statistics:")
        logger.info(f"  - Total Files Processed: {files_processed_total}")
        logger.info(f"  - Total Files Skipped (unstable): {files_skipped_total}")
        logger.info(f"  - Total Errors Encountered: {errors_encountered}")
        logger.info(f"  - Unique Files Tracked: {len(processed_files)}")
        logger.info(f"  - Success Rate: {success_rate:.1f}%")
        logger.info("=" * 80)
        
        if shutdown_requested:
            logger.info(f"🛑 Session was interrupted by user request")
        
        return {
            'success': True,
            'project_name': project_name,
            'total_duration_seconds': total_monitoring_time,
            'interval_seconds': interval_seconds,
            'completed_loops': loop_iteration,
            'files_processed_total': files_processed_total,
            'files_skipped_total': files_skipped_total,
            'errors_encountered': errors_encountered,
            'unique_files_tracked': len(processed_files),
            'success_rate': success_rate,
            'monitoring_results': monitoring_results,
            'shutdown_requested': shutdown_requested
        }
        
    except Exception as e:
        logger.error(f"❌ Fatal error in QC monitoring session: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        
        return {
            'success': False,
            'project_name': project_name,
            'error': str(e),
            'completed_loops': loop_iteration,
            'files_processed_total': files_processed_total,
            'files_skipped_total': files_skipped_total,
            'errors_encountered': errors_encountered + 1
        }

async def run_qc_monitoring_session(project_name: str) -> Dict[str, Any]:
    """
    Main entry point for QC monitoring session.
    Convenience function that wraps the main monitoring function.
    
    Args:
        project_name: Name of the project to monitor
    
    Returns:
        Dictionary with monitoring results
    """
    if not project_name:
        raise ValueError("Project name is required for QC monitoring")
    
    return await run_qc_monitoring_every_minute_for_10_minutes(project_name)

async def run_single_qc_check(project_name: str, logger, loop_iteration: int) -> Dict[str, Any]:
    """
    Run a complete QC check with full 5-step workflow for interval monitoring.
    Based on the memory requirements for proper QC pipeline integration.
    
    Steps:
    1. Check for new WIFF files (move_wiff_pairs)
    2. Convert WIFF to TXT (Convert class)
    3. Parse TXT to CSV (QTRAP_Parse class)
    4. Generate QC results (qc_results function)
    5. Move files based on pass/fail (qc_validated_move function)
    """
    try:
        logger.info(f"=== QC CHECK #{loop_iteration} STARTING ===")
        
        files_processed = 0
        
        # Step 1: Check for new WIFF files
        logger.info(f"Step 1: Checking for new WIFF files...")
        move_files, move_debug = await move_wiff_pairs(project_name=project_name)
        
        if move_files:
            logger.info(f"✅ Step 1: Found {len(move_files)} new WIFF file pairs")
            files_processed = len(move_files)
        else:
            logger.info(f"📭 Step 1: No new WIFF files found for project {project_name}")
            logger.info(f"=== QC CHECK #{loop_iteration} COMPLETED: 0 files processed ===")
            return {'success': True, 'files_processed': 0}
        
        # Step 2: Convert WIFF to TXT
        logger.info(f"Step 2: Converting WIFF to TXT...")
        dirs = get_directories(project_name)
        converter = Convert(project_name)
        
        await converter.setup_directories()
        await converter.run_conversion()
        moved = await converter.move_converted()
        logger.info(f"✅ Step 2: Conversion successful. Files moved: {moved}")
        
        # Step 3: Parse TXT to CSV
        logger.info(f"Step 3: Parsing TXT to CSV...")
        await asyncio.to_thread(lambda: dirs['qc_csv_target'].mkdir(parents=True, exist_ok=True))
        
        all_files = await asyncio.to_thread(lambda: list(dirs['qc_text_target'].glob("*.txt")))
        logger.info(f"🔍 Found {len(all_files)} total .txt files in {dirs['qc_text_target']}")
        
        if project_name:
            files = validate_project_files(all_files, project_name)
            logger.info(f"🔬 After project filter 'Proj-{project_name}': {len(files)} files")
        else:
            files = all_files
            logger.warning(f"⚠️ No project filtering applied - processing all {len(files)} files")
        
        if not files:
            logger.warning(f"⚠️ No .txt files found for project {project_name} in {dirs['qc_text_target']}")
            logger.info(f"=== QC CHECK #{loop_iteration} COMPLETED: 0 files processed ===")
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
        
        logger.info(f"✅ Step 3: Parsed {parsed_count} files to CSV")
        
        # Step 4: Generate QC results
        logger.info(f"Step 4: Generating QC results...")
        await qc_results(project_name)
        logger.info(f"✅ Step 4: QC results generated for {project_name}")
        
        # Generate worklist for failed QC results
        try:
            worklist_success = generate_worklist_for_project(project_name)
            if worklist_success:
                logger.info(f"✅ Worklist generated for failed QC results")
            else:
                logger.info(f"ℹ️ No worklist needed (no failed QC results)")
        except Exception as e:
            logger.error(f"❌ Error generating worklist: {e}")
        
        # Step 5: Move files based on pass/fail
        logger.info(f"Step 5: Moving validated files...")
        await qc_validated_move(project_name)
        logger.info(f"✅ Step 5: QC validated files moved to production and fail directories")
        
        logger.info(f"=== QC CHECK #{loop_iteration} COMPLETED: {files_processed} files processed ===")
        
        return {'success': True, 'files_processed': files_processed}
        
    except Exception as e:
        logger.error(f"❌ Error in QC check #{loop_iteration}: {e}")
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
        logger.info(f"=== QC CHECK #{loop_iteration} COMPLETED: ERROR ===")
        return {'success': False, 'error': str(e), 'files_processed': 0}

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
        
        # Generate worklist for failed QC results
        try:
            worklist_success = generate_worklist_for_project(project_name)
            if worklist_success:
                logger.info(f"✅ Worklist generated for failed QC results")
            else:
                logger.info(f"ℹ️ No worklist needed (no failed QC results)")
        except Exception as e:
            logger.error(f"❌ Error generating worklist: {e}")
        
        # Move validated files
        await qc_validated_move(project_name)
        logger.info(f"✅ QC validated files moved to production and fail directories")
        
        return {'success': True, 'files_processed': parsed_count}
        
    except Exception as e:
        logger.error(f"❌ Error in QC iteration: {e}")
        return {'success': False, 'error': str(e)}

async def convert_and_parse_node(state: QCState, config: RunnableConfig) -> QCState:
    # Extract project name (required)
    messages = state.get('messages', []) if isinstance(state, dict) else []
    project_name = get_dynamic_project_str(messages)
    
    if not project_name:
        error_msg = (
            "No project name found in message. Please include project name in your request. "
            "Examples: 'run QC for project Solvent01' or 'process project TestProject01'"
        )
        return {
            **state,
            'parsing_result': error_msg,
            'messages': state['messages'] + [
                AIMessage(content=error_msg)
            ]
        }
    
    # Use project_name for all directory operations and logging
    logger = await setup_logging(project_name)
    logger.info(f"🚀 Starting QC pipeline for project: {project_name}")
    date_msg = f"📅 Processing QC for project: {project_name}"
    logger.info("🔧 Running file flow diagnostics...")
    diagnostic_info = await diagnose_file_flow(project_name)
    move_files, move_debug = await move_wiff_pairs(project_name=project_name, check_stability=False)
    move_summary = f"✅ move_wiff_pairs: {len(move_files)} file pairs moved." if move_files else f"❌ move_wiff_pairs: No file pairs moved for project {project_name}."
    logger.info(move_summary)
    logger.debug("\n".join(move_debug))
    dirs = get_directories(project_name)
    converter = Convert(project_name)
    try:
        await converter.setup_directories()
        await converter.run_conversion()
        moved = await converter.move_converted()
        msg1 = f"✅ Conversion successful. Files moved: {moved}"
        logger.info(msg1)
        await asyncio.to_thread(lambda: dirs['qc_csv_target'].mkdir(parents=True, exist_ok=True))
        all_files = await asyncio.to_thread(lambda: list(dirs['qc_text_target'].glob("*.txt")))
        logger.info(f"🔍 Found {len(all_files)} total .txt files in {dirs['qc_text_target']}")
        
        if project_name:
            files = validate_project_files(all_files, project_name)
            logger.info(f"🔬 After project filter 'Proj-{project_name}': {len(files)} files")
            
            # Debug logging for each file that will be parsed
            for f in files:
                logger.info(f"📄 Will parse: {f.name}")
            
            if not files:
                logger.error(f"❌ No .txt files found for project {project_name} in {dirs['qc_text_target']}")
                # Show what files ARE available
                if all_files:
                    logger.error(f"📋 Available files (first 5): {[f.name for f in all_files[:5]]}")
        else:
            files = all_files
            logger.warning(f"⚠️ No project filtering applied - processing all {len(files)} files")
            
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
            await qc_results(project_name)
            qc_msg = f"✅ QC results generated for {project_name}"
            logger.info(qc_msg)
            summary += f"\n{qc_msg}"
            
            # Generate worklist for failed QC results
            try:
                worklist_success = generate_worklist_for_project(project_name)
                if worklist_success:
                    worklist_msg = f"✅ Worklist generated for failed QC results"
                    logger.info(worklist_msg)
                    summary += f"\n{worklist_msg}"
                else:
                    worklist_msg = f"ℹ️ No worklist needed (no failed QC results)"
                    logger.info(worklist_msg)
                    summary += f"\n{worklist_msg}"
            except Exception as e:
                worklist_msg = f"❌ Error generating worklist: {e}"
                logger.error(worklist_msg)
                summary += f"\n{worklist_msg}"
            try:
                await qc_validated_move(project_name)
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
            'project_name': project_name,
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
            'project_name': project_name,
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
    
    # Example usage with different monitoring types
    test_messages = [
        # NEW: 1-minute interval monitoring for 10 minutes
        {'type': 'human', 'content': 'run QC monitoring every 1 minute for 10 minutes for project Solvent01'},
        # Traditional continuous monitoring - 1 hour
        # {'type': 'human', 'content': 'run QC for project Solvent01 for 1h'},
        # Long run - 2 days  
        # {'type': 'human', 'content': 'monitor QC for project Solvent01 for 2d'},
        # Single execution (legacy)
        # {'type': 'human', 'content': 'run QC for project Solvent01'},
    ]
    
    # Create initial state using utility function
    initial_state = create_qc_state(test_messages)
    
    project_name = initial_state['project_name']
    duration_str = initial_state['runtime_duration']
    
    # Check for special 1-minute monitoring pattern
    interval_str, duration_str_from_interval = extract_interval_and_duration_from_messages(test_messages)
    
    print(f"🚀 Starting QC pipeline for project: {project_name}")
    
    # Check if this is the special 1-minute for 10-minutes monitoring
    if interval_str == "1m" and duration_str_from_interval == "10m":
        print(f"⏱️ Special monitoring: Every 1 minute for 10 minutes")
        
        # Run the new 1-minute monitoring function
        result = asyncio.run(run_qc_monitoring_every_minute_for_10_minutes(project_name))
        
        print("\n🎉 QC 1-Minute Monitoring Complete!")
        print("=" * 60)
        print(f"📊 Project: {result.get('project_name')}")
        print(f"⏱️ Total Duration: {result.get('total_duration_seconds')} seconds")
        print(f"🔄 Completed loops: {result.get('completed_loops')}")
        print(f"✅ Files processed: {result.get('files_processed_total')}")
        print(f"⏭️ Files skipped: {result.get('files_skipped_total')}")
        print(f"❌ Errors encountered: {result.get('errors_encountered')}")
        print(f"📈 Success rate: {result.get('success_rate', 0):.1f}%")
        
        if result.get('monitoring_results'):
            print(f"\n📋 Last 5 Monitoring Results:")
            for i, result_msg in enumerate(result['monitoring_results'][-5:], 1):
                print(f"  {i}. {result_msg}")
                
    elif should_use_continuous_monitoring(initial_state):
        print(f"⏱️ Monitoring duration: {duration_str}")
        
        # Run continuous monitoring
        result = asyncio.run(continuous_qc_monitoring_node(initial_state, {}))
        
        print("\n🎉 QC Continuous Monitoring Complete!")
        print("=" * 60)
        print(f"📊 Project: {result.get('project_name')}")
        print(f"⏱️ Duration: {result.get('runtime_duration')}")
        print(f"🔄 Loop iterations: {result.get('loop_iteration')}")
        print(f"✅ Files processed: {result.get('files_processed')}")
        print(f"⏭️ Files skipped: {result.get('files_skipped')}")
        print(f"❌ Errors encountered: {result.get('errors_encountered')}")
        
        if result.get('parsing_result'):
            print(f"\n📋 Monitoring Results:")
            print(result['parsing_result'])
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