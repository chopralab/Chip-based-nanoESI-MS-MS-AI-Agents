import asyncio
import logging
import shutil
import subprocess
import traceback
import re
from typing import Any, Dict, List, Literal, Optional, TypedDict
from pathlib import Path
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

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
        debug_log.append(f"📁 Scanning directory: {server_dir}")
        # Use asyncio.to_thread to make the blocking glob operation non-blocking
        all_files = await asyncio.to_thread(lambda: list(server.glob('*.wiff')))
        debug_log.append(f"📄 Found {len(all_files)} .wiff files total")
        
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
        # Parse the date string (YYYYMMDD format)
        try:
            target_date = datetime.strptime(date_str, "%Y%m%d").date()
            debug_log.append(f"🗓️ Looking for files modified on: {target_date}")
        except ValueError:
            debug_log.append(f"❌ Invalid date format '{date_str}'. Expected YYYYMMDD format.")
            return [], debug_log
        
        matching_files = []
        
        for wiff_file in all_files:
            # Get file modification date - make stat() non-blocking
            file_stat = await asyncio.to_thread(wiff_file.stat)
            file_mod_time = datetime.fromtimestamp(file_stat.st_mtime)
            file_mod_date = file_mod_time.date()
            
            if file_mod_date == target_date:
                matching_files.append(wiff_file.name)
                scan_file = wiff_file.with_suffix(wiff_file.suffix + '.scan')
                # Make exists() check non-blocking
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
            # Show recent files with their modification dates - make stat calls non-blocking
            debug_log.append("📅 Recent files and their modification dates:")
            
            # Sort files by modification time (most recent first) - make this non-blocking
            file_times = []
            for f in all_files:
                stat_info = await asyncio.to_thread(f.stat)
                file_times.append((f, stat_info.st_mtime))
            
            recent_files = sorted(file_times, key=lambda x: x[1], reverse=True)[:10]
            for f, mtime in recent_files:
                mod_time = datetime.fromtimestamp(mtime)
                debug_log.append(f"   • {f.name} - {mod_time.strftime('%Y-%m-%d %H:%M')}")
    else:
        # No date provided: Find most recent .wiff, ensure .scan exists
        debug_log.append("🕒 No date specified, looking for most recent files...")
        if not all_files:
            debug_log.append("❌ No .wiff files found in server directory")
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

# CLI entry point for direct script execution
async def main():
    print("Testing WIFF+SCAN file movement...")
    print("Enter date in YYYYMMDD format (e.g., 20250327) to find files modified on that date,")
    print("or leave blank to get the most recent files:")
    date_input = input("Date: ").strip()
    moved, debug = await move_wiff_pairs(date_str=date_input if date_input else None)
    print("\n" + "\n".join(debug))
    print(f"\nFiles moved: {moved}")

# ----------------------------------------------------------------------------
# Date Configuration
# ----------------------------------------------------------------------------
DATE_STR = datetime.now().strftime("%Y%m%d")

# ----------------------------------------------------------------------------
# Directory Configuration
# ----------------------------------------------------------------------------
TEXT_BASE_DIR = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/text")
TEXT_TARGET_DIR = TEXT_BASE_DIR / DATE_STR

LOG_DIR = Path("/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/logs/convert")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"convert_{DATE_STR}.log"

# ----------------------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.info(f"Logging initialized. Writing to {LOG_FILE}")

# ----------------------------------------------------------------------------
# Conversion Directory Configuration (Windows and WSL)
# ----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
# Updated directory mappings for ChopraLab laptop
WINDOWS_BASE_DIR = r"C:\Users\ChopraLab\Desktop\laptop\convert_raw"
WINDOWS_RAW_DIR = rf"{WINDOWS_BASE_DIR}\raw_data"
WINDOWS_CONVERTED_DIR = rf"{WINDOWS_BASE_DIR}\converted_files"
WINDOWS_BATCH_FILE = rf"{WINDOWS_BASE_DIR}\convert_files.bat"

WSL_BASE_DIR = Path("/mnt/c/Users/ChopraLab/Desktop/laptop/convert_raw")
WSL_RAW_DIR = WSL_BASE_DIR / "raw_data"
WSL_CONVERTED_DIR = WSL_BASE_DIR / "converted_files"
WSL_BATCH_FILE = WSL_BASE_DIR / "convert_files.bat"

# For compatibility with downstream code
WSL_SOURCE_DIR = WSL_CONVERTED_DIR

# Final output directory (WSL only)
WSL_TARGET_DIR = TEXT_TARGET_DIR

# ----------------------------------------------------------------------------
# Convert Class
# ----------------------------------------------------------------------------
class Convert:
    async def setup_directories(self):
        # Create the dated text output directory (for WSL/Linux post-processing)
        await asyncio.to_thread(lambda: WSL_TARGET_DIR.mkdir(parents=True, exist_ok=True))

    async def run_conversion(self) -> None:
        logger.info("Checking for batch file at: %s", WSL_BATCH_FILE)
        if not WSL_BATCH_FILE.exists():
            raise FileNotFoundError(f"Batch file not found: {WSL_BATCH_FILE}")

        cmd = [
            "cmd.exe",
            "/c",
            f"cd /d {WINDOWS_BASE_DIR} && convert_files.bat"
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
# LangGraph State and Node
# ----------------------------------------------------------------------------
class ConvertState(TypedDict):
    messages: List[BaseMessage]
    converted_files: Optional[List[str]]
    agent_state: Dict[str, Any]
    date: Optional[str]  # Add optional date parameter

async def convert_node(state: ConvertState, config: RunnableConfig) -> ConvertState:
    converter = Convert()
    debug_info = []
    
    try:
        # Get date from multiple sources in order of priority:
        date_str = None
        
        # Check direct date parameter first
        if state.get('date'):
            date_str = state['date']
            debug_info.append(f"📅 Using date from direct parameter: {date_str}")
        elif 'date_str' in state.get('agent_state', {}):
            date_str = state['agent_state']['date_str']
            debug_info.append(f"📅 Using date from agent_state: {date_str}")
        else:
            # Try to extract from last HumanMessage
            for m in reversed(state['messages']):
                if isinstance(m, HumanMessage):
                    # Look for 8-digit date pattern
                    match = re.search(r'\b(\d{8})\b', m.content)
                    if match:
                        date_str = match.group(1)
                        debug_info.append(f"📅 Extracted date from message: {date_str}")
                        break
        
        if not date_str:
            debug_info.append("📅 No date specified, will use most recent files")
        else:
            debug_info.append(f"📅 Looking for files modified on date: {date_str} (YYYYMMDD format)")
        
        # Move WIFF+SCAN files - now returns debug info too
        moved_wiff, file_debug = await move_wiff_pairs(date_str=date_str)
        debug_info.extend(file_debug)
        
        logger.info(f"Moved WIFF+SCAN pairs: {moved_wiff}")
        
        # Continue with conversion only if files were moved
        if moved_wiff:
            debug_info.append("🔄 Starting conversion process...")
            await converter.setup_directories()
            await converter.run_conversion()
            moved_files = await converter.move_converted()
            debug_info.append(f"✅ Conversion completed successfully")
            
            # Create detailed success message
            msg_parts = [
                f"🎉 SUCCESS! Processed {len(moved_wiff)} file pairs:",
                f"📤 WIFF+SCAN files moved: {moved_wiff}",
                f"📝 Text files created: {moved_files}",
                "",
                "📋 Debug Information:",
                *debug_info
            ]
            msg = "\n".join(msg_parts)
        else:
            # Create detailed failure message with debug info
            msg_parts = [
                "❌ FAILED: No files were processed",
                "",
                "📋 Debug Information:",
                *debug_info,
                "",
                f"💡 Tip: Check if files with date '{date_str}' exist in the source directory"
            ]
            msg = "\n".join(msg_parts)
            moved_files = []
            
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error: {e}\n{tb}")
        
        error_parts = [
            f"💥 ERROR: {str(e)}",
            "",
            "📋 Debug Information:",
            *debug_info,
            "",
            "🔍 Full traceback logged to file"
        ]
        msg = "\n".join(error_parts)
        moved_files = None

    return {
        **state,
        'converted_files': moved_files,
        'messages': state['messages'] + [AIMessage(content=msg)]
    }

# ----------------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------------
def route_model_output(state: ConvertState) -> Literal['__end__','tools']:
    last = next((m for m in reversed(state['messages']) if isinstance(m, AIMessage)), None)
    return 'tools' if getattr(last, 'tool_calls', None) else '__end__'

# ----------------------------------------------------------------------------
# Build LangGraph
# ----------------------------------------------------------------------------
builder = StateGraph(ConvertState)
builder.add_node('convert_node', convert_node)
builder.add_node('tools', ToolNode([]))  # Add tools as necessary
builder.set_entry_point('convert_node')
builder.add_conditional_edges('convert_node', route_model_output)
builder.add_edge('tools', 'convert_node')

graph = builder.compile()
graph.name = 'QC Convert Agent'

if __name__ == '__main__':
    asyncio.run(main())