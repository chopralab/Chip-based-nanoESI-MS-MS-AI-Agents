import asyncio
import logging
import shutil
import subprocess
import traceback
from typing import Any, Dict, List, Literal, Optional, TypedDict
from pathlib import Path
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# ----------------------------------------------------------------------------
# Date Configuration
# ----------------------------------------------------------------------------
DATE_STR = datetime.now().strftime("%Y%m%d")

# ----------------------------------------------------------------------------
# Directory Configuration
# ----------------------------------------------------------------------------
TEXT_BASE_DIR = Path("/home/qtrap/Chip-based-nanoESI-MS-MS-AI-Agents/UI_qtrap/react-agent/src/react_agent/data/text")
TEXT_TARGET_DIR = TEXT_BASE_DIR / DATE_STR

LOG_DIR = Path("/home/qtrap/Chip-based-nanoESI-MS-MS-AI-Agents/UI_qtrap/react-agent/src/react_agent/data/logs/convert")
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

async def convert_node(state: ConvertState, config: RunnableConfig) -> ConvertState:
    converter = Convert()
    try:
        await converter.setup_directories()
        await converter.run_conversion()
        moved_files = await converter.move_converted()
        msg = f"Success: Conversion completed. Files moved: {moved_files}."
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error: {e}\n{tb}")
        msg = f"Failed: {e}"
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
