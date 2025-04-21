import asyncio
import logging
import os
import shutil
import json
import traceback
from typing import Any, Dict, List, Literal, Optional, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.tools import TOOLS

# -----------------------------------------------------------------------------
# Configure logging to console and file
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, 'convert', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'worklist_agent.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# -----------------------------------------------------------------------------
# Inlined WorklistGenerator (copied from scripts/worklist_generator.py)
# -----------------------------------------------------------------------------
class WorklistGenerator:
    """
    A class to generate worklist files from input CSV data.
    """

    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.column_headers = [
            "SampleName", "SampleId", "Comments", "AcqMethod", "ProcMethod", "RackCode",
            "PlateCode", "VialPos", "DilutFact", "WghtToVol", "Type", "RackPos",
            "PlatePos", "SetName", "OutputFile", "_col1Text", "_col2Int"
        ]
        self.default_values = {
            "SampleId": "default_id",
            "Comments": "default_comment",
            "ProcMethod": "none",
            "RackCode": "10 By 10",
            "PlateCode": "N/A",
            "VialPos": 1,
            "DilutFact": 1,
            "WghtToVol": 0,
            "Type": "Unknown",
            "RackPos": 1,
            "PlatePos": 0,
            "SetName": "Set1",
            "OutputFile": "DataSet1",
            "_col1Text": "custCol1_1",
            "_col2Int": 0
        }

    def generate_worklist(self) -> pd.DataFrame:
        logger.debug(f"Loading CSV input from: {self.input_file}")
        if not os.path.isfile(self.input_file):
            logger.error(f"Input file not found: {self.input_file}")
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        input_df = pd.read_csv(self.input_file)
        logger.debug(f"Read {len(input_df)} rows from input CSV.")

        worklist_df = pd.DataFrame(columns=self.column_headers)
        worklist_df["SampleName"] = input_df.get("SampleName", pd.Series(dtype=str))
        worklist_df["AcqMethod"] = input_df.get("Method", pd.Series(dtype=str))

        for col, default in self.default_values.items():
            worklist_df[col] = default
        logger.debug("Filled default values for all columns.")

        logger.debug(f"Writing worklist to: {self.output_file}")
        with open(self.output_file, "w") as f:
            f.write("% header=" + "\t".join(self.column_headers) + "\n")
            worklist_df.to_csv(f, sep="\t", index=False, header=False)

        if os.path.isfile(self.output_file):
            logger.debug(f"Successfully created worklist file: {self.output_file}")
        else:
            logger.error(f"Worklist file missing after write: {self.output_file}")

        return worklist_df

# -----------------------------------------------------------------------------
# New Function: Move file from client to server directory
# -----------------------------------------------------------------------------
def move_client_to_server(local_path: str, dest_dir: str) -> None:
    logger.debug(f"Preparing to move file from {local_path} to {dest_dir}")
    if not os.path.isfile(local_path):
        logger.error(f"File to move does not exist: {local_path}")
        raise FileNotFoundError(f"File to move does not exist: {local_path}")

    if not os.path.exists(dest_dir):
        logger.debug(f"Destination directory not found, creating: {dest_dir}")
        os.makedirs(dest_dir, exist_ok=True)

    dest_path = os.path.join(dest_dir, os.path.basename(local_path))
    logger.debug(f"Moving file to: {dest_path}")
    shutil.move(local_path, dest_path)
    if os.path.isfile(dest_path):
        logger.debug(f"Move successful, file now at: {dest_path}")
    else:
        logger.error(f"Move reported success but file missing at destination: {dest_path}")

# -----------------------------------------------------------------------------
# TypedDict State for the Agent
# -----------------------------------------------------------------------------
class WorklistState(TypedDict):
    messages: List[BaseMessage]
    input_file: str
    output_file: str
    worklist_df: Optional[pd.DataFrame]
    agent_state: Dict[str, Any]

# -----------------------------------------------------------------------------
# Async Node: generate_worklist_node
# -----------------------------------------------------------------------------
async def generate_worklist_node(state: WorklistState, config: RunnableConfig) -> WorklistState:
    input_file = state["input_file"]
    output_file = state["output_file"]

    logger.debug("Starting worklist generation node.")
    logger.debug(f"State input_file: {input_file}")
    logger.debug(f"State output_file: {output_file}")

    try:
        generator = WorklistGenerator(input_file, output_file)
        df = await asyncio.to_thread(generator.generate_worklist)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error during worklist generation: {e}\n{tb}")
        return {
            **state,
            "worklist_df": None,
            "messages": state["messages"] + [AIMessage(content=f"Worklist generation failed: {e}")]
        }

    windows_dest_dir = "/mnt/c/Users/iyer95/OneDrive - purdue.edu/Desktop/MSConvert/worklist_generated"
    try:
        logger.debug(f"Attempting to move file to Windows directory: {windows_dest_dir}")
        await asyncio.to_thread(move_client_to_server, output_file, windows_dest_dir)
        move_msg = f"File successfully moved to {windows_dest_dir}" 
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error moving file: {e}\n{tb}")
        move_msg = f"Failed to move file: {e}"

    content = (
        f"Worklist generation successful: {len(df)} rows saved to {output_file}. {move_msg}"
    )

    return {
        **state,
        "worklist_df": df,
        "messages": state["messages"] + [AIMessage(content=content)],
    }

# -----------------------------------------------------------------------------
# Router: decide whether to invoke tools again or end
# -----------------------------------------------------------------------------
def route_model_output(state: WorklistState) -> Literal["__end__", "tools"]:
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None
    )
    if not last_ai:
        raise ValueError("No AIMessage found in state.")
    return "tools" if getattr(last_ai, "tool_calls", None) else "__end__"

# -----------------------------------------------------------------------------
# Build & compile the StateGraph
# -----------------------------------------------------------------------------
builder = StateGraph(WorklistState)
builder.add_node("generate_worklist_node", generate_worklist_node)
builder.add_node("tools", ToolNode(TOOLS))

builder.set_entry_point("generate_worklist_node")
builder.add_conditional_edges("generate_worklist_node", route_model_output)
builder.add_edge("tools", "generate_worklist_node")

graph = builder.compile()
graph.name = "Worklist Generator Agent with TypedDict State"
