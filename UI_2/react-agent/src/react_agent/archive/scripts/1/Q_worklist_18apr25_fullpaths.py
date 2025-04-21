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
# Base directory and logging
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# User-specified log directory
LOG_DIR = '/home/sanjay/QTRAP_memory/sciborg_dev/UI_2/react-agent/data/worklists/logs'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'worklist_agent.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler for INFO+
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler for DEBUG+
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.info(f"Logging initialized. Writing to {LOG_FILE}")

# ----------------------------------------------------------------------------
# Inlined WorklistGenerator
# ----------------------------------------------------------------------------
class WorklistGenerator:
    """
    Generates a tab-delimited worklist from an input CSV.
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
            raise FileNotFoundError(self.input_file)

        df_in = pd.read_csv(self.input_file)
        logger.debug(f"Input rows: {len(df_in)}")

        df_out = pd.DataFrame(columns=self.column_headers)
        df_out['SampleName'] = df_in.get('SampleName', pd.Series(dtype=str))
        df_out['AcqMethod'] = df_in.get('Method', pd.Series(dtype=str))
        for col, val in self.default_values.items():
            df_out[col] = val

        logger.debug(f"Writing output to: {self.output_file}")
        with open(self.output_file, 'w') as f:
            f.write('% header=' + '\t'.join(self.column_headers) + '\n')
            df_out.to_csv(f, sep='\t', index=False, header=False)

        if os.path.isfile(self.output_file):
            logger.info(f"Generated worklist at: {self.output_file}")
        else:
            logger.error(f"Failed to generate {self.output_file}")
        return df_out

# ----------------------------------------------------------------------------
# Copy function
# ----------------------------------------------------------------------------
def copy_client_to_windows(local_path: str, dest_dir: str) -> None:
    abs_src = os.path.abspath(local_path)
    logger.debug(f"Copy source: {abs_src}")
    if not os.path.isfile(abs_src):
        logger.error(f"Missing source file for copy: {abs_src}")
        raise FileNotFoundError(abs_src)

    os.makedirs(dest_dir, exist_ok=True)
    abs_dest = os.path.join(dest_dir, os.path.basename(abs_src))
    logger.debug(f"Copy dest: {abs_dest}")
    shutil.copy2(abs_src, abs_dest)
    if os.path.isfile(abs_dest):
        logger.info(f"Copied to Windows: {abs_dest}")
    else:
        logger.error(f"Copy failed; missing at dest: {abs_dest}")

# ----------------------------------------------------------------------------
# State and Node
# ----------------------------------------------------------------------------
class WorklistState(TypedDict):
    messages: List[BaseMessage]
    input_file: str
    output_file: str
    worklist_df: Optional[pd.DataFrame]
    agent_state: Dict[str, Any]

async def generate_worklist_node(state: WorklistState, config: RunnableConfig) -> WorklistState:
    in_file = state['input_file']
    out_file = state['output_file']

    # Resolve to absolute paths
    if not os.path.isabs(in_file):
        in_file = os.path.join(SCRIPT_DIR, in_file)
    if not os.path.isabs(out_file):
        out_file = os.path.join(SCRIPT_DIR, out_file)

    gen_dir = os.path.dirname(out_file)
    os.makedirs(gen_dir, exist_ok=True)
    logger.info(f"Ensured generate directory: {gen_dir}")

    try:
        df = await asyncio.to_thread(WorklistGenerator(in_file, out_file).generate_worklist)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Generation error: {e}\n{tb}")
        return {**state, 'worklist_df': None,
                'messages': state['messages'] + [AIMessage(content=f"Gen failed: {e}")]}

    logger.info(f"Worklist exists in WSL at: {out_file}")

    win_dir = '/mnt/c/Users/iyer95/OneDrive - purdue.edu/Desktop/MSConvert/worklist_generated'
    try:
        await asyncio.to_thread(copy_client_to_windows, out_file, win_dir)
        copy_msg = f"Copied to Windows dir: {win_dir}"
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Copy error: {e}\n{tb}")
        copy_msg = f"Copy failed: {e}"

    content = f"Success: {len(df)} rows to {out_file}. {copy_msg}"
    return {**state, 'worklist_df': df,
            'messages': state['messages'] + [AIMessage(content=content)]}


def route_model_output(state: WorklistState) -> Literal['__end__','tools']:
    last = next((m for m in reversed(state['messages']) if isinstance(m, AIMessage)), None)
    return 'tools' if getattr(last, 'tool_calls', None) else '__end__'

# ----------------------------------------------------------------------------
# Build graph
# ----------------------------------------------------------------------------
builder = StateGraph(WorklistState)
builder.add_node('generate_worklist_node', generate_worklist_node)
builder.add_node('tools', ToolNode(TOOLS))
builder.set_entry_point('generate_worklist_node')
builder.add_conditional_edges('generate_worklist_node', route_model_output)
builder.add_edge('tools', 'generate_worklist_node')

graph = builder.compile()
graph.name = 'Worklist Generator Agent'
