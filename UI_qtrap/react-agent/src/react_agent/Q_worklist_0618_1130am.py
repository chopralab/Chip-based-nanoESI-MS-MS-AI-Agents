import asyncio
import logging
import os
import shutil
import traceback
from datetime import datetime
from typing import Any, Dict, List, TypedDict, Optional, Literal

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.tools import TOOLS

# -----------------------------------------------------------------------------
# Paths & Logging
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) Main worklist input (one row per sample)
INPUT_WORKLIST = os.path.join(
    SCRIPT_DIR, "data", "worklist", "input", "input_worklist.csv"
)

# 2) Lipid→Method lookup table
METHODS_CSV = os.path.join(
    SCRIPT_DIR, "data", "worklist", "methods.csv"
)

# Where the aggregated CSV will be written
OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "data", "worklist", "generated"
)

# Logs go here, rotated by date
LOG_DIR = os.path.join(
    SCRIPT_DIR, "data", "logs", "worklist"
)
os.makedirs(LOG_DIR, exist_ok=True)
today_str = datetime.now().strftime("%Y%m%d")
LOG_FILE = os.path.join(LOG_DIR, f"worklist_{today_str}.log")

# Optional: Windows copy target
SERVER_OUTPUT_DIR = "/mnt/d_drive/Analyst Data/Projects/API Instrument/Batch/QTRAP_worklist"

# set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(console_handler)

# file handler
file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

logger.info(f"Logging to {LOG_FILE}")

# -----------------------------------------------------------------------------
# WorklistGenerator: builds one aggregated CSV per run
# -----------------------------------------------------------------------------
class WorklistGenerator:
    def __init__(self,
                 worklist_csv: str,
                 methods_csv:  str,
                 output_dir:   str):
        self.worklist_csv = worklist_csv
        self.methods_csv  = methods_csv
        self.output_dir   = output_dir

        # fixed columns
        self.column_headers = [
            "SampleName","SampleId","Comments","AcqMethod","ProcMethod","RackCode",
            "PlateCode","VialPos","DilutFact","WghtToVol","Type","RackPos",
            "PlatePos","SetName","OutputFile","_col1Text","_col2Int"
        ]

        # defaults for every column except SampleName & AcqMethod
        self.default_values = {
            "SampleId":    "default_id",
            "Comments":    "default_comment",
            "ProcMethod":  "none",          # fallback if no lipid match
            "RackCode":    "10 By 10",
            "PlateCode":   "N/A",
            "VialPos":     1,
            "DilutFact":   1,
            "WghtToVol":   0,
            "Type":        "Unknown",
            "RackPos":     1,
            "PlatePos":    0,
            "SetName":     "Set1",
            "OutputFile":  "DataSet1",
            "_col1Text":   "custCol1_1",
            "_col2Int":    0
        }

    def generate_worklist(self) -> str:
        # load inputs
        df_input   = pd.read_csv(self.worklist_csv)
        df_methods = pd.read_csv(self.methods_csv)
        lipid_map  = dict(zip(df_methods["Lipid"], df_methods["Method"]))

        os.makedirs(self.output_dir, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")
        # Add hour and minute in 12-hour format with am/pm
        now = datetime.now()
        hour = now.strftime('%I').lstrip('0')  # Remove leading zero
        minute = now.strftime('%M')
        ampm = now.strftime('%p').lower()
        time_str = f"_{hour}{minute}{ampm}"
        out_name = f"worklist_{today}{time_str}.csv"
        out_path = os.path.join(self.output_dir, out_name)

        # collect one record per input row
        records = []
        for _, row in df_input.iterrows():
            sample_name_base = "_".join(
                str(row[col]) for col in
                ["Date","Info","Lipid","Technical_Replicate","Operator"]
            )
            # Find all methods in methods.csv with this Lipid class
            matched_methods = df_methods[df_methods["Lipid"] == row["Lipid"]]["Method"].tolist()
            if not matched_methods:
                # fallback to default ProcMethod if no match
                matched_methods = [self.default_values["ProcMethod"]]
            for method in matched_methods:
                sample_name = f"{sample_name_base}_{method}" if len(matched_methods) > 1 else sample_name_base
                rec = {"SampleName": sample_name, "AcqMethod": method}
                for col, val in self.default_values.items():
                    if col == "ProcMethod":
                        rec[col] = "none"
                    elif col == "OutputFile":
                        rec[col] = sample_name  # OutputFile matches SampleName
                    else:
                        rec[col] = val
                records.append(rec)


        # build DataFrame & write once
        df_out = pd.DataFrame(records, columns=self.column_headers)
        with open(out_path, "w", encoding="utf-8", newline='\r\n') as f:
            f.write("% header=" + "\t".join(self.column_headers) + "\n")
            df_out.to_csv(f, sep="\t", index=False, header=False)

        logger.info(f"Generated aggregated worklist: {out_path}")

        # Wait 5 seconds to ensure file is fully written and synced
        import time
        time.sleep(1)

        # Also copy to the server output directory
        try:
            os.makedirs(SERVER_OUTPUT_DIR, exist_ok=True)
            shutil.copy2(out_path, SERVER_OUTPUT_DIR)
            logger.info(f"Copied to server output: {SERVER_OUTPUT_DIR}")
        except Exception:
            logger.debug("Server copy failed:\n" + traceback.format_exc())

        return out_path

# -----------------------------------------------------------------------------
# Agent state & node
# -----------------------------------------------------------------------------
class WorklistState(TypedDict):
    messages:       List[BaseMessage]
    worklist_paths: Optional[List[str]]
    agent_state:    Dict[str, Any]

async def generate_worklist_node(
    state: WorklistState,
    config: RunnableConfig
) -> WorklistState:
    try:
        path = await asyncio.to_thread(
            WorklistGenerator(INPUT_WORKLIST, METHODS_CSV, OUTPUT_DIR).generate_worklist
        )
        msg = f"Success: aggregated worklist written to {path}"
        state["worklist_paths"] = [path]
    except Exception as e:
        msg = f"Failed: {e}"
        logger.error(msg + "\n" + traceback.format_exc())
        state["worklist_paths"] = None

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=msg)]
    }

def route_model_output(state: WorklistState) -> Literal["__end__", "tools"]:
    last = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None
    )
    return "tools" if getattr(last, "tool_calls", None) else "__end__"

# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------
builder = StateGraph(WorklistState)
builder.add_node("generate_worklist_node", generate_worklist_node)
builder.add_node("tools", ToolNode(TOOLS))
builder.set_entry_point("generate_worklist_node")
builder.add_conditional_edges("generate_worklist_node", route_model_output)
builder.add_edge("tools", "generate_worklist_node")

graph = builder.compile()
graph.name = "Worklist Generator Agent"
