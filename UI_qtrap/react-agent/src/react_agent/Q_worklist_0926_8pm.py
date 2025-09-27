import asyncio
import logging
import os
import shutil
import traceback
from datetime import datetime
from typing import Any, Dict, List, TypedDict, Optional, Literal

import pandas as pd
import csv
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.tools import TOOLS
from copy import deepcopy
# -----------------------------------------------------------------------------
# Paths & Logging
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) Main worklist input (user-editable, flexible columns)
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
import pandas as pd
import csv
from datetime import datetime

class WorklistGenerator:
    def __init__(
        self,
        worklist_csv: str,
        methods_csv:  str,
        output_dir:   str,
        default_date: str = None,
        default_replicate: int = 3,
        default_operator: str = "SI"
    ):
        self.worklist_csv = worklist_csv
        self.methods_csv  = methods_csv
        self.output_dir   = output_dir

        # Use provided date or today's date, always as YYYYMMDD string
        if default_date:
            self.default_date = default_date
        else:
            self.default_date = datetime.now().strftime("%Y%m%d")
        self.default_replicate = str(default_replicate)
        self.default_operator  = default_operator

        self.column_headers = [
            "SampleName","SampleId","Comments","AcqMethod","ProcMethod","RackCode",
            "PlateCode","VialPos","DilutFact","WghtToVol","Type","RackPos",
            "PlatePos","SetName","OutputFile","_col1Text","_col2Int"
        ]

        self.default_values = {
            "SampleId":    "default_id",
            "Comments":    "default_comment",
            "ProcMethod":  "none",
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
        # --- LOAD AND FORMAT INPUT ---
        # After reading your CSV:
        df_input = pd.read_csv(self.worklist_csv, dtype=str).fillna("")
        df_input = df_input.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Ensure all needed columns are present for forward-fill to work
        # Add support for 'Project' and 'Info2' columns
        for col in ['Date', 'Technical_Replicate', 'Operator', 'Project', 'Info2']:
            if col not in df_input.columns:
                df_input[col] = ""

        # Replace empty/whitespace-only strings with pd.NA, then forward-fill from top row
        for col in ['Date', 'Technical_Replicate', 'Operator', 'Project', 'Info2']:
            df_input[col] = df_input[col].replace(r'^\s*$', pd.NA, regex=True)
            df_input[col] = df_input[col].fillna(method='ffill')

        # After forward-fill, if very first row is still missing, fill with default
        if pd.isna(df_input.at[0, 'Date']) or not str(df_input.at[0, 'Date']).strip():
            df_input.at[0, 'Date'] = self.default_date
        if pd.isna(df_input.at[0, 'Technical_Replicate']) or not str(df_input.at[0, 'Technical_Replicate']).strip():
            df_input.at[0, 'Technical_Replicate'] = self.default_replicate
        if pd.isna(df_input.at[0, 'Operator']) or not str(df_input.at[0, 'Operator']).strip():
            df_input.at[0, 'Operator'] = self.default_operator
        if pd.isna(df_input.at[0, 'Project']) or not str(df_input.at[0, 'Project']).strip():
            df_input.at[0, 'Project'] = "DefaultProject"
        if pd.isna(df_input.at[0, 'Info2']) or not str(df_input.at[0, 'Info2']).strip():
            df_input.at[0, 'Info2'] = "DefaultInfo2"

        # Forward-fill again in case first row was just set
        for col in ['Date', 'Technical_Replicate', 'Operator', 'Project', 'Info2']:
            df_input[col] = df_input[col].fillna(method='ffill')

        # Make sure column order is correct
        needed_cols = ['Info', 'Lipid', 'Date', 'Technical_Replicate', 'Operator', 'Project', 'Info2']
        df_input = df_input[needed_cols]

        # --- LOAD METHODS LOOKUP TABLE ---
        df_methods = pd.read_csv(self.methods_csv, dtype=str)
        df_methods = pd.read_csv(self.methods_csv, dtype=str).fillna("")
        # Build mapping: Lipid -> list of Methods
        import collections
        lipid_methods = collections.defaultdict(list)
        for _, row in df_methods.iterrows():
            lipid = str(row['Lipid']).strip()
            method = str(row['Method']).strip()
            if method:  # Only add if not empty
                lipid_methods[lipid].append(method)


        # --- OUTPUT FILE PATH ---
        import os
        # Extract project name from first row after processing
        project_name = df_input.iloc[0]['Project'] if not df_input.empty else "DefaultProject"
        # Create project-specific output directory
        project_output_dir = os.path.join(self.output_dir, project_name)
        os.makedirs(project_output_dir, exist_ok=True)
        today = datetime.now().strftime("%Y%m%d")
        now = datetime.now()
        hour = now.strftime('%I').lstrip('0')
        minute = now.strftime('%M')
        ampm = now.strftime('%p').lower()
        time_str = f"_{hour}{minute}{ampm}"
        # Change the filename to include project name
        out_name = f"worklist_{today}{time_str}_{project_name}.csv"
        out_path = os.path.join(project_output_dir, out_name)

        # --- BUILD RECORDS ---
        records = []
        unique_lipids_with_methods = {}

        for _, row in df_input.iterrows():
            info = row['Info'].strip() if row['Info'] else ""
            lipid = row['Lipid'].strip() if row['Lipid'] else ""
            date = str(row['Date']).replace("-", "") if row['Date'] else ""
            oper = row['Operator'].strip() if row['Operator'] else ""
            project = row['Project'].strip() if row['Project'] else "DefaultProject"
            info2 = row['Info2'].strip() if row['Info2'] else "DefaultInfo2"
            methods = lipid_methods.get(lipid, [])
            num_reps = int(str(row['Technical_Replicate']))
            
            # Collect unique lipids with their first method only
            if lipid and methods:
                unique_lipids_with_methods[lipid] = methods[0]  # Take only first method

            # If no methods, still create replicates with AcqMethod = "none"
            if not methods:
                for rep in range(1, num_reps+1):
                    sample_name = f"{date}_{info}_{info2}_LC-{lipid}_R-{rep}_Op-{oper}_Proj-{project}"
                    rec = {col: self.default_values.get(col, "") for col in self.column_headers}
                    rec["SampleName"] = sample_name
                    rec["AcqMethod"] = "none"
                    rec["ProcMethod"] = "none"
                    rec["OutputFile"] = sample_name
                    records.append(deepcopy(rec))
            else:
                for method in methods:
                    for rep in range(1, num_reps+1):
                        sample_name = f"{date}_{info}_{info2}_LC-{lipid}_R-{rep}_Op-{oper}_Proj-{project}_{method}"
                        rec = {col: self.default_values.get(col, "") for col in self.column_headers}
                        rec["SampleName"] = sample_name
                        rec["AcqMethod"] = method
                        rec["ProcMethod"] = "none"
                        rec["OutputFile"] = sample_name
                        records.append(deepcopy(rec))

        # Generate exactly 5 BLANK samples total
        lipid_list = list(unique_lipids_with_methods.keys())
        for rep in range(1, 6):  # Exactly 5 BLANKs total
            # Cycle through lipids if we have fewer than 5 lipid classes
            if lipid_list:
                lipid_index = (rep - 1) % len(lipid_list)
                lipid = lipid_list[lipid_index]
                method = unique_lipids_with_methods[lipid]
                blank_name = f"BLANK_LC-{lipid}_R-{rep}_{method}"
                rec = {col: self.default_values.get(col, "") for col in self.column_headers}
                rec["SampleName"] = blank_name
                rec["AcqMethod"] = method
                rec["Type"] = "Blank"
                rec["OutputFile"] = blank_name
                records.append(deepcopy(rec))

        # --- WRITE OUTPUT ---
        df_out = pd.DataFrame(records, columns=self.column_headers)

        with open(out_path, "w", encoding="utf-8", newline='\r\n') as f:
            f.write("% header=" + "\t".join(self.column_headers) + "\n")
            # Write lines without quotes (quoting=csv.QUOTE_NONE)
            df_out.to_csv(
                f,
                sep="\t",
                index=False,
                header=False,
                quoting=csv.QUOTE_NONE,
                escapechar='\\'
            )

        import logging
        logger.info(f"Generated aggregated worklist: {out_path}")

        import time
        time.sleep(1)

        # Server copy (optional)
        SERVER_OUTPUT_DIR = "/mnt/d_drive/Analyst Data/Projects/API Instrument/Batch/QTRAP_worklist"
        try:
            # Create project-specific server output directory
            server_project_dir = os.path.join(SERVER_OUTPUT_DIR, project_name)
            os.makedirs(server_project_dir, exist_ok=True)
            import shutil
            shutil.copy2(out_path, server_project_dir)
            logger.info(f"Copied to server output: {server_project_dir}")
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
