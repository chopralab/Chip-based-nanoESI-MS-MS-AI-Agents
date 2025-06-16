import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.tools import TOOLS

# -----------------------------------------------------------------------
# Hard‑coded directories based on input date folder
# -----------------------------------------------------------------------
# Input directory (hardcoded date subfolder)
INPUT_DIR = "/home/qtrap/Chip-based-nanoESI-MS-MS-AI-Agents/UI_qtrap/react-agent/src/react_agent/data/text/20250421"
# Derive date folder name
DATE_FOLDER = os.path.basename(INPUT_DIR)

# Output base directory
OUTPUT_BASE_DIR = "/home/qtrap/Chip-based-nanoESI-MS-MS-AI-Agents/UI_qtrap/react-agent/src/react_agent/data/csv"
# Create dated subfolder for CSV outputs
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, DATE_FOLDER)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logs base directory
LOG_BASE_DIR = "/home/qtrap/Chip-based-nanoESI-MS-MS-AI-Agents/UI_qtrap/react-agent/src/react_agent/data/logs/parse"
# Create dated subfolder for logs
LOG_DIR = os.path.join(LOG_BASE_DIR, DATE_FOLDER)
os.makedirs(LOG_DIR, exist_ok=True)
# Log file path without parentheses
LOG_FILE = os.path.join(LOG_DIR, f"parse_{DATE_FOLDER}.log")

# -----------------------------------------------------------------------
# Logger configuration
# -----------------------------------------------------------------------
formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -----------------------------------------------------------------------
# TypedDict state
# -----------------------------------------------------------------------
class QTRAPState(TypedDict):
    messages: List[BaseMessage]
    parsing_result: Optional[str]
    window_size: int
    threshold_factor: float
    top_group_count: int
    agent_state: Dict[str, Any]

# -----------------------------------------------------------------------
# QTRAP Parsing Logic with RSD & Summed TIC
# -----------------------------------------------------------------------
class QTRAP_Parse:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        window_size: int = 7,
        threshold_factor: float = 0.2,
        top_group_count: int = 5
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.top_group_count = top_group_count

        # storage for lipid‑level data
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
        # --- extract time points ---
        tic_time: Optional[float] = None
        time_points: List[float] = []
        if len(lines) > 95:
            m = re.search(r"binary:\s+\[\d+\]\s+(.*)", lines[95].strip())
            if m:
                time_points = [float(v) for v in m.group(1).split()]
                tic_time = time_points[-1] if time_points else None

        # --- extract intensity points ---
        intensity_floats: List[float] = []
        if len(lines) > 98:
            m = re.search(r"binary:\s+\[\d+\]\s+(.*)", lines[98].strip())
            if m:
                try:
                    intensity_floats = [float(v) for v in m.group(1).split()]
                except ValueError:
                    intensity_floats = []

        # --- total RSD ---
        tic_rsd_total: Optional[float]
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

        # --- rolling‑window RSD ---
        window_rsds: List[float] = []
        n = len(intensity_floats)
        threshold: Optional[float] = None
        if intensity_floats:
            top5 = sorted(intensity_floats, reverse=True)[:5]
            threshold = (sum(top5) / len(top5)) * self.threshold_factor

        if threshold is not None and n >= self.window_size:
            for i in range(n - self.window_size + 1):
                win = intensity_floats[i:i+self.window_size]
                if min(win) < threshold:
                    continue
                mean_w = sum(win) / len(win)
                if mean_w:
                    var_w = sum((x - mean_w)**2 for x in win) / (len(win) - 1)
                    std_w = var_w**0.5
                    window_rsds.append((std_w / mean_w) * 100)

        tic_rsd_window_best = min(window_rsds) if window_rsds else None

        # --- average of best N windows ---
        if window_rsds:
            best_n = sorted(window_rsds)[:self.top_group_count]
            tic_rsd_top_group = sum(best_n) / len(best_n)
        else:
            tic_rsd_top_group = None

        return tic_time, tic_rsd_total, window_rsds, tic_rsd_window_best, tic_rsd_top_group

    def Summed_TIC(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Summed_TIC'] = df.groupby('Sample_Name')['Summed_Intensity'].transform('sum')
        return df

    def parse_file(self) -> bool:
        # read lines
        try:
            with open(self.input_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading file {self.input_file}: {e}")
            return False

        # extract date & sample from filename
        base = os.path.splitext(os.path.basename(self.input_file))[0]
        parts = base.split('_', 1)
        date_str, sample_name = (parts + ['', ''])[:2]
        sample = 'Blank' if 'Blank' in sample_name else 'Sample'

        # parse lipid sections
        parsing = False
        current_q1 = current_q3 = None
        current_lipid = current_filename = ""

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

        # compute TIC RSD metrics
        tic_time, tic_rsd_total, tic_rsds, tic_best, tic_topgroup = self.TIC_RSD(lines)

        # assemble DataFrame
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

        # add summed TIC per sample
        df = self.Summed_TIC(df)

        # write out
        try:
            df.to_csv(self.output_file, index=False)
            self.parsed_df = df
            return True
        except Exception as e:
            logger.error(f"Error saving CSV to {self.output_file}: {e}")
            return False

    def run(self) -> Optional[pd.DataFrame]:
        if self.parse_file():
            try:
                return pd.read_csv(self.output_file)
            except Exception as e:
                logger.error(f"Error reading CSV back in: {e}")
        return None

# -----------------------------------------------------------------------
# Batch parsing node
# -----------------------------------------------------------------------
async def parse_all_chromatograms_node(state: QTRAPState, config: RunnableConfig) -> QTRAPState:
    results: List[str] = []

    # run os.listdir in thread to avoid blocking
    files = await asyncio.to_thread(os.listdir, INPUT_DIR)

    for fn in files:
        if not fn.lower().endswith(".txt"):
            continue

        in_path  = os.path.join(INPUT_DIR, fn)
        base     = os.path.splitext(fn)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{DATE_FOLDER}_{base}.csv")

        parser = QTRAP_Parse(
            in_path,
            out_path,
            **{k: state[k] for k in ("window_size", "threshold_factor", "top_group_count") if k in state}
        )
        df = await asyncio.to_thread(parser.run)

        if df is not None:
            msg = f"✅ Parsed {fn} → {base}.csv ({len(df)} rows)"
            logger.info(msg)
            results.append(msg)
        else:
            msg = f"❌ Failed to parse {fn}"
            logger.error(msg)
            results.append(msg)

    summary = "\n".join(results) or "No .txt files found to parse."
    return {
        **state,
        "parsing_result": summary,
        "messages": state["messages"] + [AIMessage(content=summary)],
    }

# -----------------------------------------------------------------------
# Router & Graph
# -----------------------------------------------------------------------
def route_model_output(state: QTRAPState) -> Literal["__end__", "tools"]:
    last = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    if not last:
        raise ValueError("No AIMessage found.")
    return "tools" if last.tool_calls else "__end__"

builder = StateGraph(QTRAPState)
builder.add_node("parse_all_chromatograms_node", parse_all_chromatograms_node)
builder.add_node("tools", ToolNode(TOOLS))
builder.set_entry_point("parse_all_chromatograms_node")
builder.add_conditional_edges("parse_all_chromatograms_node", route_model_output)
builder.add_edge("tools", "parse_all_chromatograms_node")

graph = builder.compile()
graph.name = "QTRAP Batch Parsing Agent"
