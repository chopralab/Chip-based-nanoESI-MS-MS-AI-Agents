import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# Optional: replace with actual tool list
from react_agent.tools import TOOLS
from react_agent.configuration import Configuration

# -----------------------------------------------------------------------------
# Configure logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Define a TypedDict-based custom state
# -----------------------------------------------------------------------------
class QTRAPState(TypedDict):
    messages: List[BaseMessage]
    input_file: str
    output_file: str
    parsing_result: Optional[str]
    agent_state: Dict[str, Any]

# -----------------------------------------------------------------------------
# QTRAP Parsing Logic
# -----------------------------------------------------------------------------
class QTRAP_Parse:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file

        self.filenames, self.q1_values, self.q3_values = [], [], []
        self.lipids, self.dates, self.sample_names = [], [], []
        self.samples, self.summed_intensities = [], []

    def parse_file(self) -> bool:
        base_name = os.path.splitext(os.path.basename(self.input_file))[0]
        parts = base_name.split('_', 1)
        date_str = parts[0] if len(parts) == 2 else ''
        sample_name = parts[1] if len(parts) == 2 else ''
        sample = 'Blank' if 'Blank' in sample_name else 'Sample'

        try:
            with open(self.input_file, 'r') as file:
                lines = file.readlines()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return False

        current_filename, current_q1, current_q3, current_lipid = "", None, None, ""
        parsing_intensity = False

        for line in lines:
            if 'sourceFile:' in line or 'name:' in line:
                match = re.search(r'name:\s+([\w.]+)', line)
                if match:
                    current_filename = match.group(1)

            if 'id: SRM SIC Q1=' in line:
                match = re.search(r'Q1=(\d+\.\d+).*Q3=(\d+\.\d+).*name=([^\s]+)', line)
                if match:
                    current_q1 = float(match.group(1))
                    current_q3 = float(match.group(2))
                    current_lipid = match.group(3)

            if 'cvParam: intensity array' in line:
                parsing_intensity = True
            elif parsing_intensity and 'binary: [' in line:
                match = re.search(r'binary:\s+\[\d+\]\s+([\d\s]+)', line)
                if match:
                    try:
                        intensities = list(map(int, match.group(1).split()))
                        current_intensity_sum = sum(intensities)
                    except ValueError:
                        current_intensity_sum = 0

                    if current_filename and current_q1 is not None and current_q3 is not None:
                        self.filenames.append(current_filename)
                        self.q1_values.append(current_q1)
                        self.q3_values.append(current_q3)
                        self.lipids.append(current_lipid)
                        self.dates.append(date_str)
                        self.sample_names.append(sample_name)
                        self.samples.append(sample)
                        self.summed_intensities.append(current_intensity_sum)
                parsing_intensity = False
            elif parsing_intensity:
                parsing_intensity = False

        if not self.filenames:
            return False

        data = {
            'Filename': self.filenames,
            'Q1': self.q1_values,
            'Q3': self.q3_values,
            'Lipid': self.lipids,
            'Date': self.dates,
            'Sample_Name': self.sample_names,
            'Sample': self.samples,
            'Summed_Intensity': self.summed_intensities,
        }
        df = pd.DataFrame(data)

        try:
            df.to_csv(self.output_file, index=False)
            self.parsed_df = df
            return True
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            return False

    def run(self) -> Optional[pd.DataFrame]:
        if self.parse_file():
            try:
                return pd.read_csv(self.output_file)
            except Exception as e:
                logger.error(f"Error reading output CSV: {e}")
                return None
        return None

# -----------------------------------------------------------------------------
# Asynchronous Node
# -----------------------------------------------------------------------------
async def parse_chromatogram_node(state: QTRAPState, config: RunnableConfig) -> QTRAPState:
    input_file = state["input_file"]
    output_file = state["output_file"]

    parser = QTRAP_Parse(input_file, output_file)
    df = await asyncio.to_thread(parser.run)

    content = (
        f"Parsing successful: DataFrame with {len(df)} records saved to {output_file}."
        if df is not None
        else "Parsing failed."
    )
    return {
        **state,
        "parsing_result": content,
        "messages": state["messages"] + [AIMessage(content=content)],
    }

# -----------------------------------------------------------------------------
# Router Function
# -----------------------------------------------------------------------------
def route_model_output(state: QTRAPState) -> Literal["__end__", "tools"]:
    last_ai_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), None)
    if not last_ai_message:
        raise ValueError("No AIMessage found.")
    return "tools" if last_ai_message.tool_calls else "__end__"

# -----------------------------------------------------------------------------
# Build the Graph
# -----------------------------------------------------------------------------
builder = StateGraph(QTRAPState)
builder.add_node("parse_chromatogram_node", parse_chromatogram_node)
builder.add_node("tools", ToolNode(TOOLS))

builder.set_entry_point("parse_chromatogram_node")
builder.add_conditional_edges("parse_chromatogram_node", route_model_output)
builder.add_edge("tools", "parse_chromatogram_node")

graph = builder.compile()
graph.name = "QTRAP Parsing Agent with TypedDict State"