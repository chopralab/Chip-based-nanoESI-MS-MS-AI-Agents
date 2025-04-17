# src/react_agent/Q_worklist.py

import asyncio
import logging
import os
import json
from typing import Any, Dict, List, Literal, Optional, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.tools import TOOLS

# -----------------------------------------------------------------------------
# Configure logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        # 1. Load input CSV
        input_df = pd.read_csv(self.input_file)

        # 2. Init output DataFrame
        worklist_df = pd.DataFrame(columns=self.column_headers)
        worklist_df["SampleName"] = input_df["SampleName"]
        worklist_df["AcqMethod"] = input_df["Method"]

        # 3. Fill defaults
        for col, default in self.default_values.items():
            worklist_df[col] = default

        # 4. Write to file with custom header
        with open(self.output_file, "w") as f:
            f.write("% header=" + "\t".join(self.column_headers) + "\n")
            worklist_df.to_csv(f, sep="\t", index=False, header=False)

        return worklist_df

    def model_dump_json(self, indent: int = 2) -> str:
        metadata = {
            "input_file": self.input_file,
            "output_file": self.output_file,
            "column_headers": self.column_headers,
            "default_values": self.default_values,
        }
        return json.dumps(metadata, indent=indent)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_file": self.input_file,
            "output_file": self.output_file,
            "column_headers": self.column_headers,
            "default_values": self.default_values,
        }

    def __repr__(self) -> str:
        return f"WorklistGenerator(input_file='{self.input_file}', output_file='{self.output_file}')"


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

    generator = WorklistGenerator(input_file, output_file)
    df = await asyncio.to_thread(generator.generate_worklist)

    content = (
        f"Worklist generation successful: {len(df)} rows saved to {output_file}."
        if df is not None
        else "Worklist generation failed."
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
