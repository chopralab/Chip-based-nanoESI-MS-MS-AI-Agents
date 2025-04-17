"""Worklist Generator Agent

This module defines a WorklistGenerator that processes an input CSV file to generate a worklist file.
It also exposes an asynchronous node function to run the generator as part of a ReAct agent,
using a state graph similar in style to the langgraph template.

The agent works with a chat model with tool calling support.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Literal, cast

import pandas as pd

# Imports for the agent / state graph functionality
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model


class WorklistGenerator:
    """
    A class to generate worklist files from input CSV data.

    Attributes:
        input_file (str): Path to the input CSV file.
        output_file (str): Path where the generated worklist file will be saved.
        column_headers (list): List of predefined column headers for the worklist.
        default_values (dict): Dictionary of default values for the worklist columns.
    """

    def __init__(self, input_file: str, output_file: str):
        """
        Initializes the WorklistGenerator with input and output file paths.

        Args:
            input_file (str): Path to the input CSV file.
            output_file (str): Path where the generated worklist file will be saved.
        """
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
        """
        Generates the worklist DataFrame by processing the input CSV file.

        Workflow:
            1. Load the input CSV file using pandas.
            2. Initialize an empty DataFrame with predefined column headers.
            3. Populate the 'SampleName' and 'AcqMethod' columns from the input data.
            4. Fill the remaining columns with predefined default values.
            5. Save the resulting DataFrame to the output file with a custom header.

        Returns:
            pd.DataFrame: The generated worklist DataFrame.

        Raises:
            FileNotFoundError: If the input CSV file does not exist.
            pd.errors.EmptyDataError: If the input CSV file is empty.
            KeyError: If required columns ('SampleName', 'Method') are missing in the input CSV.
            Exception: For any other unforeseen errors during file reading or writing.
        """
        try:
            # Load the input CSV file
            input_df = pd.read_csv(self.input_file)
            print(f"[DEBUG] Successfully loaded input file: {self.input_file}")
        except FileNotFoundError:
            print(f"[ERROR] Input file not found: {self.input_file}")
            raise
        except pd.errors.EmptyDataError:
            print(f"[ERROR] Input file is empty: {self.input_file}")
            raise
        except Exception as e:
            print(f"[ERROR] An error occurred while reading the input file: {e}")
            raise

        # Initialize the output DataFrame with predefined column headers
        worklist_df = pd.DataFrame(columns=self.column_headers)

        # Populate 'SampleName' and 'AcqMethod' from input DataFrame
        try:
            worklist_df['SampleName'] = input_df['SampleName']
            worklist_df['AcqMethod'] = input_df['Method']
            print("[DEBUG] 'SampleName' and 'AcqMethod' columns populated from input data.")
        except KeyError as e:
            print(f"[ERROR] Missing required column in input data: {e}")
            raise

        # Fill the remaining columns with default values
        for col, default_value in self.default_values.items():
            worklist_df[col] = default_value
        print("[DEBUG] Default values filled for remaining columns.")

        # Save the worklist to the output file with a custom header
        try:
            with open(self.output_file, 'w') as f:
                # Write the custom header line
                f.write('% header=' + '\t'.join(self.column_headers) + '\n')
                # Write the DataFrame data without column headers
                worklist_df.to_csv(f, sep='\t', index=False, header=False)
            print(f"[DEBUG] Worklist file saved successfully to: {os.path.abspath(self.output_file)}")
        except Exception as e:
            print(f"[ERROR] An error occurred while writing the output file: {e}")
            raise

        return worklist_df

    def model_dump_json(self, indent: int = 2) -> str:
        """
        Serializes the WorklistGenerator instance's metadata to a JSON-formatted string.

        Args:
            indent (int, optional): Number of spaces for indentation in the JSON output. Defaults to 2.

        Returns:
            str: JSON-formatted string containing the instance's metadata.
        """
        metadata = {
            'input_file': self.input_file,
            'output_file': self.output_file,
            'column_headers': self.column_headers,
            'default_values': self.default_values
        }
        return json.dumps(metadata, indent=indent)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the WorklistGenerator instance's metadata to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing the instance's metadata.
        """
        return {
            'input_file': self.input_file,
            'output_file': self.output_file,
            'column_headers': self.column_headers,
            'default_values': self.default_values
        }

    def __repr__(self) -> str:
        """
        Returns the official string representation of the WorklistGenerator instance.

        Returns:
            str: String representation of the instance.
        """
        return f"WorklistGenerator(input_file='{self.input_file}', output_file='{self.output_file}')"


async def generate_worklist_node(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """
    Asynchronous node function to generate a worklist using WorklistGenerator.

    This function instantiates a WorklistGenerator (using file paths which, in a full implementation,
    could be passed via state or configuration), generates the worklist, and returns an AIMessage
    containing the result summary.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the agent run.

    Returns:
        Dict[str, List[AIMessage]]: A dictionary with a list of AIMessage responses.
    """
    # For demonstration purposes, using fixed file paths.
    input_file = "input.csv"
    output_file = "output.txt"

    generator = WorklistGenerator(input_file, output_file)
    try:
        worklist_df = generator.generate_worklist()
        result_message = f"Worklist generated successfully with {len(worklist_df)} entries."
    except Exception as e:
        result_message = f"Error generating worklist: {e}"

    return {"messages": [AIMessage(content=result_message)]}


# Define a new state graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add the worklist generation node to the graph
builder.add_node(generate_worklist_node)

# Set the entrypoint as `generate_worklist_node`
builder.add_edge("__start__", "generate_worklist_node")


def route_agent_output(state: State) -> Literal["__end__", "tools"]:
    """
    Determine the next node based on the agent's output.

    This function checks if the last message contains tool calls. If not, the graph execution ends.

    Args:
        state (State): The current state of the conversation.

    Returns:
        Literal["__end__", "tools"]: The next node name.
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then finish the workflow.
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise, execute the requested tools.
    return "tools"


# Add a conditional edge to determine the next step after `generate_worklist_node`
builder.add_conditional_edges(
    "generate_worklist_node",
    route_agent_output,
)

# Optionally, add a tool node and cycle back if tool calls are required.
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge("tools", "generate_worklist_node")

# Compile the builder into an executable graph
graph = builder.compile(
    interrupt_before=[],  # Node names to update state before they're called
    interrupt_after=[],   # Node names to update state after they're called
)
graph.name = "Worklist Generator Agent"
