"""Define a custom QTRAP Parsing Agent using a ReAct-style graph with a custom state.

This agent wraps a chromatogram file parser (QTRAP_Parse) as an asynchronous node.
It demonstrates how to incorporate a parsing microservice into a state graph while
tracking custom variables.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast, Optional

import re
import os
import pandas as pd

# Imports for the graph and agent framework
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

from pydantic import Extra

# -----------------------------------------------------------------------------
# Define a custom state to track variables
# -----------------------------------------------------------------------------
class QTRAPState(InputState):
    input_file: str = "/home/sanjay/QTRAP_memory/sciborg_dev/UI/src/react_agent/qc_txt_files/20241115_Plasma_PC.txt"
    # input_file: str
    output_file: str = "/home/sanjay/QTRAP_memory/sciborg_dev/UI/src/react_agent/qc_txt_files/output/test.csv"
    parsing_result: Optional[str] = None

    def __init__(self, **data):
        # Pass all received keyword arguments to the base class.
        super().__init__(**data)
    class Config:
        extra = Extra.allow
         


# -----------------------------------------------------------------------------
# QTRAP Parsing Logic (almost unchanged from your original code)
# -----------------------------------------------------------------------------
class QTRAP_Parse:
    def __init__(self, input_file: str, output_file: str):
        """
        Initialize the parser with input and output file paths.

        Args:
            input_file (str): Path to the .txt file to parse.
            output_file (str): Path where the CSV will be saved.
        """
        self.input_file = input_file
        self.output_file = output_file

        # Initialize lists to store parsed data
        self.filenames = []
        self.q1_values = []
        self.q3_values = []
        self.lipids = []
        self.dates = []
        self.sample_names = []
        self.samples = []
        self.summed_intensities = []

        # Define lipid classes and their abbreviations (currently unused in parsing)
        self.lipid_classes = {
            'Phosphatidylcholine': 'PC',
            'Phosphatidylethanolamine': 'PE',
            'Phosphatidylserine': 'PS',
            'Phosphatidylinositol': 'PI',
            'Sphingomyelin': 'SM',
            'Ceramide': 'Cer',
            'Triglyceride': 'TG',
            'Diacylglycerol': 'DG',
            'Cholesterol Ester': 'CE',
            'Cardiolipin': 'CL',
            'Acyl Carnitine': 'Car',
            'Fatty Acid': 'FA'
        }

    def parse_file(self) -> bool:
        """
        Parse the input chromatogram .txt file and extract necessary data.

        Returns:
            bool: True if parsing and saving were successful, False otherwise.
        """
        file_path = self.input_file
        print(f"Parsing file: {file_path}")

        # Extract Date and Sample_Name from the filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        parts = base_name.split('_', 1)
        if len(parts) == 2:
            date_str = parts[0]
            sample_name = parts[1]
            print(f"Extracted date: {date_str}")
            print(f"Extracted sample name: {sample_name}")
        else:
            date_str = ''
            sample_name = ''
            print("Warning: Filename does not contain an underscore. Date and Sample_Name set to empty strings.")

        # Determine the Sample value
        sample = 'Blank' if 'Blank' in sample_name else 'Sample'
        print(f"Determined sample type: {sample}")

        # Open and read file lines
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            print(f"Successfully read file: {file_path} (Total lines: {len(lines)})")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return False

        # Initialize current values
        current_filename = ""
        current_q1 = None
        current_q3 = None
        current_lipid = ""
        parsing_intensity = False

        for line_num, line in enumerate(lines):
            # Extract filename from 'sourceFile:' or 'name:'
            if 'sourceFile:' in line or 'name:' in line:
                match = re.search(r'name:\s+([\w.]+)', line)
                if match:
                    current_filename = match.group(1)
                    print(f"Extracted current_filename at line {line_num}: {current_filename}")

            # Extract Q1, Q3 values and Lipid name
            if 'id: SRM SIC Q1=' in line:
                match = re.search(r'Q1=(\d+\.\d+).*Q3=(\d+\.\d+).*name=([^\s]+)', line)
                if match:
                    current_q1 = float(match.group(1))
                    current_q3 = float(match.group(2))
                    current_lipid = match.group(3)
                    print(f"Extracted Q1: {current_q1}, Q3: {current_q3}, Lipid: {current_lipid} at line {line_num}")

            # Look for intensity array data
            if 'cvParam: intensity array' in line:
                parsing_intensity = True
                print(f"'cvParam: intensity array' found at line {line_num}")
            elif parsing_intensity and 'binary: [' in line:
                match = re.search(r'binary:\s+\[\d+\]\s+([\d\s]+)', line)
                if match:
                    try:
                        intensities = list(map(int, match.group(1).split()))
                        current_intensity_sum = sum(intensities)
                        print(f"Extracted intensity data for lipid {current_lipid} at line {line_num}: Sum={current_intensity_sum}")
                    except ValueError:
                        current_intensity_sum = 0
                        print(f"Failed to parse intensity data for lipid {current_lipid} at line {line_num}")

                    if current_filename and current_q1 is not None and current_q3 is not None:
                        self.filenames.append(current_filename)
                        self.q1_values.append(current_q1)
                        self.q3_values.append(current_q3)
                        self.lipids.append(current_lipid)
                        self.dates.append(date_str)
                        self.sample_names.append(sample_name)
                        self.samples.append(sample)
                        self.summed_intensities.append(current_intensity_sum)
                        print(f"Appended data for lipid {current_lipid}: Q1={current_q1}, Q3={current_q3}, Sum_Intensity={current_intensity_sum}")
                parsing_intensity = False
            elif parsing_intensity:
                parsing_intensity = False
                print(f"Unexpected line while parsing intensities at line {line_num}: {line.strip()}")

        if not self.filenames:
            print(f"No data parsed from {file_path}")
            return False

        # Create and save DataFrame
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
        print(f"Created DataFrame with {len(df)} records.")

        try:
            df.to_csv(self.output_file, index=False)
            print(f"Data successfully saved to {self.output_file}")
            self.parsed_df = df
            return True
        except Exception as e:
            print(f"Error saving CSV to {self.output_file}: {e}")
            return False

    def run(self) -> Optional[pd.DataFrame]:
        """
        Execute the parsing process and return the parsed DataFrame.

        Returns:
            Optional[pd.DataFrame]: The parsed DataFrame if successful, None otherwise.
        """
        success = self.parse_file()
        if success:
            try:
                df = pd.read_csv(self.output_file)
                return df
            except Exception as e:
                print(f"Error reading the output CSV: {e}")
                return None
        else:
            return None


# -----------------------------------------------------------------------------
# Asynchronous Node for the Graph (wraps QTRAP_Parse)
# -----------------------------------------------------------------------------
def parse_chromatogram_node(state: QTRAPState, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    """
    Asynchronous node to parse a chromatogram file using QTRAP_Parse.

    This function retrieves file paths from the custom state, performs parsing, and updates the state.
    """
    configuration = Configuration.from_runnable_config(config)

    # Use file paths from the custom state
    input_file = state.input_file
    output_file = state.output_file

    parser = QTRAP_Parse(input_file, output_file)
    df = parser.run()  # Synchronous call
    if df is not None:
        content = f"Parsing successful: DataFrame with {len(df)} records saved to {output_file}."
    else:
        content = "Parsing failed."
    
    # Update the custom state with the parsing result
    state.parsing_result = content

    # Create and return an AIMessage as the node's output
    message = AIMessage(id="parse_result", content=content)
    return {"messages": [message]}


# -----------------------------------------------------------------------------
# Build the ReAct-style Graph using the custom state
# -----------------------------------------------------------------------------
builder = StateGraph(QTRAPState, input=QTRAPState, config_schema=Configuration)

# Add the parsing node and a tool node (to allow for tool calls if needed)
builder.add_node(parse_chromatogram_node)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as the parsing node
builder.add_edge("__start__", "parse_chromatogram_node")

def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """
    Determine the next node based on the most recent AIMessage in state's messages.
    """
    # Iterate backwards through messages to find the last AIMessage
    last_ai_message = next((msg for msg in reversed(state.messages) if isinstance(msg, AIMessage)), None)
    if last_ai_message is None:
        raise ValueError("No AIMessage found in state messages for routing.")
    if not last_ai_message.tool_calls:
        return "__end__"
    return "tools"

# Add conditional routing based on the output of the parsing node
builder.add_conditional_edges("parse_chromatogram_node", route_model_output)

# Cycle back from the tools node to the parsing node (if tool calls are made)
builder.add_edge("tools", "parse_chromatogram_node")

# Compile the graph into an executable object
graph = builder.compile(
    interrupt_before=[],  # Insert node names here if state updates are needed before nodes are called
    interrupt_after=[]    # Insert node names here if state updates are needed after nodes are called
)
graph.name = "QTRAP Parsing Agent with Custom State"
