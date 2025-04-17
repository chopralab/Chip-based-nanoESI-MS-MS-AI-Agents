# scripts/parsing.py

import re
import os
import pandas as pd
from typing import Optional

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

        # Define lipid classes and their abbreviations
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
        base_name = os.path.splitext(os.path.basename(file_path))[0]  # Removes the .txt extension
        parts = base_name.split('_', 1)                             # Split only on the first underscore
        if len(parts) == 2:
            date_str = parts[0]                                     # e.g., '20241115'
            sample_name = parts[1]                                  # e.g., 'Plasma_Acyl-Carnitines'
            print(f"Extracted date: {date_str}")
            print(f"Extracted sample name: {sample_name}")
        else:
            # Handle unexpected filename formats
            date_str = ''
            sample_name = ''
            print("Warning: Filename does not contain an underscore. Date and Sample_Name set to empty strings.")

        # Determine the Sample value based on Sample_Name
        sample = 'Blank' if 'Blank' in sample_name else 'Sample'
        print(f"Determined sample type: {sample}")

        # Open and read all lines of the file
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            print(f"Successfully read file: {file_path}")
            print(f"Total lines read: {len(lines)}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return False  # Parsing failed


        # -------------------
        # Parse Lipid Data
        # -------------------
        current_filename = ""
        current_q1 = None
        current_q3 = None
        current_lipid = ""
        parsing_intensity = False

        for line_num, line in enumerate(lines):
            # Extract the filename
            if 'sourceFile:' in line or 'name:' in line:
                match = re.search(r'name:\s+([\w.]+)', line)
                if match:
                    current_filename = match.group(1)
                    print(f"Extracted current_filename at line {line_num}: {current_filename}")
                else:
                    print(f"No match found for 'name' at line {line_num}: {line.strip()}")

            # Extract Q1, Q3 values, and Lipid name
            if 'id: SRM SIC Q1=' in line:
                match = re.search(r'Q1=(\d+\.\d+).*Q3=(\d+\.\d+).*name=([^\s]+)', line)
                if match:
                    current_q1 = float(match.group(1))
                    current_q3 = float(match.group(2))
                    current_lipid = match.group(3)
                    print(f"Extracted Q1: {current_q1}, Q3: {current_q3}, Lipid: {current_lipid} at line {line_num}")
                else:
                    print(f"No match found for Q1/Q3/Lipid at line {line_num}: {line.strip()}")

            # Check if we are parsing intensity array data for lipids
            if 'cvParam: intensity array' in line:
                parsing_intensity = True
                intensities = []
                print(f"'cvParam: intensity array' found at line {line_num}")
            elif parsing_intensity and 'binary: [' in line:
                # Extract intensity values
                match = re.search(r'binary:\s+\[\d+\]\s+([\d\s]+)', line)
                if match:
                    try:
                        intensities = list(map(int, match.group(1).split()))
                        current_intensity_sum = sum(intensities)
                        print(f"Extracted intensity data for lipid {current_lipid} at line {line_num}: Sum={current_intensity_sum}")
                    except ValueError:
                        intensities = []
                        current_intensity_sum = 0
                        print(f"Failed to parse intensity data for lipid {current_lipid} at line {line_num}: {line.strip()}")

                    # Append the extracted and calculated data to the lists
                    if current_filename and current_q1 is not None and current_q3 is not None:
                        self.filenames.append(current_filename)
                        self.q1_values.append(current_q1)
                        self.q3_values.append(current_q3)
                        self.lipids.append(current_lipid)
                        self.dates.append(date_str)                        # Append Date
                        self.sample_names.append(sample_name)              # Append Sample_Name
                        self.samples.append(sample)                        # Append Sample
                        self.summed_intensities.append(current_intensity_sum)  # Append Summed_Intensity
                        print(f"Appended data for lipid {current_lipid}: Q1={current_q1}, Q3={current_q3}, Sum_Intensity={current_intensity_sum}")
                    else:
                        print(f"Missing required fields for lipid {current_lipid} at line {line_num}")

                else:
                    print(f"No match found for lipid intensity data at line {line_num}: {line.strip()}")

                # Reset parsing flag
                parsing_intensity = False
            elif parsing_intensity:
                # If parsing was expected but 'binary: [' not found, reset flag
                parsing_intensity = False
                print(f"Unexpected line while parsing intensities at line {line_num}: {line.strip()}")

        # Check if any data was parsed
        if not self.filenames:
            print(f"No data parsed from {file_path}")
            return False

        # Create a DataFrame from the parsed data
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

        # Save the DataFrame to CSV
        try:
            df.to_csv(self.output_file, index=False)
            print(f"Data successfully saved to {self.output_file}")
            self.parsed_df = df  # Store the DataFrame as an instance attribute
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

def process_chromatogram(input_file: str, output_file: str) -> str:
    """
    Process a chromatogram .txt file and save the parsed data to a CSV file.

    Args:
        input_file (str): Path to the .txt file to parse.
        output_file (str): Path where the CSV will be saved.

    Returns:
        str: Output file path if successful, otherwise "not parsed".
    """
    parser = QTRAP_Parse(input_file, output_file)
    success = parser.run()
    return output_file if success else "not parsed"


########

import os
import sys

sys.path.insert(1, "/home/sanjay/QTRAP_memory") 

from langchain_openai import ChatOpenAI
from sciborg_dev.ai.agents.core import create_linqx_chat_agent
from sciborg_dev.ai.agents.core2 import SciborgAgent
from sciborg_dev.ai.chains.microservice import module_to_microservice
from sciborg_dev.ai.chains.workflow import create_workflow_planner_chain, create_workflow_constructor_chain

from sciborg_dev.testing.models.drivers import MicrowaveSynthesizer, MicrowaveSynthesizerObject, PubChemCaller
from sciborg_dev.core.library.base import BaseDriverMicroservice

from scripts.parsing import QTRAP_Parse
from scripts.parsing import process_chromatogram

ms_object = QTRAP_Parse( input_file='qc_txt_files/20241115_Plasma_PC.txt', output_file='qc_result/test_AI.csv')
object_driver_microservice = object_to_microservice(object=ms_object)
