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



######################
#WORKLIST GENERATOR


# scripts/worklist_generator.py

import pandas as pd
import os
import json
from typing import Dict, Any

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


# Example usage

from scripts.worklist_generator import WorklistGenerator as WGObject
ms_object = WGObject(input_file='qc_worklist/input_worklist.csv',output_file='qc_worklist/generated_worklist_AI.txt' )
object_driver_microservice = object_to_microservice(object=ms_object)
ms_agent_no_mem = create_linqx_chat_agent(
    microservice=object_driver_microservice,
    llm=ChatOpenAI(model='gpt-4', temperature=0),
    use_memory=None,
    human_interaction=True,
    verbose=True
)
output = ms_agent_no_mem.invoke({'input': 'generate worklist'})



###########################

##########################
#PLOT AGENT
########################

import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Any

class Plot:
    def __init__(self, input_csv: str, output_dir: str, sorting: str = 'Lipid'):
        """
        Initialize the Plot class with input CSV, output directory, and sorting criterion.

        Args:
            input_csv (str): Path to the CSV file containing parsed chromatogram data.
            output_dir (str): Directory to save the plots.
            sorting (str): Sorting criterion, either 'Lipid' or 'Intensity'. Default is 'Lipid'.
        """
        self._input_csv = input_csv
        self._output_dir = output_dir
        self._sorting = sorting
        self._df = None
        
    @property
    def input_csv(self) -> str:
        return self._input_csv
        
    @property
    def output_dir(self) -> str:
        return self._output_dir
        
    @property
    def sorting(self) -> str:
        return self._sorting

    def preprocess_data(self) -> Dict[str, Any]:
        """
        Preprocess the input CSV data.
        
        Returns:
            Dict[str, Any]: Status of preprocessing operation
        """
        try:
            # Load the CSV file into a DataFrame
            self._df = pd.read_csv(self._input_csv)

            # Rename 'Summed_Intensity' to 'Intensity'
            self._df.rename(columns={'Summed_Intensity': 'Intensity'}, inplace=True)

            # Handle multiple lipids in the 'Lipid' column
            self._df['Lipid'] = self._df['Lipid'].str.replace(r'^"+|"+$', '', regex=True)
            self._df['Lipid'] = self._df['Lipid'].str.split(',')

            # Explode the DataFrame so each lipid has its own row
            self._df = self._df.explode('Lipid')

            # Strip any leading/trailing whitespace from lipid names
            self._df['Lipid'] = self._df['Lipid'].str.strip()

            return {
                "status": "success",
                "message": "Data preprocessing completed successfully",
                "rows_processed": len(self._df)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during preprocessing: {str(e)}",
                "rows_processed": 0
            }

    def plot_lipid_intensities(self) -> Dict[str, Any]:
        """
        Create bar plots for lipid intensities.
        
        Returns:
            Dict[str, Any]: Summary of the plotting operation
        """
        try:
            # Initialize counters
            total_plots = 0
            processed_files = 0

            # Ensure the output directory exists
            os.makedirs(self._output_dir, exist_ok=True)

            # Preprocess the data if not already done
            if self._df is None:
                preprocess_result = self.preprocess_data()
                if preprocess_result["status"] == "error":
                    return preprocess_result

            # Group by Filename
            grouped = self._df.groupby('Filename')

            for filename, group in tqdm(grouped, desc="Processing files"):
                processed_files += 1
                file_dir = os.path.join(self._output_dir, filename)
                os.makedirs(file_dir, exist_ok=True)

                # Sort the entire group by Intensity in descending order before chunking
                group = group.sort_values(by='Intensity', ascending=False)

                # Split into chunks of 10 (ensuring highest Intensity values are in Chunk 1)
                lipid_chunks = [group.iloc[i:i + 10] for i in range(0, len(group), 10)]

                for i, chunk in enumerate(lipid_chunks):
                    total_plots += 1

                    plt.figure(figsize=(10, 6))
                    plt.bar(chunk['Lipid'], chunk['Intensity'], alpha=0.7, color='blue')
                    plt.title(f"Lipid Intensities - {filename} (Chunk {i + 1})")
                    plt.xlabel('Lipid')
                    plt.ylabel('Intensity')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    plot_file = os.path.join(file_dir, f"{filename}_chunk_{i + 1}_lipid_intensities.png")
                    plt.savefig(plot_file)
                    plt.close()

            return {
                "status": "success",
                "message": "Plot generation completed successfully",
                "plots_generated": total_plots,
                "files_processed": processed_files,
                "output_directory": os.path.abspath(self._output_dir)
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during plot generation: {str(e)}",
                "plots_generated": total_plots,
                "files_processed": processed_files
            }


# Example usage:
if __name__ == "__main__":
    plotter = Plot(input_csv='data.csv', output_dir='plots', sorting='Lipid')
    preprocess_result = plotter.preprocess_data()
    print("Preprocessing result:", preprocess_result)
    
    plot_result = plotter.plot_lipid_intensities()
    print("Plotting result:", plot_result)


# Example usage
from scripts.QTRAP_plot import Plot
plot = Plot(input_csv='qc_result/done.csv', output_dir='plots')
object_driver_microservice = object_to_microservice(object=plot)
ms_agent_no_mem = create_linqx_chat_agent(
    microservice=object_driver_microservice,
    llm=ChatOpenAI(model='gpt-4', temperature=0),
    use_memory=None,
    human_interaction=True,
    verbose=True
)

output = ms_agent_no_mem.invoke({'input': 'plot lipid intensities of qc_result/done.csv '})