import re
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, Dict, Any

# =============================================================================
# Original classes (rewritten for clarity) 
# =============================================================================

class QTRAP_Parse:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file

        self.filenames = []
        self.q1_values = []
        self.q3_values = []
        self.lipids = []
        self.dates = []
        self.sample_names = []
        self.samples = []
        self.summed_intensities = []

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
        file_path = self.input_file
        print(f"Parsing file: {file_path}")

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
            print("Warning: Filename does not contain an underscore.")

        sample = 'Blank' if 'Blank' in sample_name else 'Sample'
        print(f"Determined sample type: {sample}")

        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            print(f"Successfully read file: {file_path}")
            print(f"Total lines read: {len(lines)}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return False

        current_filename = ""
        current_q1 = None
        current_q3 = None
        current_lipid = ""
        parsing_intensity = False

        for line_num, line in enumerate(lines):
            if 'sourceFile:' in line or 'name:' in line:
                match = re.search(r'name:\s+([\w.]+)', line)
                if match:
                    current_filename = match.group(1)
                    print(f"Extracted current_filename at line {line_num}: {current_filename}")
            if 'id: SRM SIC Q1=' in line:
                match = re.search(r'Q1=(\d+\.\d+).*Q3=(\d+\.\d+).*name=([^\s]+)', line)
                if match:
                    current_q1 = float(match.group(1))
                    current_q3 = float(match.group(2))
                    current_lipid = match.group(3)
                    print(f"Extracted Q1: {current_q1}, Q3: {current_q3}, Lipid: {current_lipid} at line {line_num}")
            if 'cvParam: intensity array' in line:
                parsing_intensity = True
                intensities = []
                print(f"'cvParam: intensity array' found at line {line_num}")
            elif parsing_intensity and 'binary: [' in line:
                match = re.search(r'binary:\s+\[\d+\]\s+([\d\s]+)', line)
                if match:
                    try:
                        intensities = list(map(int, match.group(1).split()))
                        current_intensity_sum = sum(intensities)
                        print(f"Extracted intensity data for lipid {current_lipid} at line {line_num}: Sum={current_intensity_sum}")
                    except ValueError:
                        intensities = []
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

        if not self.filenames:
            print(f"No data parsed from {file_path}")
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
        if self.parse_file():
            try:
                df = pd.read_csv(self.output_file)
                return df
            except Exception as e:
                print(f"Error reading the output CSV: {e}")
                return None
        else:
            return None

def process_chromatogram(input_file: str, output_file: str) -> str:
    parser = QTRAP_Parse(input_file, output_file)
    success = parser.run()
    return output_file if success else "not parsed"

class WorklistGenerator:
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
        try:
            input_df = pd.read_csv(self.input_file)
            print(f"[DEBUG] Successfully loaded input file: {self.input_file}")
        except Exception as e:
            print(f"[ERROR] Issue loading input file: {e}")
            raise

        worklist_df = pd.DataFrame(columns=self.column_headers)
        try:
            worklist_df['SampleName'] = input_df['SampleName']
            worklist_df['AcqMethod'] = input_df['Method']
            print("[DEBUG] 'SampleName' and 'AcqMethod' columns populated.")
        except KeyError as e:
            print(f"[ERROR] Missing required column: {e}")
            raise

        for col, default_value in self.default_values.items():
            worklist_df[col] = default_value
        print("[DEBUG] Default values filled for remaining columns.")

        try:
            with open(self.output_file, 'w') as f:
                f.write('% header=' + '\t'.join(self.column_headers) + '\n')
                worklist_df.to_csv(f, sep='\t', index=False, header=False)
            print(f"[DEBUG] Worklist saved to: {os.path.abspath(self.output_file)}")
        except Exception as e:
            print(f"[ERROR] Error writing worklist: {e}")
            raise

        return worklist_df

class Plot:
    def __init__(self, input_csv: str, output_dir: str, sorting: str = 'Lipid'):
        self._input_csv = input_csv
        self._output_dir = output_dir
        self._sorting = sorting
        self._df = None

    def preprocess_data(self) -> Dict[str, Any]:
        try:
            self._df = pd.read_csv(self._input_csv)
            self._df.rename(columns={'Summed_Intensity': 'Intensity'}, inplace=True)
            self._df['Lipid'] = self._df['Lipid'].str.replace(r'^"+|"+$', '', regex=True)
            self._df['Lipid'] = self._df['Lipid'].str.split(',')
            self._df = self._df.explode('Lipid')
            self._df['Lipid'] = self._df['Lipid'].str.strip()
            return {
                "status": "success",
                "message": "Data preprocessing completed",
                "rows_processed": len(self._df)
            }
        except Exception as e:
            return {"status": "error", "message": f"Error during preprocessing: {e}", "rows_processed": 0}

    def plot_lipid_intensities(self) -> Dict[str, Any]:
        try:
            total_plots = 0
            processed_files = 0
            os.makedirs(self._output_dir, exist_ok=True)
            if self._df is None:
                preprocess_result = self.preprocess_data()
                if preprocess_result["status"] == "error":
                    return preprocess_result
            grouped = self._df.groupby('Filename')
            for filename, group in tqdm(grouped, desc="Processing files"):
                processed_files += 1
                file_dir = os.path.join(self._output_dir, filename)
                os.makedirs(file_dir, exist_ok=True)
                group = group.sort_values(by='Intensity', ascending=False)
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
                "message": "Plot generation completed",
                "plots_generated": total_plots,
                "files_processed": processed_files,
                "output_directory": os.path.abspath(self._output_dir)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error during plot generation: {e}",
                "plots_generated": total_plots,
                "files_processed": processed_files
            }

# =============================================================================
# LangGraph Node Definitions and Workflow Assembly
# =============================================================================
# (Assume that “langgraph” is a framework that provides a Node and Graph API.
#  The following classes wrap the above functionality as nodes.)

# For this example, we define a simple Node base class and Graph class.
# In your actual implementation, replace these with the real langgraph classes.

class Node:
    def __init__(self, name: str = ""):
        self.name = name

    def run(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        raise NotImplementedError("Each node must implement its run method.")

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node: Node, name: str):
        node.name = name
        self.nodes[name] = node

    def connect(self, from_node: str, to_node: str, description: str = ""):
        # Here, we simply store the edge connection.
        # In a full implementation, you would use the edge mapping to pass outputs as inputs.
        self.edges.append({"from": from_node, "to": to_node, "description": description})
        print(f"Connected node '{from_node}' to node '{to_node}': {description}")

    def execute(self, initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        results = {}
        # For simplicity, we assume that nodes run in the order they were added.
        for node_name, node in self.nodes.items():
            print(f"Executing node: {node_name}")
            result = node.run(initial_data)
            results[node_name] = result
        return results

# Node wrapping for QTRAP parsing
class QTRAPParserNode(Node):
    def __init__(self, input_file: str, output_file: str, name: str = "QTRAP Parser"):
        super().__init__(name)
        self.input_file = input_file
        self.output_file = output_file

    def run(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        print(f"[{self.name}] Running with input file: {self.input_file}")
        parsed_file = process_chromatogram(self.input_file, self.output_file)
        return {"parse_success": parsed_file != "not parsed", "parsed_file": parsed_file}

# Node wrapping for Worklist Generation
class WorklistGeneratorNode(Node):
    def __init__(self, input_file: str, output_file: str, name: str = "Worklist Generator"):
        super().__init__(name)
        self.input_file = input_file
        self.output_file = output_file

    def run(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        print(f"[{self.name}] Running with input file: {self.input_file}")
        wg = WorklistGenerator(self.input_file, self.output_file)
        worklist_df = wg.generate_worklist()
        return {"worklist_df": worklist_df, "worklist_file": self.output_file}

# Node wrapping for Plot Generation
class PlotNode(Node):
    def __init__(self, input_csv: str, output_dir: str, sorting: str = 'Lipid', name: str = "Plot Generator"):
        super().__init__(name)
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.sorting = sorting

    def run(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        print(f"[{self.name}] Running with input CSV: {self.input_csv}")
        plotter = Plot(self.input_csv, self.output_dir, self.sorting)
        result = plotter.plot_lipid_intensities()
        return result

# =============================================================================
# Build the workflow graph and define node connections (edges)
# =============================================================================

def build_workflow() -> Graph:
    graph = Graph()

    # Instantiate nodes with the appropriate file paths
    parser_node = QTRAPParserNode(
        input_file='qc_txt_files/20241115_Plasma_PC.txt',
        output_file='qc_result/test_AI.csv'
    )
    worklist_node = WorklistGeneratorNode(
        input_file='qc_worklist/input_worklist.csv',
        output_file='qc_worklist/generated_worklist_AI.txt'
    )
    plot_node = PlotNode(
        input_csv='qc_result/done.csv',
        output_dir='plots',
        sorting='Lipid'
    )

    # Add nodes to the graph with friendly names
    graph.add_node(parser_node, name='QTRAP Parser')
    graph.add_node(worklist_node, name='Worklist Generator')
    graph.add_node(plot_node, name='Plot Generator')

    # Create edges:
    # Here we show that after parsing, we want to trigger both the worklist generation and plot generation.
    # In a more advanced implementation, you might map the parsed CSV (output from the parser) to be used as input for the other nodes.
    graph.connect('QTRAP Parser', 'Worklist Generator', description='Trigger worklist generation after parsing')
    graph.connect('QTRAP Parser', 'Plot Generator', description='Trigger plot generation after parsing')

    return graph

# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    workflow = build_workflow()
    results = workflow.execute(initial_data={})
    print("\nWorkflow execution results:")
    for node_name, result in results.items():
        print(f"{node_name}: {result}")
