#!/usr/bin/env python3
"""
Q_QC_TIC.py - TIC Data Extraction and Plotting Module
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import asyncio
import pandas as pd
import csv

class TICExtractor:
    def __init__(self, source_dir: str, target_dir: str, project_name: str = ""):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.project_name = project_name
        self.logger = logging.getLogger(f"tic_extractor_{project_name}")
        
        # Create main target directory and subdirectories
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.chromatograms_dir = self.target_dir / "chromatograms"
        self.hidden_chromatograms_dir = self.target_dir / "hidden_chromatograms"
        self.chromatograms_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_chromatograms_dir.mkdir(parents=True, exist_ok=True)
        
        # Reference CSV for filename-number mapping
        self.reference_csv_path = self.target_dir / f"{project_name}_reference.csv"
        
        # Load existing reference mapping or create new one
        self.filename_to_number = {}
        self.filename_to_result = {}  # Store QC results (pass/fail)
        self.filename_to_threshold = {}  # Store threshold values
        self.filename_to_result_data = {}  # Store actual TIC RSD values
        self.next_number = 1
        self._load_reference_mapping()
        
        self.stats = {
            'files_processed': 0, 
            'files_failed': 0, 
            'plots_created': 0, 
            'hidden_plots_created': 0,
            'total_files': 0
        }
    
    def _load_reference_mapping(self):
        """Load existing filename-to-number mapping from CSV file."""
        if self.reference_csv_path.exists():
            try:
                with open(self.reference_csv_path, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        filename = row['Filename']
                        number = int(row['Number'])
                        result = row.get('Result', 'unknown')
                        threshold = row.get('Threshold', 'N/A')
                        result_data = row.get('Result_Data', 'N/A')
                        
                        self.filename_to_number[filename] = number
                        self.filename_to_result[filename] = result
                        self.filename_to_threshold[filename] = threshold
                        self.filename_to_result_data[filename] = result_data
                        self.next_number = max(self.next_number, number + 1)
                self.logger.debug(f"Loaded {len(self.filename_to_number)} filename mappings from reference CSV")
            except Exception as e:
                self.logger.warning(f"Error loading reference mapping: {e}")
    
    def _save_reference_mapping(self):
        """Save filename-to-number mapping with QC results, threshold, and TIC RSD data to CSV file."""
        try:
            with open(self.reference_csv_path, 'w', newline='') as csvfile:
                fieldnames = ['Filename', 'Number', 'Result', 'Threshold', 'Result_Data']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for filename, number in sorted(self.filename_to_number.items(), key=lambda x: x[1]):
                    result = self.filename_to_result.get(filename, 'unknown')
                    threshold = self.filename_to_threshold.get(filename, 'N/A')
                    result_data = self.filename_to_result_data.get(filename, 'N/A')
                    writer.writerow({
                        'Filename': filename, 
                        'Number': number,
                        'Result': result,
                        'Threshold': threshold,
                        'Result_Data': result_data
                    })
            self.logger.debug(f"Saved {len(self.filename_to_number)} filename mappings with QC data to reference CSV")
        except Exception as e:
            self.logger.error(f"Error saving reference mapping: {e}")
    
    def _get_or_assign_number(self, filename: str) -> int:
        """Get existing number for filename or assign a new one."""
        if filename not in self.filename_to_number:
            self.filename_to_number[filename] = self.next_number
            self.next_number += 1
        return self.filename_to_number[filename]
    
    def _load_qc_results(self):
        """Load QC results from the QC results CSV file."""
        qc_results_path = Path(f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results/{self.project_name}/QC_{self.project_name}_RESULTS.csv")
        
        if not qc_results_path.exists():
            self.logger.warning(f"QC results file not found: {qc_results_path}")
            return
        
        try:
            qc_df = pd.read_csv(qc_results_path)
            self.logger.debug(f"Loaded QC results from: {qc_results_path}")
            
            # Get the threshold from the Q_QC module (25.0)
            threshold_value = "25.0"
            
            # Create mapping from filename to QC result, threshold, and TIC RSD data
            for _, row in qc_df.iterrows():
                filename = str(row['Filename'])
                qc_result = str(row.get('QC_Result', 'unknown')).lower()
                tic_rsd_value = row.get('TIC_RSD_TopGroupWindow', None)
                
                # Format TIC RSD value
                if pd.isna(tic_rsd_value) or tic_rsd_value == '':
                    tic_rsd_formatted = 'N/A'
                else:
                    try:
                        tic_rsd_formatted = f"{float(tic_rsd_value):.2f}"
                    except (ValueError, TypeError):
                        tic_rsd_formatted = 'N/A'
                
                # Remove file extensions to match our base filename format
                base_filename = filename.replace('.dam.txt', '').replace('.txt', '').replace('.dam', '')
                
                self.filename_to_result[base_filename] = qc_result
                self.filename_to_threshold[base_filename] = threshold_value
                self.filename_to_result_data[base_filename] = tic_rsd_formatted
                
            self.logger.debug(f"Loaded QC results for {len(self.filename_to_result)} files")
            
        except Exception as e:
            self.logger.error(f"Error loading QC results: {e}")
    
    def extract_tic_data(self, file_path: Path) -> Optional[Tuple[List[float], List[float]]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find TIC section
            tic_pattern = r'id:\s*TIC.*?(?=chromatogram:|$)'
            tic_match = re.search(tic_pattern, content, re.DOTALL | re.IGNORECASE)
            if not tic_match:
                return None
            
            tic_section = tic_match.group(0)
            
            # Extract time and intensity arrays
            time_pattern = r'cvParam:\s*time array.*?binary:\s*\[(\d+)\]\s*([0-9e\-\.\s]+)'
            intensity_pattern = r'cvParam:\s*intensity array.*?binary:\s*\[(\d+)\]\s*([0-9\s]+)'
            
            time_match = re.search(time_pattern, tic_section, re.DOTALL | re.IGNORECASE)
            intensity_match = re.search(intensity_pattern, tic_section, re.DOTALL | re.IGNORECASE)
            
            if not time_match or not intensity_match:
                return None
            
            # Parse data
            time_count = int(time_match.group(1))
            time_data = time_match.group(2).strip().split()
            time_array = [float(x) for x in time_data[:time_count]]
            
            intensity_count = int(intensity_match.group(1))
            intensity_data = intensity_match.group(2).strip().split()
            intensity_array = [float(x) for x in intensity_data[:intensity_count]]
            
            # Ensure equal lengths
            min_len = min(len(time_array), len(intensity_array))
            return time_array[:min_len], intensity_array[:min_len]
            
        except Exception as e:
            self.logger.error(f"Error extracting TIC data from {file_path.name}: {e}")
            return None
    
    def create_tic_plot(self, time_array: List[float], intensity_array: List[float], 
                       filename: str, output_path: Path, title: str = None) -> bool:
        """Create TIC plot with customizable title."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(time_array, intensity_array, 'b-', linewidth=1.2)
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Intensity (detector counts)')
            
            # Use custom title if provided, otherwise use default format
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f'Total Ion Chromatogram (TIC)\n{filename}')
            
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating plot: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return False
    
    def create_hidden_chromatogram(self, time_array: List[float], intensity_array: List[float], 
                                  base_filename: str) -> bool:
        """Create anonymized version of chromatogram for bias-free review."""
        try:
            # Get or assign number for this filename
            number = self._get_or_assign_number(base_filename)
            
            # Create hidden chromatogram filename
            hidden_filename = f"{self.project_name}_{number}.png"
            hidden_output_path = self.hidden_chromatograms_dir / hidden_filename
            
            # Create plot with only the number as title
            success = self.create_tic_plot(
                time_array, 
                intensity_array, 
                "", 
                hidden_output_path, 
                title=str(number)
            )
            
            if success:
                self.stats['hidden_plots_created'] += 1
                self.logger.debug(f"Created hidden chromatogram: {hidden_filename}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error creating hidden chromatogram for {base_filename}: {e}")
            return False
    
    def process_all_files(self) -> Dict[str, Any]:
        xml_files = list(self.source_dir.glob('*.txt'))
        self.stats['total_files'] = len(xml_files)
        
        self.logger.info(f"🔍 Processing {len(xml_files)} files for TIC extraction")
        
        # Load QC results to get pass/fail information
        self._load_qc_results()
        
        for file_path in xml_files:
            tic_data = self.extract_tic_data(file_path)
            if tic_data:
                time_array, intensity_array = tic_data
                base_name = file_path.stem.replace('.dam', '')
                
                # Get QC data for this file
                qc_result = self.filename_to_result.get(base_name, 'unknown')
                threshold = self.filename_to_threshold.get(base_name, '25.0')
                result_data = self.filename_to_result_data.get(base_name, 'N/A')
                
                # Create regular chromatogram in chromatograms subdirectory
                regular_output_path = self.chromatograms_dir / f"{base_name}_TIC.png"
                
                if self.create_tic_plot(time_array, intensity_array, file_path.name, regular_output_path):
                    self.stats['files_processed'] += 1
                    self.stats['plots_created'] += 1
                    
                    # Store all QC data for this filename
                    self.filename_to_result[base_name] = qc_result
                    self.filename_to_threshold[base_name] = threshold
                    self.filename_to_result_data[base_name] = result_data
                    
                    # Create hidden chromatogram for bias-free review
                    self.create_hidden_chromatogram(time_array, intensity_array, base_name)
                    
                else:
                    self.stats['files_failed'] += 1
            else:
                self.stats['files_failed'] += 1
        
        # Save the reference mapping to CSV (now includes QC results)
        self._save_reference_mapping()
        
        # Log summary
        self.logger.info(f"📊 TIC Processing Summary:")
        self.logger.info(f"   ✅ Regular plots: {self.stats['plots_created']}")
        self.logger.info(f"   🔒 Hidden plots: {self.stats['hidden_plots_created']}")
        self.logger.info(f"   ❌ Failed: {self.stats['files_failed']}")
        self.logger.info(f"   📋 Reference CSV (5 columns): {self.reference_csv_path}")
        self.logger.info(f"   📊 Columns: Filename, Number, Result, Threshold, Result_Data")
        
        return self.stats

async def generate_tic_plots_for_project(project_name: str) -> Dict[str, Any]:
    source_dir = f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/text/{project_name}"
    target_dir = f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/TIC/{project_name}"
    
    extractor = TICExtractor(source_dir, target_dir, project_name)
    return await asyncio.to_thread(extractor.process_all_files)

if __name__ == "__main__":
    import asyncio
    result = asyncio.run(generate_tic_plots_for_project("solventmatrix"))
    print(f"Processed: {result['files_processed']}, Failed: {result['files_failed']}")