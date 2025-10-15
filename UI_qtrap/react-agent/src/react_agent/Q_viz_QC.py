#!/usr/bin/env python3
"""
QC Visualization Script for QTRAP Pipeline

This script generates visualizations for QC pass/fail results from the QTRAP pipeline.
It creates dot plots showing pass rates by solvent matrix and exports summary statistics.

Author: QTRAP Pipeline Team
Date: 2025-10-15
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def extract_solvent_matrix_from_filename(filename: str) -> Optional[str]:
    """Extract solvent matrix (2nd underscore-delimited field) from filename.
    
    Parameters:
        filename: Sample filename string
        
    Returns:
        Solvent matrix identifier or None if extraction fails
        
    Example:
        >>> extract_solvent_matrix_from_filename("20250916_532MeOHIPAACN_BrainLipidEx_LC-PC_R-1_Op-TGL_Proj-solventmatrix_PC_withSPLASH.dam")
        '532MeOHIPAACN'
    """
    if not filename or not isinstance(filename, str):
        return None
    
    try:
        parts = filename.split('_')
        if len(parts) >= 2:
            return parts[1]
        else:
            logging.warning(f"Filename has unexpected format (less than 2 parts): {filename}")
            return None
    except Exception as e:
        logging.error(f"Error extracting solvent matrix from '{filename}': {e}")
        return None


def extract_project_name_from_filename(filename: str) -> Optional[str]:
    """Extract project name from 'Proj-{name}' pattern in filename.
    
    Parameters:
        filename: Sample filename string
        
    Returns:
        Project name or None if not found
        
    Example:
        >>> extract_project_name_from_filename("...Proj-solventmatrix_PC...")
        'solventmatrix'
    """
    if not filename or not isinstance(filename, str):
        return None
    
    try:
        match = re.search(r'Proj-([^_]+)', filename)
        if match:
            return match.group(1)
        else:
            return None
    except Exception as e:
        logging.error(f"Error extracting project name from '{filename}': {e}")
        return None


def filter_valid_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with BLANK in filename and invalid data.
    
    Parameters:
        df: Input DataFrame with QC results
        
    Returns:
        Filtered DataFrame with valid samples only
    """
    if df is None or df.empty:
        logging.warning("Input DataFrame is empty or None")
        return pd.DataFrame()
    
    initial_count = len(df)
    
    # Filter out rows with missing critical columns
    required_cols = ['QC_Result', 'Filename']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()
    
    # Remove rows with missing QC_Result or Filename
    df_filtered = df.dropna(subset=['QC_Result', 'Filename']).copy()
    
    # Filter out BLANK samples (case-insensitive)
    df_filtered = df_filtered[~df_filtered['Filename'].str.contains('BLANK', case=False, na=False)]
    
    removed_count = initial_count - len(df_filtered)
    if removed_count > 0:
        logging.info(f"Filtered out {removed_count} invalid/blank samples ({initial_count} → {len(df_filtered)})")
    
    return df_filtered


def calculate_passrate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate pass rate statistics for each solvent matrix.
    
    Parameters:
        df: DataFrame with 'SolventMatrix' and 'QC_Result' columns
        
    Returns:
        Summary DataFrame with columns:
        [SolventMatrix, Total_Samples, Passed, Failed, Pass_Rate_Percent]
    """
    if df is None or df.empty:
        logging.warning("Cannot calculate pass rate stats: DataFrame is empty")
        return pd.DataFrame()
    
    if 'SolventMatrix' not in df.columns or 'QC_Result' not in df.columns:
        logging.error("Required columns missing for pass rate calculation")
        return pd.DataFrame()
    
    try:
        # Create summary statistics manually
        stats_list = []
        
        for solvent in df['SolventMatrix'].unique():
            solvent_data = df[df['SolventMatrix'] == solvent]
            total = len(solvent_data)
            passed = (solvent_data['QC_Result'] == 'pass').sum()
            failed = (solvent_data['QC_Result'] == 'fail').sum()
            pass_rate = (passed / total) * 100 if total > 0 else 0
            
            stats_list.append({
                'SolventMatrix': solvent,
                'Total_Samples': total,
                'Passed': passed,
                'Failed': failed,
                'Pass_Rate_Percent': pass_rate
            })
        
        # Create DataFrame from list
        stats = pd.DataFrame(stats_list)
        
        # Sort by solvent matrix name
        stats = stats.sort_values('SolventMatrix')
        
        logging.info(f"Calculated pass rate stats for {len(stats)} solvent matrices")
        
        return stats
        
    except Exception as e:
        logging.error(f"Error calculating pass rate statistics: {e}")
        return pd.DataFrame()


def create_qc_passrate_dotplot(df: pd.DataFrame, output_dir: str, project_name: str):
    """Create dot plot showing QC pass/fail rates by solvent matrix.
    
    This function generates a visualization showing:
    - Individual pass/fail results as jittered dots (green=pass, red=fail)
    - Overall pass rate for each solvent matrix
    - Sample counts and statistics
    
    Parameters:
        df: QC results DataFrame with columns ['QC_Result', 'Filename']
        output_dir: Directory path for saving outputs
        project_name: Project identifier for file naming
        
    Outputs:
        - PNG: 'qc_passrate_by_solvent.png' (300 DPI)
        - PDF: 'qc_passrate_by_solvent.pdf'
        - CSV: 'qc_passrate_summary.csv'
    """
    if df is None or df.empty:
        logging.warning("Cannot create QC pass rate plot: DataFrame is empty")
        return
    
    logging.info("Creating QC pass rate dot plot...")
    
    try:
        # Step 1: Extract solvent matrix for each row
        df = df.copy()
        df['SolventMatrix'] = df['Filename'].apply(extract_solvent_matrix_from_filename)
        
        # Remove rows where solvent matrix extraction failed
        df = df.dropna(subset=['SolventMatrix'])
        
        if df.empty:
            logging.error("No valid solvent matrix data found after extraction")
            return
        
        # Step 2: Filter out BLANK samples
        df = filter_valid_samples(df)
        
        if df.empty:
            logging.error("No valid samples remaining after filtering")
            return
        
        # Step 3: Calculate pass rate statistics
        stats_df = calculate_passrate_stats(df)
        
        if stats_df.empty:
            logging.error("Failed to calculate pass rate statistics")
            return
        
        # Save summary CSV
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_path / 'qc_passrate_summary.csv'
        stats_df.to_csv(csv_path, index=False)
        logging.info(f"Pass rate summary CSV saved to: {csv_path}")
        
        # Step 4: Create dot plot
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
        
        # Get unique solvent matrices in sorted order
        solvent_matrices = stats_df['SolventMatrix'].values
        x_positions = np.arange(len(solvent_matrices))
        
        # Create mapping for x-axis positions
        solvent_to_x = {solvent: i for i, solvent in enumerate(solvent_matrices)}
        
        # Plot individual data points with jitter
        for solvent in solvent_matrices:
            solvent_data = df[df['SolventMatrix'] == solvent]
            x_pos = solvent_to_x[solvent]
            
            # Separate pass and fail
            pass_data = solvent_data[solvent_data['QC_Result'] == 'pass']
            fail_data = solvent_data[solvent_data['QC_Result'] == 'fail']
            
            # Add jitter to x-position for visibility
            jitter_amount = 0.15
            
            if len(pass_data) > 0:
                x_jitter_pass = x_pos + np.random.uniform(-jitter_amount, jitter_amount, len(pass_data))
                y_pass = np.ones(len(pass_data)) * 100  # Pass = 100%
                ax.scatter(x_jitter_pass, y_pass, color='green', s=80, alpha=0.6, 
                          edgecolors='darkgreen', linewidth=1, label='Pass' if solvent == solvent_matrices[0] else '', zorder=3)
            
            if len(fail_data) > 0:
                x_jitter_fail = x_pos + np.random.uniform(-jitter_amount, jitter_amount, len(fail_data))
                y_fail = np.zeros(len(fail_data))  # Fail = 0%
                ax.scatter(x_jitter_fail, y_fail, color='red', s=80, alpha=0.6, 
                          edgecolors='darkred', linewidth=1, label='Fail' if solvent == solvent_matrices[0] else '', zorder=3)
        
        # Overlay pass rate as larger markers
        ax.scatter(x_positions, stats_df['Pass_Rate_Percent'].values, 
                  color='blue', s=200, marker='D', edgecolors='black', linewidth=2,
                  label='Pass Rate', zorder=4, alpha=0.8)
        
        # Add horizontal line at 100% and 0%
        ax.axhline(y=100, color='green', linestyle='--', linewidth=1, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.3)
        
        # Add sample count annotations
        for i, (x_pos, row) in enumerate(zip(x_positions, stats_df.itertuples())):
            ax.text(x_pos, -10, f'n={row.Total_Samples}', 
                   ha='center', va='top', fontsize=10, fontweight='bold')
            
            # Add pass rate percentage above the diamond
            ax.text(x_pos, row.Pass_Rate_Percent + 5, f'{row.Pass_Rate_Percent:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color='blue')
        
        # Calculate overall pass rate
        overall_passed = stats_df['Passed'].sum()
        overall_total = stats_df['Total_Samples'].sum()
        overall_pass_rate = (overall_passed / overall_total) * 100 if overall_total > 0 else 0
        
        # Set labels and title
        ax.set_xlabel('Solvent Matrix', fontsize=16, fontweight='bold')
        ax.set_ylabel('Pass Rate (%)', fontsize=16, fontweight='bold')
        ax.set_title(f'QC Pass Rate by Solvent Matrix - {project_name}\n'
                    f'Overall: {overall_pass_rate:.1f}% ({overall_passed}/{overall_total} passed)',
                    fontsize=18, fontweight='bold', pad=20)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(solvent_matrices, rotation=45, ha='right', fontsize=12)
        ax.set_ylim(-15, 110)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save PNG
        png_path = output_path / 'qc_passrate_by_solvent.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        logging.info(f"PNG plot saved to: {png_path}")
        
        # Save PDF
        pdf_path = output_path / 'qc_passrate_by_solvent.pdf'
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        logging.info(f"PDF plot saved to: {pdf_path}")
        
        plt.close(fig)
        
        # Log summary statistics
        logging.info(f"QC Summary - Overall Pass Rate: {overall_pass_rate:.1f}% ({overall_passed}/{overall_total})")
        for row in stats_df.itertuples():
            logging.info(f"  {row.SolventMatrix}: {row.Pass_Rate_Percent:.1f}% ({row.Passed}/{row.Total_Samples})")
        
    except Exception as e:
        logging.error(f"Failed to create QC pass rate dot plot: {e}")
        import traceback
        traceback.print_exc()

def create_qc_diverging_bar_chart(df: pd.DataFrame, output_dir: str, project_name: str):
    """Create horizontal diverging bar chart showing pass/fail counts by solvent matrix.
    
    This function generates a butterfly-style visualization with:
    - Fail counts extending LEFT (red bars)
    - Pass counts extending RIGHT (green bars)
    - Center line at 0
    - Sample counts and pass rate annotations
    
    Parameters:
        df: QC results DataFrame with columns ['QC_Result', 'Filename']
        output_dir: Directory path for saving outputs
        project_name: Project identifier for file naming
        
    Outputs:
        - PNG: 'qc_diverging_bar_chart.png' (300 DPI)
        - PDF: 'qc_diverging_bar_chart.pdf'
    """
    if df is None or df.empty:
        logging.warning("Cannot create diverging bar chart: DataFrame is empty")
        return
    
    logging.info("Creating QC diverging bar chart...")
    
    try:
        # Step 1: Extract solvent matrix for each row
        df = df.copy()
        df['SolventMatrix'] = df['Filename'].apply(extract_solvent_matrix_from_filename)
        
        # Remove rows where solvent matrix extraction failed
        df = df.dropna(subset=['SolventMatrix'])
        
        if df.empty:
            logging.error("No valid solvent matrix data found after extraction")
            return
        
        # Step 2: Filter out BLANK samples
        df = filter_valid_samples(df)
        
        if df.empty:
            logging.error("No valid samples remaining after filtering")
            return
        
        # Step 3: Calculate pass rate statistics
        stats_df = calculate_passrate_stats(df)
        
        if stats_df.empty:
            logging.error("Failed to calculate pass rate statistics")
            return
        
        # Sort by pass rate (descending) for better visualization
        stats_df = stats_df.sort_values('Pass_Rate_Percent', ascending=True)
        
        # Step 4: Create diverging bar chart
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
        
        # Extract data
        solvents = stats_df['SolventMatrix'].values
        pass_counts = stats_df['Passed'].values
        fail_counts = stats_df['Failed'].values
        total_counts = stats_df['Total_Samples'].values
        pass_rates = stats_df['Pass_Rate_Percent'].values
        
        y_positions = np.arange(len(solvents))
        
        # Create bars
        # Fail bars extend LEFT (negative values)
        fail_bars = ax.barh(y_positions, -fail_counts, height=0.7, 
                           color='#d62728', edgecolor='darkred', linewidth=1.5,
                           label='Failed', alpha=0.85)
        
        # Pass bars extend RIGHT (positive values)
        pass_bars = ax.barh(y_positions, pass_counts, height=0.7,
                           color='#2ca02c', edgecolor='darkgreen', linewidth=1.5,
                           label='Passed', alpha=0.85)
        
        # Add center line at x=0
        ax.axvline(x=0, color='black', linewidth=2.5, zorder=3)
        
        # Add value labels on bars
        for i, (passed, failed, total, rate) in enumerate(zip(pass_counts, fail_counts, total_counts, pass_rates)):
            # Fail count label (on left side)
            if failed > 0:
                ax.text(-failed/2, i, str(int(failed)), 
                       ha='center', va='center', fontsize=11, fontweight='bold', 
                       color='white')
            
            # Pass count label (on right side)
            if passed > 0:
                ax.text(passed/2, i, str(int(passed)), 
                       ha='center', va='center', fontsize=11, fontweight='bold',
                       color='white')
            
            # Total sample count removed - visible from bar lengths
            
            # Pass rate percentage annotation (to the right of pass bar)
            x_pos = passed + 0.8
            color = 'green' if rate >= 80 else 'orange' if rate >= 60 else 'red'
            ax.text(x_pos, i, f'{rate:.1f}%', 
                   ha='left', va='center', fontsize=11, fontweight='bold',
                   color=color)
        
        # Customize axes
        ax.set_yticks(y_positions)
        ax.set_yticklabels(solvents, fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Samples', fontsize=14, fontweight='bold')
        ax.set_ylabel('Solvent Matrix', fontsize=14, fontweight='bold')
        
        # Set x-axis limits with some padding
        max_val = max(pass_counts.max(), fail_counts.max())
        ax.set_xlim(-max_val * 1.3, max_val * 1.3)
        
        # Customize x-axis to show absolute values
        x_ticks = ax.get_xticks()
        ax.set_xticklabels([str(int(abs(x))) for x in x_ticks], fontsize=11)
        
        # Simple FAIL and PASS labels without arrows
        ax.text(-max_val * 1.15, len(solvents) + 0.3, 'FAIL', 
               ha='center', va='bottom', fontsize=13, fontweight='bold',
               color='#d62728')
        ax.text(max_val * 1.15, len(solvents) + 0.3, 'PASS', 
               ha='center', va='bottom', fontsize=13, fontweight='bold',
               color='#2ca02c')
        
        # Calculate overall statistics
        overall_passed = pass_counts.sum()
        overall_total = total_counts.sum()
        overall_pass_rate = (overall_passed / overall_total) * 100 if overall_total > 0 else 0
        
        # Title removed for cleaner look
        
        # Add legend
        ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Grid on x-axis only
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save outputs
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save PNG
        png_path = output_path / 'qc_diverging_bar_chart.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        logging.info(f"PNG diverging bar chart saved to: {png_path}")
        
        # Save PDF
        pdf_path = output_path / 'qc_diverging_bar_chart.pdf'
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        logging.info(f"PDF diverging bar chart saved to: {pdf_path}")
        
        plt.close(fig)
        
        logging.info("Diverging bar chart created successfully!")
        
    except Exception as e:
        logging.error(f"Failed to create diverging bar chart: {e}")
        import traceback
        traceback.print_exc()


def create_qc_diverging_bar_chart_compact(df: pd.DataFrame, output_dir: str, project_name: str):
    """Create compact horizontal diverging bar chart with fixed x-axis range.
    
    This is a more concise version with x-axis limited to ±10 to reduce whitespace.
    
    Parameters:
        df: QC results DataFrame with columns ['QC_Result', 'Filename']
        output_dir: Directory path for saving outputs
        project_name: Project identifier for file naming
        
    Outputs:
        - PNG: 'qc_diverging_bar_chart_compact.png' (300 DPI)
        - PDF: 'qc_diverging_bar_chart_compact.pdf'
    """
    if df is None or df.empty:
        logging.warning("Cannot create compact diverging bar chart: DataFrame is empty")
        return
    
    logging.info("Creating compact QC diverging bar chart...")
    
    try:
        # Step 1: Extract solvent matrix for each row
        df = df.copy()
        df['SolventMatrix'] = df['Filename'].apply(extract_solvent_matrix_from_filename)
        
        # Remove rows where solvent matrix extraction failed
        df = df.dropna(subset=['SolventMatrix'])
        
        if df.empty:
            logging.error("No valid solvent matrix data found after extraction")
            return
        
        # Step 2: Filter out BLANK samples
        df = filter_valid_samples(df)
        
        if df.empty:
            logging.error("No valid samples remaining after filtering")
            return
        
        # Step 3: Calculate pass rate statistics
        stats_df = calculate_passrate_stats(df)
        
        if stats_df.empty:
            logging.error("Failed to calculate pass rate statistics")
            return
        
        # Sort by pass rate (descending) for better visualization
        stats_df = stats_df.sort_values('Pass_Rate_Percent', ascending=True)
        
        # Step 4: Create compact diverging bar chart
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        
        # Extract data
        solvents = stats_df['SolventMatrix'].values
        pass_counts = stats_df['Passed'].values
        fail_counts = stats_df['Failed'].values
        total_counts = stats_df['Total_Samples'].values
        pass_rates = stats_df['Pass_Rate_Percent'].values
        
        y_positions = np.arange(len(solvents))
        
        # Create bars
        # Fail bars extend LEFT (negative values)
        fail_bars = ax.barh(y_positions, -fail_counts, height=0.7, 
                           color='#d62728', edgecolor='darkred', linewidth=1.5,
                           label='Failed', alpha=0.85)
        
        # Pass bars extend RIGHT (positive values)
        pass_bars = ax.barh(y_positions, pass_counts, height=0.7,
                           color='#2ca02c', edgecolor='darkgreen', linewidth=1.5,
                           label='Passed', alpha=0.85)
        
        # Add center line at x=0
        ax.axvline(x=0, color='black', linewidth=2.5, zorder=3)
        
        # Add value labels on bars
        for i, (passed, failed, total, rate) in enumerate(zip(pass_counts, fail_counts, total_counts, pass_rates)):
            # Fail count label (on left side)
            if failed > 0:
                ax.text(-failed/2, i, str(int(failed)), 
                       ha='center', va='center', fontsize=11, fontweight='bold', 
                       color='white')
            
            # Pass count label (on right side)
            if passed > 0:
                ax.text(passed/2, i, str(int(passed)), 
                       ha='center', va='center', fontsize=11, fontweight='bold',
                       color='white')
            
            # Pass rate percentage annotation (to the right of pass bar, adjusted for compact view)
            x_pos = 10.5  # Fixed position outside the plot area
            color = 'green' if rate >= 80 else 'orange' if rate >= 60 else 'red'
            ax.text(x_pos, i, f'{rate:.1f}%', 
                   ha='left', va='center', fontsize=11, fontweight='bold',
                   color=color)
        
        # Customize axes
        ax.set_yticks(y_positions)
        ax.set_yticklabels(solvents, fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Samples', fontsize=14, fontweight='bold')
        ax.set_ylabel('Solvent Matrix', fontsize=14, fontweight='bold')
        
        # Set FIXED x-axis limits for compact view (±10)
        ax.set_xlim(-10, 10)
        
        # Customize x-axis to show absolute values
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_xticklabels(['10', '5', '0', '5', '10'], fontsize=11)
        
        # Simple FAIL and PASS labels without arrows
        ax.text(-8, len(solvents) + 0.3, 'FAIL', 
               ha='center', va='bottom', fontsize=13, fontweight='bold',
               color='#d62728')
        ax.text(8, len(solvents) + 0.3, 'PASS', 
               ha='center', va='bottom', fontsize=13, fontweight='bold',
               color='#2ca02c')
        
        # Add legend
        ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Grid on x-axis only
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save outputs with different filename
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save PNG
        png_path = output_path / 'qc_diverging_bar_chart_compact.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        logging.info(f"PNG compact diverging bar chart saved to: {png_path}")
        
        # Save PDF
        pdf_path = output_path / 'qc_diverging_bar_chart_compact.pdf'
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        logging.info(f"PDF compact diverging bar chart saved to: {pdf_path}")
        
        plt.close(fig)
        
        logging.info("Compact diverging bar chart created successfully!")
        
    except Exception as e:
        logging.error(f"Failed to create compact diverging bar chart: {e}")
        import traceback
        traceback.print_exc()


def visualize_qc_results(csv_file_path: str, project_name: str, output_base_dir: Optional[str] = None):
    """Main entry point for QC visualization.
    
    Parameters:
        csv_file_path: Path to input QC results CSV file
        project_name: Project identifier (e.g., 'solventmatrix')
        output_base_dir: Base output directory (default: data/viz/QC)
    """
    logging.info(f"Starting QC visualization for project: {project_name}")
    logging.info(f"Input CSV: {csv_file_path}")
    
    # Validate input file exists
    csv_path = Path(csv_file_path)
    if not csv_path.exists():
        logging.error(f"Input CSV file not found: {csv_file_path}")
        sys.exit(1)
    
    # Load CSV
    try:
        df = pd.read_csv(csv_file_path)
        logging.info(f"Loaded {len(df)} rows from CSV")
    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        sys.exit(1)
    
    # Validate required columns exist
    required_columns = ['QC_Result', 'Filename']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns in CSV: {missing_columns}")
        logging.error(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Set output directory
    if output_base_dir is None:
        # Default: relative to script location
        script_dir = Path(__file__).parent
        output_base_dir = script_dir / 'data' / 'viz' / 'QC'
    else:
        output_base_dir = Path(output_base_dir)
    
    output_dir = output_base_dir / project_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Create visualizations
    create_qc_passrate_dotplot(df, str(output_dir), project_name)
    create_qc_diverging_bar_chart(df, str(output_dir), project_name)
    create_qc_diverging_bar_chart_compact(df, str(output_dir), project_name)
    
    logging.info("QC visualization completed successfully!")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate QC pass/fail visualizations for QTRAP pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths based on project name
  python Q_viz_QC.py --project solventmatrix
  
  # Specify custom input CSV
  python Q_viz_QC.py --project solventmatrix --input /path/to/custom_qc_results.csv
  
  # Specify custom output directory
  python Q_viz_QC.py --project solventmatrix --output /path/to/output
        """
    )
    
    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='Project name (e.g., solventmatrix). Used for file naming and default paths.'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input QC results CSV file. If not specified, uses: '
             'data/qc/results/{project}/QC_{project}_RESULTS.csv'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Base output directory for visualizations. If not specified, uses: data/viz/QC'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine input CSV path
    if args.input is None:
        # Use default path based on project name
        script_dir = Path(__file__).parent
        input_csv = script_dir / 'data' / 'qc' / 'results' / args.project / f'QC_{args.project}_RESULTS.csv'
    else:
        input_csv = Path(args.input)
    
    # Run visualization
    visualize_qc_results(
        csv_file_path=str(input_csv),
        project_name=args.project,
        output_base_dir=args.output
    )


if __name__ == '__main__':
    main()