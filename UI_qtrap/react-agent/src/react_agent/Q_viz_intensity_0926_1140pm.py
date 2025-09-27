#!/usr/bin/env python3
"""
Script Name: Q_viz_intensity.py
Author: Cascade
Date: 2025-09-26
Version: 1.0

Description:
This script generates bar plots for lipid analysis data, specifically visualizing
Total Ion Count (TIC) intensities across different Base IDs and Lipid Classes.

Usage:
    python Q_viz_intensity.py --input /path/to/organized_data.csv

Example:
    python Q_viz_intensity.py --input ../data/qc/results/solventmatrix/organized/organized_QC_solventmatrix_RESULTS.csv
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from matplotlib.patches import Patch

# --- Utility Functions ---

def setup_logging(log_level: str = 'INFO') -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def extract_project_name(filename: str) -> Optional[str]:
    """Extract project name from filename using 'Proj-' pattern."""
    if pd.isna(filename):
        return None
    match = re.search(r'Proj-([^\\._]+)', str(filename))
    return match.group(1) if match else None

def extract_solvent_matrix(base_id: str) -> str:
    """
    Extract solvent matrix identifier from BaseId.
    """
    parts = base_id.split('_')
    if len(parts) >= 3:
        return parts[1]  # Return the middle part
    return base_id  # Fallback to original if format is unexpected

def assign_colors_by_solvent_type(solvent_matrices: List[str]) -> Dict[str, str]:
    """
    Assign colors to solvent matrices based on their prefixes.
    """
    color_map = {}
    for matrix in solvent_matrices:
        if matrix.startswith(('21Me', '532Me')):
            color_map[matrix] = '#1f77b4'  # Blue
        elif matrix.startswith(('124CH', '12CH')):
            color_map[matrix] = '#d62728'  # Red
    return color_map

# --- Data Processing Functions ---

def prepare_visualization_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Process data for plotting: group, aggregate, and pivot."""
    required_columns = ['BaseId', 'LipidClass', 'Summed_TIC']
    if not all(col in df.columns for col in required_columns):
        logging.error(f"Missing required columns. Found: {list(df.columns)}, Required: {required_columns}")
        return None

    logging.info("Preparing data for visualization...")
    
    # Group by BaseId and LipidClass
    grouped = df.groupby(['BaseId', 'LipidClass'])
    
    # Calculate highest, average, and standard deviation of TIC
    agg_data = grouped['Summed_TIC'].agg(['max', 'mean', 'std']).reset_index()
    agg_data.rename(columns={'max': 'Highest_Summed_TIC', 'mean': 'Average_Summed_TIC', 'std': 'Std_Summed_TIC'}, inplace=True)
    agg_data['Std_Summed_TIC'].fillna(0, inplace=True) # Replace NaN std for single replicates with 0

    # Calculate Std Dev Percent (Coefficient of Variation)
    agg_data['Std_Dev_Percent'] = (agg_data['Std_Summed_TIC'] / agg_data['Average_Summed_TIC']) * 100
    agg_data['Std_Dev_Percent'].fillna(0, inplace=True)
    
    # Add simplified solvent matrix labels
    agg_data['SolventMatrix'] = agg_data['BaseId'].apply(extract_solvent_matrix)

    # Sort by the new SolventMatrix for consistent plot order
    agg_data = agg_data.sort_values('SolventMatrix').reset_index(drop=True)
    
    logging.info(f"Data prepared. Found {len(agg_data)} BaseId/LipidClass combinations.")
    return agg_data

# --- Plotting Functions ---

def create_highest_tic_plot(df: pd.DataFrame, output_dir: str, project_name: str, lipid_class: str):
    """Generate and save a bar plot for the highest Summed_TIC for a specific lipid class."""
    logging.info(f"Creating plot for Highest Summed TIC for {lipid_class}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)

    # Assign colors based on solvent matrix
    solvent_matrices = df['SolventMatrix'].unique()
    color_map = assign_colors_by_solvent_type(solvent_matrices)

    sns.barplot(data=df, x='SolventMatrix', y='Highest_Summed_TIC', palette=color_map, ax=ax)
    
    ax.set_xlabel('Solvent Matrix', fontsize=12, fontweight='bold')
    ax.set_ylabel('Highest Summed TIC', fontsize=12)
    
    ax.tick_params(axis='x', labelrotation=45, labelsize=10)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Create custom legend for solvent types
    legend_elements = []
    used_colors = set(color_map.values())

    if '#1f77b4' in used_colors:
        legend_elements.append(Patch(facecolor='#1f77b4', label='Human'))
    if '#d62728' in used_colors:
        legend_elements.append(Patch(facecolor='#d62728', label='RAG Agent'))


    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout
    
    output_path = Path(output_dir) / f'highest_tic_{lipid_class}.png'
    try:
        plt.savefig(output_path)
        logging.info(f"Plot saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save plot for {lipid_class}: {e}")
    plt.close(fig)

def create_average_tic_plot(df: pd.DataFrame, output_dir: str, project_name: str, lipid_class: str):
    """Generate and save a bar plot for the average Summed_TIC for a specific lipid class."""
    logging.info(f"Creating plot for Average Summed TIC for {lipid_class}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)

    # Assign colors based on solvent matrix
    solvent_matrices = df['SolventMatrix'].unique()
    color_map = assign_colors_by_solvent_type(solvent_matrices)
    colors = [color_map.get(matrix) for matrix in df['SolventMatrix']]

    # Use matplotlib bar plot to have control over error bars (yerr)
    ax.bar(df['SolventMatrix'], df['Average_Summed_TIC'], yerr=df['Std_Summed_TIC'], 
           color=colors, capsize=5, ecolor='black')

    ax.set_xlabel('Solvent Matrix', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Summed TIC', fontsize=12)

    ax.tick_params(axis='x', labelrotation=45, labelsize=10)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    # Create custom legend for solvent types
    legend_elements = []
    used_colors = set(color_map.values())

    if '#1f77b4' in used_colors:
        legend_elements.append(Patch(facecolor='#1f77b4', label='Human'))
    if '#d62728' in used_colors:
        legend_elements.append(Patch(facecolor='#d62728', label='RAG Agent'))

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout

    output_path = Path(output_dir) / f'average_tic_{lipid_class}.png'
    try:
        plt.savefig(output_path)
        logging.info(f"Plot saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save plot for {lipid_class}: {e}")
    plt.close(fig)

def create_tic_barplots(df: pd.DataFrame, output_dir: str, project_name: str):
    """Create and save both highest and average TIC bar plots for each lipid class."""
    if df is None or df.empty:
        logging.warning("Data for plotting is empty. Skipping plot generation.")
        return

    lipid_classes = df['LipidClass'].unique()
    logging.info(f"Found {len(lipid_classes)} lipid classes to plot: {', '.join(lipid_classes)}")

    for lipid_class in lipid_classes:
        logging.info(f"--- Processing plots for {lipid_class} ---")
        lipid_df = df[df['LipidClass'] == lipid_class].copy()
        
        if lipid_df.empty:
            logging.warning(f"No data for lipid class {lipid_class}, skipping.")
            continue

        create_highest_tic_plot(lipid_df, output_dir, project_name, lipid_class)
        create_average_tic_plot(lipid_df, output_dir, project_name, lipid_class)

def generate_intensity_win_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize which BaseId has the most 'wins' for highest and average TIC."""
    if df is None or df.empty:
        return pd.DataFrame()

    logging.info("Generating intensity win summary...")

    # Find winner for Highest TIC for each lipid class
    highest_tic_winners = df.loc[df.groupby('LipidClass')['Highest_Summed_TIC'].idxmax()]
    
    # Find winner for Average TIC for each lipid class
    average_tic_winners = df.loc[df.groupby('LipidClass')['Average_Summed_TIC'].idxmax()]

    # Count wins for each SolventMatrix
    highest_wins = highest_tic_winners['SolventMatrix'].value_counts().reset_index()
    highest_wins.columns = ['SolventMatrix', 'Highest_TIC_Wins']
    
    average_wins = average_tic_winners['SolventMatrix'].value_counts().reset_index()
    average_wins.columns = ['SolventMatrix', 'Average_TIC_Wins']

    avg_stats = df.groupby('SolventMatrix').agg(
        Avg_Std_Dev=('Std_Summed_TIC', 'mean'),
        Avg_Std_Dev_Percent=('Std_Dev_Percent', 'mean')
    ).reset_index()

    # Merge summaries and stats
    summary_df = pd.merge(highest_wins, average_wins, on='SolventMatrix', how='outer')
    summary_df = pd.merge(summary_df, avg_stats, on='SolventMatrix', how='outer').fillna(0)
    
    # Convert counts to integer
    summary_df[['Highest_TIC_Wins', 'Average_TIC_Wins']] = summary_df[['Highest_TIC_Wins', 'Average_TIC_Wins']].astype(int)
    
    # Sort by total wins
    summary_df['Total_Wins'] = summary_df['Highest_TIC_Wins'] + summary_df['Average_TIC_Wins']
    summary_df = summary_df.sort_values(by='Total_Wins', ascending=False).reset_index(drop=True)
    
    return summary_df

def generate_detailed_win_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a detailed report of the winning BaseId for each lipid class."""
    if df is None or df.empty:
        return pd.DataFrame()

    logging.info("Generating detailed intensity win report...")

    # Find winner for Highest TIC for each lipid class
    highest_tic_winners = df.loc[df.groupby('LipidClass')['Highest_Summed_TIC'].idxmax()]
    highest_tic_winners = highest_tic_winners[['LipidClass', 'SolventMatrix', 'Highest_Summed_TIC']].rename(columns={
        'SolventMatrix': 'Highest_TIC_Winner_SolventMatrix',
        'Highest_Summed_TIC': 'Highest_TIC_Value'
    })

    # Find winner for Average TIC for each lipid class
    average_tic_winners = df.loc[df.groupby('LipidClass')['Average_Summed_TIC'].idxmax()].copy()
    
    average_tic_winners = average_tic_winners[['LipidClass', 'SolventMatrix', 'Average_Summed_TIC', 'Std_Summed_TIC', 'Std_Dev_Percent']].rename(columns={
        'SolventMatrix': 'Average_TIC_Winner_SolventMatrix',
        'Average_Summed_TIC': 'Average_TIC_Value',
        'Std_Summed_TIC': 'Std_Dev'
    })

    # Merge the reports
    detailed_report_df = pd.merge(highest_tic_winners, average_tic_winners, on='LipidClass', how='outer')
    
    return detailed_report_df

def generate_std_dev_report(df: pd.DataFrame, output_dir: str):
    """Generate a CSV report of Std Dev and CV for each SolventMatrix, pivoted by LipidClass."""
    if df is None or df.empty:
        return

    logging.info("Generating standard deviation report...")

    # Select relevant columns
    std_dev_data = df[['SolventMatrix', 'LipidClass', 'Std_Summed_TIC', 'Std_Dev_Percent']]

    # Pivot the table
    pivoted_df = std_dev_data.pivot(index='SolventMatrix', columns='LipidClass', values=['Std_Summed_TIC', 'Std_Dev_Percent'])

    # Flatten the multi-level column index and format names
    pivoted_df.columns = [f'{col[1]}_{col[0].replace("Std_Summed_TIC", "Std_Dev").replace("Std_Dev_Percent", "Std_Dev_Percent")}' for col in pivoted_df.columns]
    
    # Reorder columns to group by lipid class for readability
    lipid_classes = sorted(df['LipidClass'].unique())
    new_column_order = []
    for lipid in lipid_classes:
        new_column_order.append(f'{lipid}_Std_Dev')
        new_column_order.append(f'{lipid}_Std_Dev_Percent')
    
    pivoted_df = pivoted_df[new_column_order]

    pivoted_df.reset_index(inplace=True)

    # Save to CSV
    output_path = Path(output_dir) / 'std_dev_by_lipid_class.csv'
    try:
        pivoted_df.to_csv(output_path, index=False)
        logging.info(f"Standard deviation report saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save standard deviation report: {e}")

# --- Main Controller ---

def visualize_tic_intensity(csv_file_path: str, output_base_dir: Optional[str] = None):
    """Main entry point to load, process, and visualize TIC data."""
    logging.info(f"Starting TIC intensity visualization for: {csv_file_path}")
    
    # Validate input file
    if not Path(csv_file_path).exists():
        logging.error(f"Input file not found: {csv_file_path}")
        return

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        logging.error(f"Failed to load CSV file: {e}")
        return

    if 'Filename' not in df.columns:
        logging.error("Missing 'Filename' column required to extract project name.")
        return

    # Extract project name from the first valid filename
    project_name = df['Filename'].apply(extract_project_name).dropna().iloc[0]
    if not project_name:
        logging.error("Could not extract project name from 'Proj-' pattern in any filename.")
        return
    logging.info(f"Extracted project name: {project_name}")

    # Prepare output directory
    if output_base_dir is None:
        output_base_dir = Path(__file__).parent / 'data' / 'viz' / 'intensity'
    
    output_dir = Path(output_base_dir) / project_name
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory set to: {output_dir}")
    except Exception as e:
        logging.error(f"Could not create output directory: {e}")
        return

    # Process data for visualization
    viz_data = prepare_visualization_data(df)
    if viz_data is None:
        logging.error("Failed to prepare data for visualization.")
        return

    # Generate plots
    create_tic_barplots(viz_data, str(output_dir), project_name)

    # Generate and save win summary
    win_summary_df = generate_intensity_win_summary(viz_data)
    if not win_summary_df.empty:
        summary_path = output_dir / 'intensity_win_summary.csv'
        try:
            win_summary_df.to_csv(summary_path, index=False)
            logging.info(f"Intensity win summary saved to: {summary_path}")
        except Exception as e:
            logging.error(f"Failed to save win summary: {e}")

    # Generate and save detailed win report
    detailed_report_df = generate_detailed_win_report(viz_data)
    if not detailed_report_df.empty:
        detailed_report_path = output_dir / 'detailed_intensity_winners.csv'
        try:
            detailed_report_df.to_csv(detailed_report_path, index=False)
            logging.info(f"Detailed win report saved to: {detailed_report_path}")
        except Exception as e:
            logging.error(f"Failed to save detailed win report: {e}")

    # Generate and save the new standard deviation report
    generate_std_dev_report(viz_data, output_dir)
    
    logging.info("Visualization script finished.")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TIC intensity visualizations from lipid analysis data.")
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the input organized CSV file.'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Optional: Base directory for visualization output. Defaults to ../data/viz/intensity.'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level.'
    )
    return parser.parse_args()

def main():
    """Main execution function when run as a script."""
    args = parse_arguments()
    setup_logging(args.log_level)
    visualize_tic_intensity(args.input, args.output)

if __name__ == "__main__":
    main()