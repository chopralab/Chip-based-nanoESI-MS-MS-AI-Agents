#!/usr/bin/env python3
"""
Advanced visualization functions for Q_viz_intensity.py
These functions will be integrated into the main script.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def create_heatmap_plot(df: pd.DataFrame, output_dir: str, project_name: str):
    """Generate and save a heatmap showing TIC values across lipid classes and solvent matrices.
    
    Parameters:
        df: DataFrame with columns ['LipidClass', 'SolventMatrix', 'Average_Summed_TIC']
        output_dir: Directory path to save output files
        project_name: Project identifier for logging
    
    Outputs:
        - heatmap_all_lipids.png (300 DPI)
        - heatmap_all_lipids.pdf
    """
    if df is None or df.empty:
        logging.warning("Data for heatmap is empty. Skipping heatmap generation.")
        return
    
    logging.info("Creating heatmap for all lipid classes...")
    
    try:
        # Pivot data for heatmap: rows = lipid classes, columns = solvent matrices
        heatmap_data = df.pivot(index='LipidClass', columns='SolventMatrix', values='Average_Summed_TIC')
        
        # Convert to millions for readability
        heatmap_data_millions = heatmap_data / 1e6
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
        
        # Create heatmap with annotations
        sns.heatmap(
            heatmap_data_millions,
            annot=True,
            fmt='.2f',
            cmap='viridis',
            cbar_kws={'label': 'TIC Area (×10⁶)'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        
        ax.set_xlabel('Solvent System', fontsize=16, fontweight='bold')
        ax.set_ylabel('Lipid Class', fontsize=16, fontweight='bold')
        ax.set_title(f'TIC Intensity Heatmap - {project_name}', fontsize=18, fontweight='bold', pad=20)
        
        ax.tick_params(axis='x', labelrotation=45, labelsize=14)
        ax.tick_params(axis='y', labelrotation=0, labelsize=14)
        
        # Adjust colorbar label size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('TIC Area (×10⁶)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save PNG
        output_path_png = Path(output_dir) / 'heatmap_all_lipids.png'
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        logging.info(f"PNG heatmap saved to: {output_path_png}")
        
        # Save PDF
        output_path_pdf = Path(output_dir) / 'heatmap_all_lipids.pdf'
        plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
        logging.info(f"PDF heatmap saved to: {output_path_pdf}")
        
        plt.close(fig)
        
    except Exception as e:
        logging.error(f"Failed to create heatmap: {e}")


def create_grouped_barplot(df: pd.DataFrame, output_dir: str, project_name: str):
    """Generate and save a grouped bar plot with all lipid classes.
    
    Parameters:
        df: DataFrame with columns ['SolventMatrix', 'LipidClass', 'Average_Summed_TIC']
        output_dir: Directory path to save output files
        project_name: Project identifier for logging
    
    Outputs:
        - grouped_barplot_all_lipids.png (300 DPI)
        - grouped_barplot_all_lipids.pdf
    """
    if df is None or df.empty:
        logging.warning("Data for grouped barplot is empty. Skipping generation.")
        return
    
    logging.info("Creating grouped bar plot for all lipid classes...")
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
        
        # Use colorblind-friendly palette
        lipid_classes = sorted(df['LipidClass'].unique())
        palette = sns.color_palette('colorblind', n_colors=len(lipid_classes))
        
        # Create grouped bar plot
        solvent_matrices = sorted(df['SolventMatrix'].unique())
        x = np.arange(len(solvent_matrices))
        width = 0.8 / len(lipid_classes)
        
        for i, lipid_class in enumerate(lipid_classes):
            lipid_data = df[df['LipidClass'] == lipid_class].sort_values('SolventMatrix')
            offset = width * i - (width * len(lipid_classes) / 2) + width / 2
            ax.bar(
                x + offset,
                lipid_data['Average_Summed_TIC'] / 1e6,
                width,
                label=lipid_class,
                color=palette[i]
            )
        
        ax.set_xlabel('Solvent System', fontsize=16, fontweight='bold')
        ax.set_ylabel('TIC Area (×10⁶)', fontsize=16, fontweight='bold')
        ax.set_title(f'Grouped TIC Comparison - {project_name}', fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(solvent_matrices, rotation=45, ha='right')
        
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        ax.legend(title='Lipid Class', fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save PNG
        output_path_png = Path(output_dir) / 'grouped_barplot_all_lipids.png'
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        logging.info(f"PNG grouped barplot saved to: {output_path_png}")
        
        # Save PDF
        output_path_pdf = Path(output_dir) / 'grouped_barplot_all_lipids.pdf'
        plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
        logging.info(f"PDF grouped barplot saved to: {output_path_pdf}")
        
        plt.close(fig)
        
    except Exception as e:
        logging.error(f"Failed to create grouped barplot: {e}")


def create_dotplot(df: pd.DataFrame, output_dir: str, project_name: str):
    """Generate and save a dot plot with lines connecting points across solvent matrices.
    
    Parameters:
        df: DataFrame with columns ['SolventMatrix', 'LipidClass', 'Average_Summed_TIC', 'Std_Summed_TIC']
        output_dir: Directory path to save output files
        project_name: Project identifier for logging
    
    Outputs:
        - dotplot_all_lipids.png (300 DPI)
        - dotplot_all_lipids.pdf
    """
    if df is None or df.empty:
        logging.warning("Data for dotplot is empty. Skipping generation.")
        return
    
    logging.info("Creating dot plot for all lipid classes...")
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
        
        # Use colorblind-friendly palette
        lipid_classes = sorted(df['LipidClass'].unique())
        palette = sns.color_palette('colorblind', n_colors=len(lipid_classes))
        
        solvent_matrices = sorted(df['SolventMatrix'].unique())
        x_positions = np.arange(len(solvent_matrices))
        
        for i, lipid_class in enumerate(lipid_classes):
            lipid_data = df[df['LipidClass'] == lipid_class].sort_values('SolventMatrix')
            
            if lipid_data.empty:
                continue
            
            y_values = lipid_data['Average_Summed_TIC'].values / 1e6
            y_errors = lipid_data['Std_Summed_TIC'].values / 1e6
            
            ax.errorbar(
                x_positions,
                y_values,
                yerr=y_errors,
                marker='o',
                markersize=8,
                linewidth=2,
                label=lipid_class,
                color=palette[i],
                capsize=5,
                capthick=2
            )
        
        ax.set_xlabel('Solvent System', fontsize=16, fontweight='bold')
        ax.set_ylabel('Average TIC (×10⁶)', fontsize=16, fontweight='bold')
        ax.set_title(f'TIC Trends Across Solvent Systems - {project_name}', fontsize=18, fontweight='bold', pad=20)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(solvent_matrices, rotation=45, ha='right')
        
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        ax.legend(title='Lipid Class', fontsize=12, title_fontsize=14, loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(axis='both', alpha=0.3)
        
        plt.tight_layout()
        
        # Save PNG
        output_path_png = Path(output_dir) / 'dotplot_all_lipids.png'
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        logging.info(f"PNG dotplot saved to: {output_path_png}")
        
        # Save PDF
        output_path_pdf = Path(output_dir) / 'dotplot_all_lipids.pdf'
        plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
        logging.info(f"PDF dotplot saved to: {output_path_pdf}")
        
        plt.close(fig)
        
    except Exception as e:
        logging.error(f"Failed to create dotplot: {e}")


def create_faceted_panel_plot(df: pd.DataFrame, output_dir: str, project_name: str, plot_type: str = 'average'):
    """Generate and save a multi-panel faceted plot with one subplot per lipid class.
    
    Parameters:
        df: DataFrame with columns ['SolventMatrix', 'LipidClass', 'Average_Summed_TIC', 'Highest_Summed_TIC', 'Std_Summed_TIC']
        output_dir: Directory path to save output files
        project_name: Project identifier for logging
        plot_type: Type of plot - 'average' or 'highest' (default: 'average')
    
    Outputs:
        - faceted_panel_{plot_type}_tic.png (300 DPI)
        - faceted_panel_{plot_type}_tic.pdf
    """
    if df is None or df.empty:
        logging.warning("Data for faceted panel plot is empty. Skipping generation.")
        return
    
    logging.info(f"Creating faceted panel plot ({plot_type} TIC) for all lipid classes...")
    
    try:
        from Q_viz_intensity import assign_colors_by_solvent_type
        
        lipid_classes = sorted(df['LipidClass'].unique())
        n_lipids = len(lipid_classes)
        
        # Determine grid layout (prefer wider than tall)
        if n_lipids <= 4:
            nrows, ncols = 2, 2
        elif n_lipids <= 6:
            nrows, ncols = 2, 3
        elif n_lipids <= 9:
            nrows, ncols = 3, 3
        else:
            nrows, ncols = 3, 4
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12), dpi=300)
        axes = axes.flatten() if n_lipids > 1 else [axes]
        
        # Determine which column to use
        if plot_type == 'highest':
            value_col = 'Highest_Summed_TIC'
            ylabel = 'Highest TIC (×10⁶)'
            title_prefix = 'Highest'
        else:
            value_col = 'Average_Summed_TIC'
            ylabel = 'Average TIC (×10⁶)'
            title_prefix = 'Average'
        
        # Get global y-axis limits for consistent scaling
        all_values = df[value_col].values / 1e6
        if plot_type == 'average':
            all_errors = df['Std_Summed_TIC'].values / 1e6
            y_max = (all_values + all_errors).max() * 1.15
        else:
            y_max = all_values.max() * 1.15
        y_min = 0
        
        # Panel labels
        panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        
        for idx, lipid_class in enumerate(lipid_classes):
            ax = axes[idx]
            lipid_df = df[df['LipidClass'] == lipid_class].sort_values('SolventMatrix')
            
            if lipid_df.empty:
                ax.text(0.5, 0.5, f'No data for {lipid_class}', 
                       ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Get colors based on solvent type
            solvent_matrices = lipid_df['SolventMatrix'].values
            color_map = assign_colors_by_solvent_type(solvent_matrices)
            colors = [color_map.get(sm, '#gray') for sm in solvent_matrices]
            
            # Create bar plot
            x_positions = np.arange(len(solvent_matrices))
            
            if plot_type == 'average':
                # Average with error bars
                ax.bar(
                    x_positions,
                    lipid_df[value_col].values / 1e6,
                    yerr=lipid_df['Std_Summed_TIC'].values / 1e6,
                    color=colors,
                    capsize=5,
                    ecolor='black',
                    linewidth=1.5
                )
            else:
                # Highest without error bars
                ax.bar(
                    x_positions,
                    lipid_df[value_col].values / 1e6,
                    color=colors,
                    linewidth=1.5
                )
            
            # Set consistent y-axis limits
            ax.set_ylim(y_min, y_max)
            
            # Add panel label in top-left corner
            ax.text(
                0.02, 0.98,
                f'({panel_labels[idx]})',
                transform=ax.transAxes,
                fontsize=16,
                fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5)
            )
            
            # Subplot title
            ax.set_title(lipid_class, fontsize=14, fontweight='bold', pad=10)
            
            # X-axis
            ax.set_xticks(x_positions)
            ax.set_xticklabels(solvent_matrices, rotation=45, ha='right', fontsize=11)
            
            # Y-axis
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.tick_params(axis='y', labelsize=11)
            
            # Grid
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Hide unused subplots
        for idx in range(n_lipids, len(axes)):
            axes[idx].axis('off')
        
        # Main title
        fig.suptitle(
            f'{title_prefix} TIC Across Solvent Systems - {project_name}',
            fontsize=20,
            fontweight='bold',
            y=0.995
        )
        
        # Create figure caption
        caption_text = (
            f"Figure: Multi-panel comparison of {plot_type} TIC intensities across different solvent systems. "
            f"Each panel (A-{panel_labels[n_lipids-1]}) represents a different lipid class. "
            f"Blue bars indicate Human solvent systems (21Me/532Me), "
            f"red bars indicate RAG solvent systems (124CH/12CH). "
        )
        
        if plot_type == 'average':
            caption_text += "Error bars represent standard deviation. "
        
        caption_text += f"All panels share the same y-axis scale for direct comparison. n={n_lipids} lipid classes."
        
        # Add caption below the figure
        fig.text(
            0.5, 0.02,
            caption_text,
            ha='center',
            fontsize=10,
            style='italic',
            wrap=True,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8)
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.99])
        
        # Save PNG
        output_path_png = Path(output_dir) / f'faceted_panel_{plot_type}_tic.png'
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        logging.info(f"PNG faceted panel plot saved to: {output_path_png}")
        
        # Save PDF
        output_path_pdf = Path(output_dir) / f'faceted_panel_{plot_type}_tic.pdf'
        plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
        logging.info(f"PDF faceted panel plot saved to: {output_path_pdf}")
        
        plt.close(fig)
        
    except Exception as e:
        logging.error(f"Failed to create faceted panel plot: {e}")
