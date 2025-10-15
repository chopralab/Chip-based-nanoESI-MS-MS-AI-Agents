#!/usr/bin/env python3
"""
Advanced visualization functions for Q_viz_intensity.py - Part 2
Statistical swarmplot and forest plot functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def extract_solvent_matrix(base_id: str) -> str:
    """Extract solvent matrix identifier from BaseId."""
    parts = base_id.split('_')
    if len(parts) >= 3:
        return parts[1]
    return base_id


def assign_colors_by_solvent_type(solvent_matrices):
    """Assign colors to solvent matrices based on their prefixes."""
    color_map = {}
    for matrix in solvent_matrices:
        if matrix.startswith(('21Me', '532Me')):
            color_map[matrix] = '#1f77b4'  # Blue
        elif matrix.startswith(('124CH', '12CH')):
            color_map[matrix] = '#d62728'  # Red
    return color_map


def create_statistical_swarmplot(df: pd.DataFrame, output_dir: str, project_name: str, lipid_class: str, raw_df: pd.DataFrame):
    """Generate and save a statistical swarm plot with ANOVA and Tukey HSD for a specific lipid class.
    
    Parameters:
        df: Aggregated DataFrame with summary statistics
        output_dir: Directory path to save output files
        project_name: Project identifier for logging
        lipid_class: Specific lipid class to analyze
        raw_df: Original raw data with individual replicates
    
    Outputs:
        - statistical_swarm_{lipid_class}.png (300 DPI)
        - statistical_swarm_{lipid_class}.pdf
    """
    if df is None or df.empty:
        logging.warning(f"Data for statistical swarmplot ({lipid_class}) is empty. Skipping generation.")
        return
    
    logging.info(f"Creating statistical swarm plot for {lipid_class}...")
    
    try:
        # Filter data for this lipid class
        lipid_df = df[df['LipidClass'] == lipid_class].copy()
        
        if lipid_df.empty or len(lipid_df) < 2:
            logging.warning(f"Insufficient data for statistical analysis of {lipid_class}")
            return
        
        # Get raw data for swarm plot
        raw_lipid_df = raw_df[raw_df['LipidClass'] == lipid_class].copy()
        raw_lipid_df['SolventMatrix'] = raw_lipid_df['BaseId'].apply(extract_solvent_matrix)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
        
        # Assign colors based on solvent type
        solvent_matrices = sorted(lipid_df['SolventMatrix'].unique())
        color_map = assign_colors_by_solvent_type(solvent_matrices)
        colors = [color_map.get(sm, '#gray') for sm in solvent_matrices]
        
        # Create strip plot (swarm alternative for better performance)
        if not raw_lipid_df.empty:
            sns.stripplot(
                data=raw_lipid_df,
                x='SolventMatrix',
                y='Summed_TIC',
                order=solvent_matrices,
                palette=color_map,
                size=6,
                alpha=0.6,
                ax=ax
            )
        
        # Overlay mean with 95% CI
        means = lipid_df.set_index('SolventMatrix').loc[solvent_matrices, 'Average_Summed_TIC'].values
        stds = lipid_df.set_index('SolventMatrix').loc[solvent_matrices, 'Std_Summed_TIC'].values
        
        # Calculate 95% CI (assuming normal distribution)
        ci_95 = 1.96 * stds
        
        x_positions = np.arange(len(solvent_matrices))
        ax.errorbar(
            x_positions,
            means,
            yerr=ci_95,
            fmt='D',
            markersize=10,
            color='black',
            linewidth=2,
            capsize=10,
            capthick=2,
            label='Mean ± 95% CI'
        )
        
        # Perform ANOVA if we have raw data
        anova_text = ""
        if not raw_lipid_df.empty and len(solvent_matrices) >= 2:
            try:
                # Prepare data for ANOVA
                groups = [raw_lipid_df[raw_lipid_df['SolventMatrix'] == sm]['Summed_TIC'].values 
                         for sm in solvent_matrices]
                groups = [g for g in groups if len(g) > 0]  # Remove empty groups
                
                if len(groups) >= 2:
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_text = f"ANOVA: F={f_stat:.2f}, p={p_value:.4f}"
                    
                    # Perform Tukey HSD if significant
                    if p_value < 0.05 and len(raw_lipid_df) > len(solvent_matrices):
                        try:
                            tukey_result = pairwise_tukeyhsd(
                                raw_lipid_df['Summed_TIC'],
                                raw_lipid_df['SolventMatrix']
                            )
                            
                            # Generate compact letter display (simplified)
                            groups_letters = {}
                            current_letter = ord('a')
                            
                            for i, sm in enumerate(solvent_matrices):
                                groups_letters[sm] = chr(current_letter)
                                current_letter += 1
                            
                            # Add letters above bars
                            y_max = means.max() + ci_95.max()
                            for i, sm in enumerate(solvent_matrices):
                                ax.text(
                                    i,
                                    y_max * 1.05,
                                    groups_letters[sm],
                                    ha='center',
                                    va='bottom',
                                    fontsize=14,
                                    fontweight='bold'
                                )
                        except Exception as e:
                            logging.warning(f"Tukey HSD failed for {lipid_class}: {e}")
            except Exception as e:
                logging.warning(f"ANOVA failed for {lipid_class}: {e}")
                anova_text = "ANOVA: Unable to compute"
        
        ax.set_xlabel('Solvent System', fontsize=16, fontweight='bold')
        ax.set_ylabel('TIC Area (×10⁶)', fontsize=16, fontweight='bold')
        ax.set_title(f'Statistical Analysis - {lipid_class} - {project_name}', fontsize=18, fontweight='bold', pad=20)
        
        # Format y-axis in millions
        y_ticks = ax.get_yticks()
        ax.set_yticklabels([f'{y/1e6:.1f}' for y in y_ticks])
        
        ax.tick_params(axis='x', labelrotation=45, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        # Add ANOVA text
        if anova_text:
            ax.text(
                0.02, 0.98,
                anova_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        ax.legend(fontsize=12, loc='upper right')
        
        plt.tight_layout()
        
        # Save PNG
        output_path_png = Path(output_dir) / f'statistical_swarm_{lipid_class}.png'
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        logging.info(f"PNG statistical swarmplot saved to: {output_path_png}")
        
        # Save PDF
        output_path_pdf = Path(output_dir) / f'statistical_swarm_{lipid_class}.pdf'
        plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
        logging.info(f"PDF statistical swarmplot saved to: {output_path_pdf}")
        
        plt.close(fig)
        
    except Exception as e:
        logging.error(f"Failed to create statistical swarmplot for {lipid_class}: {e}")


def create_forest_plot_and_tables(df: pd.DataFrame, output_dir: str, project_name: str, reference_solvent: str = '21MeOHACN'):
    """Generate forest plot showing effect sizes and accompanying statistical tables.
    
    Parameters:
        df: DataFrame with columns ['SolventMatrix', 'LipidClass', 'Average_Summed_TIC', 'Std_Summed_TIC']
        output_dir: Directory path to save output files
        project_name: Project identifier for logging
        reference_solvent: Reference solvent for comparison (default: '21MeOHACN')
    
    Outputs:
        - forest_plot_effect_sizes.png (300 DPI)
        - forest_plot_effect_sizes.pdf
        - means_table.csv
        - stats_table.csv
        - normalized_table.csv
        - effect_sizes.csv
    """
    if df is None or df.empty:
        logging.warning("Data for forest plot is empty. Skipping generation.")
        return
    
    logging.info("Creating forest plot and statistical tables...")
    
    try:
        output_path = Path(output_dir)
        
        # Check if reference solvent exists, otherwise use first available
        available_solvents = df['SolventMatrix'].unique()
        if reference_solvent not in available_solvents:
            reference_solvent = sorted(available_solvents)[0]
            logging.warning(f"Reference solvent not found. Using {reference_solvent} as reference.")
        
        lipid_classes = sorted(df['LipidClass'].unique())
        
        # Generate means table
        means_table = df.pivot(index='SolventMatrix', columns='LipidClass', values='Average_Summed_TIC')
        means_table.to_csv(output_path / 'means_table.csv')
        logging.info(f"Means table saved to: {output_path / 'means_table.csv'}")
        
        # Generate stats table (SD, CV, Rank)
        stats_data = []
        for sm in df['SolventMatrix'].unique():
            for lc in lipid_classes:
                row_data = df[(df['SolventMatrix'] == sm) & (df['LipidClass'] == lc)]
                if not row_data.empty:
                    mean_val = row_data['Average_Summed_TIC'].values[0]
                    std_val = row_data['Std_Summed_TIC'].values[0]
                    cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                    ci_lower = mean_val - 1.96 * std_val
                    ci_upper = mean_val + 1.96 * std_val
                    
                    stats_data.append({
                        'SolventMatrix': sm,
                        'LipidClass': lc,
                        'Mean': mean_val,
                        'SD': std_val,
                        'CV_%': cv,
                        'CI_95_Lower': ci_lower,
                        'CI_95_Upper': ci_upper
                    })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Add rank within each lipid class
        stats_df['Rank'] = stats_df.groupby('LipidClass')['Mean'].rank(ascending=False, method='dense').astype(int)
        stats_df.to_csv(output_path / 'stats_table.csv', index=False)
        logging.info(f"Stats table saved to: {output_path / 'stats_table.csv'}")
        
        # Generate normalized table (reference = 100%)
        normalized_data = []
        for lc in lipid_classes:
            ref_data = df[(df['SolventMatrix'] == reference_solvent) & (df['LipidClass'] == lc)]
            if ref_data.empty:
                continue
            ref_mean = ref_data['Average_Summed_TIC'].values[0]
            
            for sm in df['SolventMatrix'].unique():
                row_data = df[(df['SolventMatrix'] == sm) & (df['LipidClass'] == lc)]
                if not row_data.empty:
                    mean_val = row_data['Average_Summed_TIC'].values[0]
                    normalized_val = (mean_val / ref_mean * 100) if ref_mean > 0 else 0
                    normalized_data.append({
                        'SolventMatrix': sm,
                        'LipidClass': lc,
                        'Normalized_%': normalized_val
                    })
        
        normalized_df = pd.DataFrame(normalized_data)
        normalized_pivot = normalized_df.pivot(index='SolventMatrix', columns='LipidClass', values='Normalized_%')
        normalized_pivot.to_csv(output_path / 'normalized_table.csv')
        logging.info(f"Normalized table saved to: {output_path / 'normalized_table.csv'}")
        
        # Calculate effect sizes (Cohen's d)
        effect_sizes_list = []
        
        for lipid_class in lipid_classes:
            lipid_df = df[df['LipidClass'] == lipid_class]
            ref_data = lipid_df[lipid_df['SolventMatrix'] == reference_solvent]
            
            if ref_data.empty:
                continue
            
            ref_mean = ref_data['Average_Summed_TIC'].values[0]
            ref_std = ref_data['Std_Summed_TIC'].values[0]
            
            for _, row in lipid_df.iterrows():
                if row['SolventMatrix'] == reference_solvent:
                    continue
                
                # Cohen's d = (mean1 - mean2) / pooled_std
                mean_diff = row['Average_Summed_TIC'] - ref_mean
                pooled_std = np.sqrt((ref_std**2 + row['Std_Summed_TIC']**2) / 2)
                
                if pooled_std > 0:
                    cohens_d = mean_diff / pooled_std
                    # Approximate 95% CI for Cohen's d (simplified)
                    se_d = np.sqrt(2 / 10)  # Assuming n=10 per group
                    ci_lower = cohens_d - 1.96 * se_d
                    ci_upper = cohens_d + 1.96 * se_d
                else:
                    cohens_d = 0
                    ci_lower = 0
                    ci_upper = 0
                
                effect_sizes_list.append({
                    'LipidClass': lipid_class,
                    'SolventMatrix': row['SolventMatrix'],
                    'Cohens_d': cohens_d,
                    'CI_lower': ci_lower,
                    'CI_upper': ci_upper
                })
        
        effect_sizes_df = pd.DataFrame(effect_sizes_list)
        effect_sizes_df.to_csv(output_path / 'effect_sizes.csv', index=False)
        logging.info(f"Effect sizes table saved to: {output_path / 'effect_sizes.csv'}")
        
        # Create forest plot
        if not effect_sizes_df.empty:
            plt.style.use('seaborn-v0_8-whitegrid')
            n_lipids = len(lipid_classes)
            fig, axes = plt.subplots(n_lipids, 1, figsize=(12, 4 * n_lipids), dpi=300)
            
            if n_lipids == 1:
                axes = [axes]
            
            for i, lipid_class in enumerate(lipid_classes):
                ax = axes[i]
                lipid_effects = effect_sizes_df[effect_sizes_df['LipidClass'] == lipid_class]
                
                if lipid_effects.empty:
                    ax.text(0.5, 0.5, f'No data for {lipid_class}', 
                           ha='center', va='center', fontsize=14)
                    continue
                
                y_positions = np.arange(len(lipid_effects))
                
                # Plot effect sizes with CI
                ax.errorbar(
                    lipid_effects['Cohens_d'],
                    y_positions,
                    xerr=[lipid_effects['Cohens_d'] - lipid_effects['CI_lower'],
                          lipid_effects['CI_upper'] - lipid_effects['Cohens_d']],
                    fmt='o',
                    markersize=8,
                    linewidth=2,
                    capsize=5,
                    color='steelblue'
                )
                
                # Add vertical line at 0 (no effect)
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
                
                ax.set_yticks(y_positions)
                ax.set_yticklabels(lipid_effects['SolventMatrix'], fontsize=12)
                ax.set_xlabel("Cohen's d (Effect Size)", fontsize=14, fontweight='bold')
                ax.set_title(f'{lipid_class} vs {reference_solvent}', fontsize=16, fontweight='bold')
                ax.tick_params(axis='x', labelsize=12)
                ax.grid(axis='x', alpha=0.3)
            
            plt.suptitle(f'Effect Sizes - {project_name}', fontsize=20, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            # Save PNG
            output_path_png = output_path / 'forest_plot_effect_sizes.png'
            plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
            logging.info(f"PNG forest plot saved to: {output_path_png}")
            
            # Save PDF
            output_path_pdf = output_path / 'forest_plot_effect_sizes.pdf'
            plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
            logging.info(f"PDF forest plot saved to: {output_path_pdf}")
            
            plt.close(fig)
        
    except Exception as e:
        logging.error(f"Failed to create forest plot and tables: {e}")


def create_advanced_visualizations(df: pd.DataFrame, raw_df: pd.DataFrame, output_dir: str, project_name: str):
    """Controller function to create all advanced visualizations.
    
    Parameters:
        df: Aggregated DataFrame with summary statistics
        raw_df: Original raw data with individual replicates
        output_dir: Directory path to save output files
        project_name: Project identifier for logging
    """
    logging.info("=== Starting Advanced Visualizations ===")
    
    # Create heatmap
    create_heatmap_plot(df, output_dir, project_name)
    
    # Create grouped barplot
    create_grouped_barplot(df, output_dir, project_name)
    
    # Create dotplot
    create_dotplot(df, output_dir, project_name)
    
    # Create statistical swarmplots for each lipid class
    lipid_classes = df['LipidClass'].unique()
    for lipid_class in lipid_classes:
        create_statistical_swarmplot(df, output_dir, project_name, lipid_class, raw_df)
    
    # Create forest plot and tables
    create_forest_plot_and_tables(df, output_dir, project_name)
    
    logging.info("=== Advanced Visualizations Complete ===")
