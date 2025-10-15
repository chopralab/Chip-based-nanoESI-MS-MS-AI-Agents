# ============================================================================
# COPY ALL FUNCTIONS BELOW INTO Q_viz_intensity.py
# Insert after generate_std_dev_report() and before # --- Main Controller ---
# ============================================================================

# --- Advanced Visualization Functions ---

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
        heatmap_data = df.pivot(index='LipidClass', columns='SolventMatrix', values='Average_Summed_TIC')
        heatmap_data_millions = heatmap_data / 1e6
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
        
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
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('TIC Area (×10⁶)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_path_png = Path(output_dir) / 'heatmap_all_lipids.png'
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        logging.info(f"PNG heatmap saved to: {output_path_png}")
        
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
        
        lipid_classes = sorted(df['LipidClass'].unique())
        palette = sns.color_palette('colorblind', n_colors=len(lipid_classes))
        
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
        
        output_path_png = Path(output_dir) / 'grouped_barplot_all_lipids.png'
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        logging.info(f"PNG grouped barplot saved to: {output_path_png}")
        
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
        
        output_path_png = Path(output_dir) / 'dotplot_all_lipids.png'
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        logging.info(f"PNG dotplot saved to: {output_path_png}")
        
        output_path_pdf = Path(output_dir) / 'dotplot_all_lipids.pdf'
        plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
        logging.info(f"PDF dotplot saved to: {output_path_pdf}")
        
        plt.close(fig)
        
    except Exception as e:
        logging.error(f"Failed to create dotplot: {e}")


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
        lipid_df = df[df['LipidClass'] == lipid_class].copy()
        
        if lipid_df.empty or len(lipid_df) < 2:
            logging.warning(f"Insufficient data for statistical analysis of {lipid_class}")
            return
        
        raw_lipid_df = raw_df[raw_df['LipidClass'] == lipid_class].copy()
        raw_lipid_df['SolventMatrix'] = raw_lipid_df['BaseId'].apply(extract_solvent_matrix)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
        
        solvent_matrices = sorted(lipid_df['SolventMatrix'].unique())
        color_map = assign_colors_by_solvent_type(solvent_matrices)
        
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
        
        means = lipid_df.set_index('SolventMatrix').loc[solvent_matrices, 'Average_Summed_TIC'].values
        stds = lipid_df.set_index('SolventMatrix').loc[solvent_matrices, 'Std_Summed_TIC'].values
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
        
        anova_text = ""
        if not raw_lipid_df.empty and len(solvent_matrices) >= 2:
            try:
                groups = [raw_lipid_df[raw_lipid_df['SolventMatrix'] == sm]['Summed_TIC'].values 
                         for sm in solvent_matrices]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_text = f"ANOVA: F={f_stat:.2f}, p={p_value:.4f}"
                    
                    if p_value < 0.05 and len(raw_lipid_df) > len(solvent_matrices):
                        try:
                            tukey_result = pairwise_tukeyhsd(
                                raw_lipid_df['Summed_TIC'],
                                raw_lipid_df['SolventMatrix']
                            )
                            
                            groups_letters = {}
                            current_letter = ord('a')
                            
                            for i, sm in enumerate(solvent_matrices):
                                groups_letters[sm] = chr(current_letter)
                                current_letter += 1
                            
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
        
        y_ticks = ax.get_yticks()
        ax.set_yticklabels([f'{y/1e6:.1f}' for y in y_ticks])
        
        ax.tick_params(axis='x', labelrotation=45, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
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
        
        output_path_png = Path(output_dir) / f'statistical_swarm_{lipid_class}.png'
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        logging.info(f"PNG statistical swarmplot saved to: {output_path_png}")
        
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
        
        available_solvents = df['SolventMatrix'].unique()
        if reference_solvent not in available_solvents:
            reference_solvent = sorted(available_solvents)[0]
            logging.warning(f"Reference solvent not found. Using {reference_solvent} as reference.")
        
        lipid_classes = sorted(df['LipidClass'].unique())
        
        means_table = df.pivot(index='SolventMatrix', columns='LipidClass', values='Average_Summed_TIC')
        means_table.to_csv(output_path / 'means_table.csv')
        logging.info(f"Means table saved to: {output_path / 'means_table.csv'}")
        
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
        stats_df['Rank'] = stats_df.groupby('LipidClass')['Mean'].rank(ascending=False, method='dense').astype(int)
        stats_df.to_csv(output_path / 'stats_table.csv', index=False)
        logging.info(f"Stats table saved to: {output_path / 'stats_table.csv'}")
        
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
                
                mean_diff = row['Average_Summed_TIC'] - ref_mean
                pooled_std = np.sqrt((ref_std**2 + row['Std_Summed_TIC']**2) / 2)
                
                if pooled_std > 0:
                    cohens_d = mean_diff / pooled_std
                    se_d = np.sqrt(2 / 10)
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
                
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
                
                ax.set_yticks(y_positions)
                ax.set_yticklabels(lipid_effects['SolventMatrix'], fontsize=12)
                ax.set_xlabel("Cohen's d (Effect Size)", fontsize=14, fontweight='bold')
                ax.set_title(f'{lipid_class} vs {reference_solvent}', fontsize=16, fontweight='bold')
                ax.tick_params(axis='x', labelsize=12)
                ax.grid(axis='x', alpha=0.3)
            
            plt.suptitle(f'Effect Sizes - {project_name}', fontsize=20, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            output_path_png = output_path / 'forest_plot_effect_sizes.png'
            plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
            logging.info(f"PNG forest plot saved to: {output_path_png}")
            
            output_path_pdf = output_path / 'forest_plot_effect_sizes.pdf'
            plt.savefig(output_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
            logging.info(f"PDF forest plot saved to: {output_path_pdf}")
            
            plt.close(fig)
        
    except Exception as e:
        logging.error(f"Failed to create forest plot and tables: {e}")


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


def create_advanced_visualizations(df: pd.DataFrame, raw_df: pd.DataFrame, output_dir: str, project_name: str):
    """Controller function to create all advanced visualizations.
    
    Parameters:
        df: Aggregated DataFrame with summary statistics
        raw_df: Original raw data with individual replicates
        output_dir: Directory path to save output files
        project_name: Project identifier for logging
    """
    logging.info("=== Starting Advanced Visualizations ===")
    
    create_heatmap_plot(df, output_dir, project_name)
    create_grouped_barplot(df, output_dir, project_name)
    create_dotplot(df, output_dir, project_name)
    create_faceted_panel_plot(df, output_dir, project_name, plot_type='average')
    create_faceted_panel_plot(df, output_dir, project_name, plot_type='highest')
    
    lipid_classes = df['LipidClass'].unique()
    for lipid_class in lipid_classes:
        create_statistical_swarmplot(df, output_dir, project_name, lipid_class, raw_df)
    
    create_forest_plot_and_tables(df, output_dir, project_name)
    
    logging.info("=== Advanced Visualizations Complete ===")
