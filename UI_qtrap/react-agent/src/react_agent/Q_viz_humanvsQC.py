import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set high-quality figure parameters for publication
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Define paths
DATA_PATH = Path(__file__).parent / "data/qc/TIC/solventmatrix/humanvsQCagent/Human_QC_Results_HumanandAIAgent_Normalized_v4.xlsx"
OUTPUT_DIR = Path(__file__).parent / "data/qc/TIC/solventmatrix/humanvsQCagent/results"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the data
print("Loading data from:", DATA_PATH)
df = pd.read_excel(DATA_PATH)

print("\nData columns:", df.columns.tolist())
print("\nData shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Identify columns for QC Agent and Humans
# Assuming columns are named something like: 'QC_Agent', 'Human_1', 'Human_2', 'Human_3', etc.
# Or 'AI_Agent', 'Evaluator_1', 'Evaluator_2', 'Evaluator_3', etc.

# Let's identify the relevant columns
qc_agent_col = None
human_cols = []

for col in df.columns:
    col_lower = col.lower()
    if 'agent' in col_lower or 'ai' in col_lower:
        if 'human' not in col_lower:
            qc_agent_col = col
    elif 'human' in col_lower or 'evaluator' in col_lower or 'rater' in col_lower:
        human_cols.append(col)

print(f"\nQC Agent column: {qc_agent_col}")
print(f"Human columns: {human_cols}")

# If we can't auto-detect, try to find them manually
if qc_agent_col is None or len(human_cols) == 0:
    print("\nAttempting manual column detection...")
    # Look for columns with pass/fail values
    pass_fail_cols = []
    for col in df.columns:
        if col.lower() in ['chromatogram', 'sample', 'id']:
            continue
        unique_vals = df[col].dropna().unique()
        unique_vals_lower = [str(v).lower() for v in unique_vals]
        if any('pass' in v or 'fail' in v for v in unique_vals_lower):
            print(f"Column '{col}' contains pass/fail values: {unique_vals}")
            pass_fail_cols.append(col)
    
    # Identify QC Agent column if not found
    if qc_agent_col is None:
        for col in pass_fail_cols:
            if 'agent' in col.lower() or 'ai' in col.lower() or 'qc' in col.lower():
                qc_agent_col = col
                pass_fail_cols.remove(col)
                break
    
    # Remaining columns are human evaluators
    if len(human_cols) == 0:
        human_cols = [col for col in pass_fail_cols if col != qc_agent_col]

print(f"\nFinal QC Agent column: {qc_agent_col}")
print(f"Final Human columns: {human_cols}")

# Normalize the data to binary (1 for pass, 0 for fail)
def normalize_qc_value(val):
    """Convert QC assessment to binary: 1 for pass, 0 for fail"""
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower().strip()
    if 'pass' in val_str:
        return 1
    elif 'fail' in val_str:
        return 0
    else:
        return np.nan

# Create normalized dataframe
df_normalized = pd.DataFrame()

# Add chromatogram identifier if exists
if 'Chromatogram' in df.columns:
    df_normalized['Chromatogram'] = df['Chromatogram']
elif 'Sample' in df.columns:
    df_normalized['Chromatogram'] = df['Sample']
elif 'ID' in df.columns:
    df_normalized['Chromatogram'] = df['ID']
else:
    df_normalized['Chromatogram'] = range(1, len(df) + 1)

# Normalize QC Agent column
if qc_agent_col:
    df_normalized['QC_Agent'] = df[qc_agent_col].apply(normalize_qc_value)

# Normalize Human columns
for i, col in enumerate(human_cols, 1):
    df_normalized[f'Human_{i}'] = df[col].apply(normalize_qc_value)

print("\nNormalized data:")
print(df_normalized.head(10))

# Remove rows with all NaN values
df_normalized = df_normalized.dropna(how='all', subset=[c for c in df_normalized.columns if c != 'Chromatogram'])

# ============================================================================
# VISUALIZATION 1: HEATMAP - Green (pass) and Red (fail)
# ============================================================================

def create_heatmap():
    """Create heatmap showing pass/fail for QC Agent and all humans"""
    
    # Prepare data for heatmap - QC Agent at bottom
    heatmap_cols = [c for c in df_normalized.columns if c.startswith('Human_')] + ['QC_Agent']
    heatmap_data = df_normalized[heatmap_cols].T
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create custom colormap: Red for fail (0), Green for pass (1), White for NaN
    from matplotlib.colors import ListedColormap
    colors = ['#d32f2f', '#388e3c']  # Red, Green
    cmap = ListedColormap(colors)
    
    # Plot heatmap
    sns.heatmap(heatmap_data, 
                cmap=cmap, 
                cbar_kws={'label': 'QC Result', 'ticks': [0.25, 0.75]},
                linewidths=0.5,
                linecolor='white',
                ax=ax,
                vmin=0,
                vmax=1,
                square=False)
    
    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['Fail', 'Pass'])
    
    # Set labels
    ax.set_xlabel('Chromatogram', fontsize=14, fontweight='bold')
    ax.set_ylabel('')  # Remove y-axis label
    ax.set_title('QC Assessment Heatmap: QC Agent vs Human Evaluators', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Adjust y-axis labels - humans first, then QC Agent
    y_labels = [f'Human {i}' for i in range(1, len(human_cols) + 1)] + ['QC Agent']
    ax.set_yticklabels(y_labels, rotation=0)
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    png_path = OUTPUT_DIR / 'heatmap_qc_assessment.png'
    pdf_path = OUTPUT_DIR / 'heatmap_qc_assessment.pdf'
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"\nHeatmap saved to:")
    print(f"  - {png_path}")
    print(f"  - {pdf_path}")
    
    plt.close()

# ============================================================================
# VISUALIZATION 1B: HEATMAP WITH DISAGREEMENT OVERLAY
# ============================================================================

def create_heatmap_with_disagreement():
    """Create heatmap with hatching to show disagreements with QC Agent (ground truth)"""
    
    # Prepare data for heatmap - humans first, then QC Agent at bottom
    heatmap_cols = [c for c in df_normalized.columns if c.startswith('Human_')] + ['QC_Agent']
    heatmap_data = df_normalized[heatmap_cols].T
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create custom colormap: Red for fail (0), Green for pass (1)
    from matplotlib.colors import ListedColormap
    colors = ['#d32f2f', '#388e3c']  # Red, Green
    cmap = ListedColormap(colors)
    
    # Plot base heatmap
    sns.heatmap(heatmap_data, 
                cmap=cmap, 
                cbar_kws={'label': 'QC Result', 'ticks': [0.25, 0.75]},
                linewidths=0.5,
                linecolor='white',
                ax=ax,
                vmin=0,
                vmax=1,
                square=False)
    
    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['Fail', 'Pass'])
    
    # Add hatching for disagreements with QC Agent (only for humans, not QC Agent itself)
    qc_agent_results = df_normalized['QC_Agent'].values
    human_cols_only = [c for c in df_normalized.columns if c.startswith('Human_')]
    
    for i, human_col in enumerate(human_cols_only):
        human_results = df_normalized[human_col].values
        
        for j, (human_val, qc_val) in enumerate(zip(human_results, qc_agent_results)):
            # If human disagrees with QC Agent, add hatching
            if pd.notna(human_val) and pd.notna(qc_val) and human_val != qc_val:
                # Add diagonal lines (hatching) to indicate disagreement
                ax.add_patch(plt.Rectangle((j, i), 1, 1, 
                                          fill=False, 
                                          edgecolor='black', 
                                          hatch='///', 
                                          linewidth=0))
    
    # Set labels
    ax.set_xlabel('Chromatogram', fontsize=14, fontweight='bold')
    ax.set_ylabel('', fontsize=14, fontweight='bold')
    # No title
    
    # Adjust y-axis labels - humans first, then QC Agent
    y_labels = [f'Human {i}' for i in range(1, len(human_cols_only) + 1)] + ['QC Agent']
    ax.set_yticklabels(y_labels, rotation=0)
    
    # Add simplified legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#388e3c', label='Pass'),
        Patch(facecolor='#d32f2f', label='Fail'),
        Patch(facecolor='white', edgecolor='black', hatch='///', label='Incorrect')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1), 
             frameon=True, shadow=True)
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    png_path = OUTPUT_DIR / 'heatmap_disagreement_overlay.png'
    pdf_path = OUTPUT_DIR / 'heatmap_disagreement_overlay.pdf'
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"\nDisagreement overlay heatmap saved to:")
    print(f"  - {png_path}")
    print(f"  - {pdf_path}")
    
    plt.close()

# ============================================================================
# VISUALIZATION 2: BAR CHART - False Positives and False Negatives
# ============================================================================

def create_detailed_log():
    """Create detailed log file showing calculation for every chromatogram"""
    
    human_cols_list = [c for c in df_normalized.columns if c.startswith('Human_')]
    
    # Create detailed log data
    log_data = []
    
    for idx, row in df_normalized.iterrows():
        chrom_id = row['Chromatogram']
        qc_agent = row['QC_Agent']
        
        # Get human results
        human_results = {}
        human_values = []
        for i, col in enumerate(human_cols_list, 1):
            val = row[col]
            human_results[f'Human_{i}'] = 'Pass' if val == 1 else 'Fail' if val == 0 else 'N/A'
            if pd.notna(val):
                human_values.append(val)
        
        # Calculate human majority (2/3 or more)
        if len(human_values) > 0:
            human_consensus = np.mean(human_values)
            human_majority = 1 if human_consensus >= 0.67 else 0
            human_majority_str = 'Pass' if human_majority == 1 else 'Fail'
        else:
            human_majority = np.nan
            human_majority_str = 'N/A'
        
        # QC Agent result
        qc_agent_str = 'Pass' if qc_agent == 1 else 'Fail' if qc_agent == 0 else 'N/A'
        
        # Determine disagreement type
        disagreement_type = 'Agreement'
        if pd.notna(qc_agent) and pd.notna(human_majority):
            if qc_agent == 1 and human_majority == 0:
                disagreement_type = 'False Positive (AI Pass, Human Fail)'
            elif qc_agent == 0 and human_majority == 1:
                disagreement_type = 'False Negative (AI Fail, Human Pass)'
        
        # Build log entry
        log_entry = {
            'Chromatogram': chrom_id,
            'Human_1': human_results.get('Human_1', 'N/A'),
            'Human_2': human_results.get('Human_2', 'N/A'),
            'Human_3': human_results.get('Human_3', 'N/A'),
            'Human_Majority': human_majority_str,
            'QC_Agent_Ground_Truth': qc_agent_str,
            'Disagreement_Type': disagreement_type
        }
        
        log_data.append(log_entry)
    
    # Create DataFrame and save
    log_df = pd.DataFrame(log_data)
    
    # Save to CSV
    csv_path = OUTPUT_DIR / 'detailed_chromatogram_log.csv'
    log_df.to_csv(csv_path, index=False)
    
    print(f"\nDetailed chromatogram log saved to: {csv_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("DETAILED LOG SUMMARY")
    print("="*80)
    print(f"\nTotal chromatograms: {len(log_df)}")
    print(f"\nDisagreement breakdown:")
    print(log_df['Disagreement_Type'].value_counts())
    
    # Show examples of each type
    print("\n" + "="*80)
    print("EXAMPLES OF EACH DISAGREEMENT TYPE")
    print("="*80)
    
    # False Positives
    false_pos = log_df[log_df['Disagreement_Type'] == 'False Positive (AI Pass, Human Fail)']
    if len(false_pos) > 0:
        print(f"\nFalse Positives (AI Pass, Human Fail) - First 5 examples:")
        print(false_pos.head().to_string(index=False))
    
    # False Negatives
    false_neg = log_df[log_df['Disagreement_Type'] == 'False Negative (AI Fail, Human Pass)']
    if len(false_neg) > 0:
        print(f"\nFalse Negatives (AI Fail, Human Pass) - All examples:")
        print(false_neg.to_string(index=False))
    else:
        print(f"\nFalse Negatives (AI Fail, Human Pass): NONE FOUND")
    
    # Agreements
    agreements = log_df[log_df['Disagreement_Type'] == 'Agreement']
    print(f"\nAgreements: {len(agreements)} chromatograms")
    print(f"First 5 examples:")
    print(agreements.head().to_string(index=False))
    
    return log_df

def create_disagreement_barchart():
    """Create bar chart showing false positives and false negatives where humans disagree with AI"""
    
    # Calculate majority vote for humans (2/3 or more)
    human_cols_list = [c for c in df_normalized.columns if c.startswith('Human_')]
    
    # Calculate human consensus (majority vote)
    human_consensus = df_normalized[human_cols_list].mean(axis=1)
    # If >= 0.67 (2/3), it's a pass, otherwise fail
    human_majority = (human_consensus >= 0.67).astype(int)
    
    # Get valid comparisons (where both AI and human consensus have values)
    valid_mask = df_normalized['QC_Agent'].notna() & human_majority.notna()
    
    ai_results = df_normalized.loc[valid_mask, 'QC_Agent']
    human_results = human_majority.loc[valid_mask]
    
    # False Positive: AI says pass (1), Human consensus says fail (0)
    false_positives = ((ai_results == 1) & (human_results == 0)).sum()
    
    # False Negative: AI says fail (0), Human consensus says pass (1)
    false_negatives = ((ai_results == 0) & (human_results == 1)).sum()
    
    # Agreement
    agreements = (ai_results == human_results).sum()
    
    # Total comparisons
    total = len(ai_results)
    
    results_data = {
        'Evaluator': 'Human Consensus',
        'False Positives': false_positives,
        'False Negatives': false_negatives,
        'Agreements': agreements,
        'Total': total
    }
    
    print("\nDisagreement Analysis (Human Consensus - 2/3 Majority):")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Agreements: {agreements}")
    print(f"Total: {total}")
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 7))
    
    x = np.array([0, 1])  # Two separate positions for the bars
    width = 0.5
    
    bars1 = ax.bar(x[0], [false_positives], width, 
                   color='#d32f2f', edgecolor='black', linewidth=1.5)  # Red
    bars2 = ax.bar(x[1], [false_negatives], width, 
                   color='#1976d2', edgecolor='black', linewidth=1.5)  # Blue
    
    # Add value labels above bars
    ax.text(x[0], false_positives + 0.5, f'{int(false_positives)}',
           ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.text(x[1], false_negatives + 0.5, f'{int(false_negatives)}',
           ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Customize plot
    ax.set_xlabel('≥ 2/3 Majority', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    # No title
    ax.set_xticks(x)
    ax.set_xticklabels(['False Positive', 'False Negative'])
    # No legend
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save as PNG and PDF
    png_path = OUTPUT_DIR / 'barchart_disagreements.png'
    pdf_path = OUTPUT_DIR / 'barchart_disagreements.pdf'
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"\nBar chart saved to:")
    print(f"  - {png_path}")
    print(f"  - {pdf_path}")
    
    plt.close()
    
    # Save results to CSV
    csv_path = OUTPUT_DIR / 'disagreement_analysis.csv'
    results_df = pd.DataFrame([results_data])
    results_df.to_csv(csv_path, index=False)
    print(f"\nDisagreement analysis saved to: {csv_path}")

# ============================================================================
# ADDITIONAL VISUALIZATION: Agreement Rates
# ============================================================================

def create_agreement_summary():
    """Create summary statistics and agreement rate visualization"""
    
    human_cols_list = [c for c in df_normalized.columns if c.startswith('Human_')]
    
    agreement_data = []
    
    for human_col in human_cols_list:
        valid_mask = df_normalized['QC_Agent'].notna() & df_normalized[human_col].notna()
        
        ai_results = df_normalized.loc[valid_mask, 'QC_Agent']
        human_results = df_normalized.loc[valid_mask, human_col]
        
        agreements = (ai_results == human_results).sum()
        total = len(ai_results)
        agreement_rate = (agreements / total * 100) if total > 0 else 0
        
        human_num = human_col.split('_')[1]
        agreement_data.append({
            'Human': f'Human {human_num}',
            'Agreement Rate (%)': agreement_rate,
            'Agreements': agreements,
            'Total': total
        })
    
    agreement_df = pd.DataFrame(agreement_data)
    
    # Create bar chart for agreement rates
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(agreement_df['Human'], agreement_df['Agreement Rate (%)'], 
                  color='#6c5ce7', edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_xlabel('Human Evaluator', fontsize=14, fontweight='bold')
    ax.set_ylabel('Agreement Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('QC Agent vs Human Evaluators: Agreement Rates', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    png_path = OUTPUT_DIR / 'barchart_agreement_rates.png'
    pdf_path = OUTPUT_DIR / 'barchart_agreement_rates.pdf'
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"\nAgreement rate chart saved to:")
    print(f"  - {png_path}")
    print(f"  - {pdf_path}")
    
    plt.close()
    
    # Save to CSV
    csv_path = OUTPUT_DIR / 'agreement_rates.csv'
    agreement_df.to_csv(csv_path, index=False)
    print(f"\nAgreement rates saved to: {csv_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("QC AGENT VS HUMAN EVALUATORS - VISUALIZATION GENERATOR")
    print("="*80)
    
    try:
        # Generate detailed log first
        create_detailed_log()
        
        # Generate visualizations
        create_heatmap()
        create_heatmap_with_disagreement()
        create_disagreement_barchart()
        create_agreement_summary()
        
        print("\n" + "="*80)
        print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("="*80)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()