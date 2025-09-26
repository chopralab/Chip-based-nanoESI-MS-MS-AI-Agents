import pandas as pd
import re
import numpy as np
import os
from typing import Dict, Optional, Tuple

def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse filename to extract base identifier, lipid class, and replicate number.
    
    Args:
        filename: The filename to parse
        
    Returns:
        Dictionary with parsed components or None if parsing fails
    """
    if pd.isna(filename) or not isinstance(filename, str):
        return None
    
    # Split by underscores to get base parts
    parts = filename.split('_')
    if len(parts) < 4:
        return None
    
    # Get the base identifier (first 3 parts joined with underscores)
    base_id = '_'.join(parts[:3])
    
    # Find LC- and R- components using regex
    lc_match = re.search(r'LC-([^_]+)', filename)
    r_match = re.search(r'R-(\d+)', filename)
    
    if not lc_match or not r_match:
        return None
    
    return {
        'base_id': base_id,
        'lipid_class': lc_match.group(1),
        'replicate': int(r_match.group(1)),
        'full_filename': filename
    }

def create_sort_key(parsed_info: Optional[Dict[str, str]]) -> str:
    """
    Create a sorting key for proper organization.
    
    Args:
        parsed_info: Dictionary with parsed filename components
        
    Returns:
        String key for sorting
    """
    if not parsed_info:
        return 'zzz'  # Put unparseable files at the end
    
    return f"{parsed_info['base_id']}_LC-{parsed_info['lipid_class']}_R-{parsed_info['replicate']:02d}"

def calculate_replicate_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate replicate averages for each base_id + lipid_class combination.
    
    Args:
        df: DataFrame with parsed filename information
        
    Returns:
        DataFrame with replicate averages
    """
    # Define numeric columns to average
    numeric_columns = ['TIC_RSD_TopGroupWindow', 'TIC_RSD_WindowBest', 'Summed_TIC', 'TIC_Time']
    
    # Group by base_id and lipid_class
    grouped = df.groupby(['base_id', 'lipid_class'])
    
    averages_list = []
    
    for (base_id, lipid_class), group in grouped:
        avg_dict = {
            'base_id': base_id,
            'lipid_class': lipid_class,
            'replicate_count': len(group),
            'pass_count': len(group[group['QC_Result'] == 'pass']),
            'fail_count': len(group[group['QC_Result'] == 'fail']),
        }
        
        # Calculate pass rate
        avg_dict['pass_rate'] = f"{avg_dict['pass_count']}/{avg_dict['replicate_count']}"
        
        # Calculate averages for numeric columns
        for col in numeric_columns:
            values = group[col].dropna()
            if len(values) > 0:
                avg_dict[f'avg_{col.lower()}'] = values.mean()
            else:
                avg_dict[f'avg_{col.lower()}'] = np.nan

        # Find highest Summed_TIC in the group
        max_tic_values = group['Summed_TIC'].dropna()
        if len(max_tic_values) > 0:
            avg_dict['highest_tic_per_group'] = max_tic_values.max()
        else:
            avg_dict['highest_tic_per_group'] = np.nan
        
        averages_list.append(avg_dict)
    
    return pd.DataFrame(averages_list)

def process_lipid_data(csv_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to process lipid data: organize and calculate replicate averages.
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        Tuple of (organized_data, replicate_averages)
    """
    
    # Read the CSV file
    print(f"Reading CSV file: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    print(f"Original data shape: {df.shape}")
    
    # Filter out BLANK entries
    df_filtered = df[~df['Filename'].str.contains('BLANK', na=False)].copy()
    print(f"After removing BLANK entries: {df_filtered.shape}")
    
    # Parse filenames
    print("Parsing filenames...")
    df_filtered['parsed_info'] = df_filtered['Filename'].apply(parse_filename)
    
    # Filter out rows that couldn't be parsed
    valid_rows = df_filtered['parsed_info'].notna()
    df_valid = df_filtered[valid_rows].copy()
    
    if len(df_valid) == 0:
        print("Warning: No valid rows found after parsing filenames!")
        return df_filtered, pd.DataFrame()
    
    print(f"Successfully parsed: {len(df_valid)} rows")
    
    # Extract parsed components into separate columns
    df_valid['base_id'] = df_valid['parsed_info'].apply(lambda x: x['base_id'] if x else None)
    df_valid['lipid_class'] = df_valid['parsed_info'].apply(lambda x: x['lipid_class'] if x else None)
    df_valid['replicate'] = df_valid['parsed_info'].apply(lambda x: x['replicate'] if x else None)
    
    # Create sort key and sort
    df_valid['sort_key'] = df_valid['parsed_info'].apply(create_sort_key)
    df_sorted = df_valid.sort_values('sort_key').copy()
    
    # Calculate replicate averages
    print("Calculating replicate averages...")
    replicate_averages = calculate_replicate_averages(df_sorted)
    
    # Merge averages back to main dataframe
    df_final = df_sorted.merge(
        replicate_averages,
        on=['base_id', 'lipid_class'],
        how='left',
        suffixes=('', '_avg')
    )
    
    # Select and rename columns for final output
    final_columns = [
        'QC_Result',
        'Filename', 
        'TIC_RSD_TopGroupWindow',
        'TIC_RSD_WindowBest',
        'Summed_TIC',
        'TIC_Time',
        'TIC_RSD_Window',
        'base_id',
        'lipid_class', 
        'replicate',
        'avg_tic_rsd_topgroupwindow',
        'avg_tic_rsd_windowbest',
        'avg_summed_tic',
        'avg_tic_time',
        'pass_rate',
        'highest_tic_per_group',
        'replicate_count'
    ]
    
    # Rename columns for clarity
    column_mapping = {
        'base_id': 'BaseId',
        'lipid_class': 'LipidClass',
        'replicate': 'Replicate',
        'avg_tic_rsd_topgroupwindow': 'Avg_TIC_RSD_TopGroupWindow',
        'avg_tic_rsd_windowbest': 'Avg_TIC_RSD_WindowBest', 
        'avg_summed_tic': 'Avg_Summed_TIC',
        'avg_tic_time': 'Avg_TIC_Time',
        'pass_rate': 'Pass_Rate',
        'replicate_count': 'Replicate_Count',
        'highest_tic_per_group': 'Highest_TIC_Per_Group'
    }
    
    df_output = df_final[final_columns].rename(columns=column_mapping)
    
    return df_output, replicate_averages

def main():
    """
    Main execution function with example usage.
    """
    # Define input and output paths
    input_file = "/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results/solventmatrix/QC_solventmatrix_RESULTS.csv"
    output_dir = "/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results/solventmatrix/organized"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file paths
    base_filename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, f"organized_{base_filename}")
    averages_file = os.path.join(output_dir, f"replicate_averages_{base_filename}")

    try:
        # Process the data
        organized_data, replicate_averages = process_lipid_data(input_file)
        
        # Save organized data
        if not organized_data.empty:
            organized_data.to_csv(output_file, index=False)
            print(f"\nOrganized data saved to: {output_file}")
        else:
            print("\nNo organized data to save.")

        # Save replicate averages summary
        if not replicate_averages.empty:
            replicate_averages.to_csv(averages_file, index=False)
            print(f"Replicate averages saved to: {averages_file}")
        else:
            print("No replicate averages to save.")
        
        # Display summary
        print(f"\nSummary:")
        print(f"Total organized rows: {len(organized_data)}")
        print(f"Unique replicate groups: {len(replicate_averages)}")
        
        # Show first few organized rows
        print(f"\nFirst 5 organized rows:")
        for idx, row in organized_data.head().iterrows():
            print(f"{idx+1}. {row['Filename']}")
            print(f"   Base: {row['BaseId']}, Lipid: {row['LipidClass']}, Rep: {row['Replicate']}")
        
        # Show replicate averages summary
        print(f"\nReplicate Averages Summary (first 3 groups):")
        for idx, row in replicate_averages.head(3).iterrows():
            print(f"{row['base_id']}_LC-{row['lipid_class']}:")
            print(f"  Pass Rate: {row['pass_rate']}")
            print(f"  Avg TIC_RSD_WindowBest: {row['avg_tic_rsd_windowbest']:.2f}" if not pd.isna(row['avg_tic_rsd_windowbest']) else "  Avg TIC_RSD_WindowBest: N/A")
            print(f"  Avg Summed_TIC: {row['avg_summed_tic']:.0f}" if not pd.isna(row['avg_summed_tic']) else "  Avg Summed_TIC: N/A")
            print(f"  Highest TIC in Group: {row['highest_tic_per_group']:.0f}" if not pd.isna(row['highest_tic_per_group']) else "  Highest TIC in Group: N/A")
            
    except FileNotFoundError:
        print(f"Error: Could not find file '{input_file}'. Please check the file path.")
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    main()