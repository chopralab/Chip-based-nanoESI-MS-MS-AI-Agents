# Solution 3: Inline Functions - Add these directly to Q_QC.py

# Replace the import section with these inline functions:

# ----------------------------------------------------------------------------
# Inline QC Worklist Generator Functions
# ----------------------------------------------------------------------------

def parse_filename_components_inline(filename: str) -> Dict[str, str]:
    """Parse complex filename into components (inline version)."""
    logger = logging.getLogger()
    
    # Remove file extension if present
    base_filename = filename.replace('.txt', '').replace('.csv', '')
    parts = base_filename.split('_')
    
    # Initialize result dictionary with defaults
    result = {
        'Date': '', 'Info': '', 'Info2': '', 'Lipid': '',
        'Operator': '', 'Project': '', 'Technical_Replicate': '3'
    }
    
    try:
        # Extract basic parts
        if len(parts) >= 3:
            result['Info'] = parts[1] if len(parts) > 1 else ''
            result['Info2'] = parts[2] if len(parts) > 2 else ''
        
        # Find LC- pattern for Date and Lipid extraction
        lc_pattern = None
        lc_index = -1
        for i, part in enumerate(parts):
            if part.startswith('LC-'):
                lc_pattern = part
                lc_index = i
                break
        
        if lc_pattern and lc_index >= 0:
            # Date: everything up to and including LC-
            date_parts = parts[:lc_index + 1]
            result['Date'] = '_'.join(date_parts)
            
            # Lipid: extract from LC-{lipid} pattern
            lipid_match = re.match(r'LC-(.+)', lc_pattern)
            if lipid_match:
                result['Lipid'] = lipid_match.group(1)
        
        # Extract Operator from Op-{name} pattern
        for part in parts:
            op_match = re.match(r'Op-(.+)', part)
            if op_match:
                result['Operator'] = op_match.group(1)
                break
        
        # Extract Project from Proj-{name} pattern
        for part in parts:
            proj_match = re.match(r'Proj-(.+)', part)
            if proj_match:
                result['Project'] = proj_match.group(1)
                break
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing filename '{filename}': {e}")
        return result


def generate_worklist_for_project_inline(project_name: str) -> bool:
    """Generate worklist for failed QC results (inline version)."""
    logger = logging.getLogger(f"qc_pipeline_{project_name}")
    
    try:
        # Define paths
        qc_results_dir = Path(f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results/{project_name}")
        qc_results_file = qc_results_dir / f"QC_{project_name}_RESULTS.csv"
        worklist_dir = Path(f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/worklist/{project_name}")
        worklist_file = worklist_dir / f"worklist_inputfile_{project_name}.csv"
        
        # Check if QC results file exists
        if not qc_results_file.exists():
            logger.warning(f"❌ QC results file not found: {qc_results_file}")
            return False
        
        # Read QC results
        qc_df = pd.read_csv(qc_results_file)
        
        # Filter for failed results
        if 'QC_Result' not in qc_df.columns:
            logger.error(f"❌ QC_Result column not found")
            return False
        
        failed_df = qc_df[qc_df['QC_Result'] == 'fail'].copy()
        
        if len(failed_df) == 0:
            logger.info(f"✅ No failed QC results found - no worklist needed")
            return False
        
        # Parse filenames and create worklist data
        worklist_data = []
        
        for _, row in failed_df.iterrows():
            filename = str(row['Filename'])
            components = parse_filename_components_inline(filename)
            
            # Validate required components
            if not components['Project']:
                components['Project'] = project_name
            
            if not components['Date']:
                logger.warning(f"⚠️ Could not extract date from filename: {filename}")
                continue
            
            # Add to worklist data
            worklist_row = {
                'Date': components['Date'],
                'Info': components['Info'],
                'Lipid': components['Lipid'],
                'Technical_Replicate': components['Technical_Replicate'],
                'Operator': components['Operator'],
                'Project': components['Project'],
                'Info2': components['Info2']
            }
            
            worklist_data.append(worklist_row)
        
        if not worklist_data:
            logger.warning(f"⚠️ No valid worklist entries created")
            return False
        
        # Create worklist DataFrame
        worklist_df = pd.DataFrame(worklist_data)
        column_order = ['Date', 'Info', 'Lipid', 'Technical_Replicate', 'Operator', 'Project', 'Info2']
        worklist_df = worklist_df[column_order]
        
        # Create output directory and save
        worklist_dir.mkdir(parents=True, exist_ok=True)
        worklist_df.to_csv(worklist_file, index=False)
        
        logger.info(f"✅ Worklist created: {worklist_file} ({len(worklist_df)} entries)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error generating worklist: {e}")
        return False

# Then replace the function calls in the pipeline with:
# generate_worklist_for_project_inline(project_name)
