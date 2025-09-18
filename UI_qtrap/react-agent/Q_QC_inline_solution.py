# Alternative Solution 3: Inline Function Solution
# Add these functions directly to Q_QC.py to eliminate import dependency

# ----------------------------------------------------------------------------
# Inline QC Worklist Generator Functions
# ----------------------------------------------------------------------------

def parse_filename_components_inline(filename: str) -> Dict[str, str]:
    """
    Parse complex filename into components (inline version).
    
    Example filename: 20250916_21MeOHACN_BrainLipidExtract_LC-PC_R-1_Op-Tom_Proj-Solvent01_PC_withSPLASH-NIST_1950_But_100x_1
    """
    logger = logging.getLogger()
    
    # Remove file extension if present
    base_filename = filename.replace('.txt', '').replace('.csv', '')
    
    # Split by underscores
    parts = base_filename.split('_')
    
    # Initialize result dictionary with defaults
    result = {
        'Date': '',
        'Info': '',
        'Info2': '',
        'Lipid': '',
        'Operator': '',
        'Project': '',
        'Technical_Replicate': '3'  # Always default to 3
    }
    
    try:
        # Extract basic parts (first 3 underscore-separated parts)
        if len(parts) >= 3:
            # Info is the second part
            result['Info'] = parts[1] if len(parts) > 1 else ''
            
            # Info2 is the third part
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
        
        logger.debug(f"Parsed filename '{filename}' -> {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error parsing filename '{filename}': {e}")
        # Return partial results with what we could extract
        return result


async def create_worklist_from_failed_qc_inline(project_name: str) -> Optional[Path]:
    """
    Create worklist CSV from failed QC results (inline version).
    """
    logger = logging.getLogger(f"qc_pipeline_{project_name}")
    
    try:
        # Define paths
        qc_results_dir = Path(f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results/{project_name}")
        qc_results_file = qc_results_dir / f"QC_{project_name}_RESULTS.csv"
        
        worklist_dir = Path(f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/worklist/{project_name}")
        worklist_file = worklist_dir / f"worklist_inputfile_{project_name}.csv"
        
        logger.info(f"🔍 Looking for QC results file: {qc_results_file}")
        
        # Check if QC results file exists
        if not await asyncio.to_thread(qc_results_file.exists):
            logger.warning(f"❌ QC results file not found: {qc_results_file}")
            return None
        
        # Read QC results
        logger.info(f"📖 Reading QC results from: {qc_results_file}")
        qc_df = await asyncio.to_thread(pd.read_csv, qc_results_file)
        
        logger.info(f"📊 QC results loaded: {len(qc_df)} total rows")
        
        # Filter for failed results
        if 'QC_Result' not in qc_df.columns:
            logger.error(f"❌ QC_Result column not found in {qc_results_file}")
            return None
        
        failed_df = qc_df[qc_df['QC_Result'] == 'fail'].copy()
        logger.info(f"🔍 Found {len(failed_df)} failed QC results")
        
        if len(failed_df) == 0:
            logger.info(f"✅ No failed QC results found for project {project_name} - no worklist needed")
            return None
        
        # Parse filenames and create worklist data
        worklist_data = []
        
        for _, row in failed_df.iterrows():
            filename = str(row['Filename'])
            logger.debug(f"Processing failed file: {filename}")
            
            # Parse filename components using inline function
            components = parse_filename_components_inline(filename)
            
            # Validate required components
            if not components['Project']:
                logger.warning(f"⚠️ Could not extract project from filename: {filename}")
                components['Project'] = project_name  # Use provided project name as fallback
            
            if not components['Date']:
                logger.warning(f"⚠️ Could not extract date from filename: {filename}")
                continue  # Skip this file as date is critical
            
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
            logger.debug(f"Added to worklist: {worklist_row}")
        
        if not worklist_data:
            logger.warning(f"⚠️ No valid worklist entries created from failed files")
            return None
        
        # Create worklist DataFrame
        worklist_df = pd.DataFrame(worklist_data)
        
        # Define column order
        column_order = ['Date', 'Info', 'Lipid', 'Technical_Replicate', 'Operator', 'Project', 'Info2']
        worklist_df = worklist_df[column_order]
        
        # Create output directory
        await asyncio.to_thread(worklist_dir.mkdir, parents=True, exist_ok=True)
        logger.info(f"📁 Created worklist directory: {worklist_dir}")
        
        # Save worklist file
        await asyncio.to_thread(worklist_df.to_csv, worklist_file, index=False)
        
        logger.info(f"✅ Worklist created successfully: {worklist_file}")
        logger.info(f"📊 Worklist contains {len(worklist_df)} entries for reprocessing")
        
        return worklist_file
        
    except Exception as e:
        logger.error(f"❌ Error creating worklist for project {project_name}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def generate_worklist_for_project_inline(project_name: str) -> bool:
    """
    Integration function for pipeline - generates worklist for a project (inline version).
    """
    logger = logging.getLogger(f"qc_pipeline_{project_name}")
    
    try:
        logger.info(f"🔄 Generating worklist for project: {project_name}")
        
        # Run the async worklist creation
        worklist_file = asyncio.run(create_worklist_from_failed_qc_inline(project_name))
        
        if worklist_file:
            logger.info(f"✅ Worklist generation successful: {worklist_file}")
            return True
        else:
            logger.info(f"ℹ️ No worklist generated (no failed files or error)")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in worklist generation for project {project_name}: {e}")
        return False

# Replace the import line with:
# generate_worklist_for_project = generate_worklist_for_project_inline
