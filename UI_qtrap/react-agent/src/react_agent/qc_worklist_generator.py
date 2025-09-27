#!/usr/bin/env python3
"""
QC Worklist Generator Script

Generates worklist files from failed QC results for mass spectrometry data processing.
Parses complex filenames and creates worklist CSV files for reprocessing failed samples.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import asyncio


def parse_filename_components(filename: str) -> Dict[str, str]:
    """
    Parse complex filename into components.
    
    Example filename: 20250916_21MeOHACN_BrainLipidExtract_LC-PC_R-1_Op-Tom_Proj-Solvent01_PC_withSPLASH-NIST_1950_But_100x_1
    
    Args:
        filename: The filename to parse (without extension)
    
    Returns:
        Dictionary with parsed components:
        - Date: 20250916_21MeOHACN_BrainLipidExtract_LC
        - Info: 21MeOHACN
        - Info2: BrainLipidExtract
        - Lipid: PC (from LC-PC pattern)
        - Operator: Tom (from Op-Tom pattern)
        - Project: Solvent01 (from Proj-Solvent01 pattern)
        - Technical_Replicate: 3 (always default)
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
        # Extract date from first part (should be YYYYMMDD format)
        if len(parts) >= 1:
            first_part = parts[0]
            # Extract date pattern (8 digits at start)
            date_match = re.match(r'^(\d{8})', first_part)
            if date_match:
                result['Date'] = date_match.group(1)
        
        # Extract basic parts (first 3 underscore-separated parts)
        if len(parts) >= 3:
            # Info is the second part
            result['Info'] = parts[1] if len(parts) > 1 else ''
            
            # Info2 is the third part
            result['Info2'] = parts[2] if len(parts) > 2 else ''
        
        # Find LC- pattern for Lipid extraction
        lc_pattern = None
        for part in parts:
            if part.startswith('LC-'):
                lc_pattern = part
                break
        
        if lc_pattern:
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


async def create_worklist_from_failed_qc(project_name: str) -> Optional[Path]:
    """
    Create worklist CSV from failed QC results.
    
    Args:
        project_name: The project name (e.g., 'Solvent01')
    
    Returns:
        Path to created worklist file, or None if no failed files or error
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
        logger.debug(f"QC results columns: {list(qc_df.columns)}")
        
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
            
            # Parse filename components
            components = parse_filename_components(filename)
            
            # Check if this is a blank sample - skip blanks as they don't need to be rerun
            if filename.upper().startswith('BLANK') or 'BLANK' in filename.upper():
                logger.info(f"🚫 Skipping blank sample (blanks don't need rerun): {filename}")
                continue
            
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
        
        # Log sample of worklist data
        logger.info("📋 Worklist sample (first 3 rows):")
        for i, (_, row) in enumerate(worklist_df.head(3).iterrows()):
            logger.info(f"  Row {i+1}: {dict(row)}")
        
        return worklist_file
        
    except Exception as e:
        logger.error(f"❌ Error creating worklist for project {project_name}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


async def generate_worklist_for_project_async(project_name: str) -> bool:
    """
    Async integration function for pipeline - generates worklist for a project.
    
    Args:
        project_name: The project name (e.g., 'Solvent01')
    
    Returns:
        True if worklist was created successfully, False otherwise
    """
    logger = logging.getLogger(f"qc_pipeline_{project_name}")
    
    try:
        logger.info(f"🔄 Generating worklist for project: {project_name}")
        
        # Run the async worklist creation
        worklist_file = await create_worklist_from_failed_qc(project_name)
        
        if worklist_file:
            logger.info(f"✅ Worklist generation successful: {worklist_file}")
            return True
        else:
            logger.info(f"ℹ️ No worklist generated (no failed files or error)")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in worklist generation for project {project_name}: {e}")
        return False


def generate_worklist_for_project(project_name: str) -> bool:
    """
    Sync integration function for pipeline - generates worklist for a project.
    
    Args:
        project_name: The project name (e.g., 'Solvent01')
    
    Returns:
        True if worklist was created successfully, False otherwise
    """
    logger = logging.getLogger(f"qc_pipeline_{project_name}")
    
    try:
        logger.info(f"🔄 Generating worklist for project: {project_name}")
        
        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, use asyncio.create_task or similar
            logger.warning("⚠️ Called from async context, using sync fallback")
            # Create a sync version by calling the async function directly
            import asyncio
            import concurrent.futures
            
            # Use a thread pool to run the async function
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, create_worklist_from_failed_qc(project_name))
                worklist_file = future.result()
                
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            worklist_file = asyncio.run(create_worklist_from_failed_qc(project_name))
        
        if worklist_file:
            logger.info(f"✅ Worklist generation successful: {worklist_file}")
            return True
        else:
            logger.info(f"ℹ️ No worklist generated (no failed files or error)")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in worklist generation for project {project_name}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


# ----------------------------------------------------------------------------
# Test Cases and Example Usage
# ----------------------------------------------------------------------------

def test_filename_parsing():
    """Test the filename parsing function with various examples."""
    print("🧪 Testing Filename Parsing")
    print("=" * 50)
    
    test_cases = [
        "20250916_21MeOHACN_BrainLipidExtract_LC-PC_R-1_Op-Tom_Proj-Solvent01_PC_withSPLASH-NIST_1950_But_100x_1",
        "20250917_30MeOH_PlasmaExtract_LC-PE_R-2_Op-Sarah_Proj-Test02_PE_withSPLASH",
        "20250918_15ACN_SerumExtract_LC-PS_R-3_Op-Mike_Proj-Control01_PS_standard",
        "malformed_filename_without_patterns",
        "20250919_BadFormat_NoLC_Op-John_Proj-BadTest"
    ]
    
    for filename in test_cases:
        print(f"\n📄 Testing: {filename}")
        result = parse_filename_components(filename)
        for key, value in result.items():
            print(f"  {key}: '{value}'")


async def test_worklist_creation():
    """Test worklist creation with mock data."""
    print("\n🧪 Testing Worklist Creation")
    print("=" * 50)
    
    # This would require actual QC results file to test
    # For demo purposes, showing the expected workflow
    test_project = "TestProject01"
    
    print(f"📋 Testing worklist creation for project: {test_project}")
    print("Note: This requires actual QC results file to test fully")
    
    # Show expected file paths
    qc_results_file = Path(f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results/{test_project}/QC_{test_project}_RESULTS.csv")
    worklist_file = Path(f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/worklist/{test_project}/worklist_inputfile_{test_project}.csv")
    
    print(f"📖 Expected QC results file: {qc_results_file}")
    print(f"📝 Expected worklist output: {worklist_file}")


def create_sample_qc_results(project_name: str) -> Path:
    """Create a sample QC results file for testing."""
    print(f"\n🔧 Creating sample QC results for testing: {project_name}")
    
    # Create sample data with some failed results
    sample_data = [
        {
            'QC_Result': 'pass',
            'Filename': '20250916_21MeOHACN_BrainLipidExtract_LC-PC_R-1_Op-Tom_Proj-Solvent01_PC_withSPLASH_1',
            'TIC_RSD_TopGroupWindow': 3.2
        },
        {
            'QC_Result': 'fail',
            'Filename': '20250916_21MeOHACN_BrainLipidExtract_LC-PC_R-2_Op-Tom_Proj-Solvent01_PC_withSPLASH_2',
            'TIC_RSD_TopGroupWindow': 8.7
        },
        {
            'QC_Result': 'fail',
            'Filename': '20250917_30MeOH_PlasmaExtract_LC-PE_R-1_Op-Sarah_Proj-Solvent01_PE_standard_1',
            'TIC_RSD_TopGroupWindow': 12.1
        },
        {
            'QC_Result': 'pass',
            'Filename': '20250918_15ACN_SerumExtract_LC-PS_R-3_Op-Mike_Proj-Solvent01_PS_control_1',
            'TIC_RSD_TopGroupWindow': 2.8
        }
    ]
    
    # Create directories
    results_dir = Path(f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results/{project_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample file
    sample_df = pd.DataFrame(sample_data)
    results_file = results_dir / f"QC_{project_name}_RESULTS.csv"
    sample_df.to_csv(results_file, index=False)
    
    print(f"✅ Created sample QC results: {results_file}")
    print(f"📊 Sample contains {len(sample_data)} results ({len([r for r in sample_data if r['QC_Result'] == 'fail'])} failed)")
    
    return results_file


async def main():
    """Main function for testing and demonstration."""
    print("🚀 QC Worklist Generator - Test Suite")
    print("=" * 60)
    
    # Test filename parsing
    test_filename_parsing()
    
    # Test worklist creation workflow
    await test_worklist_creation()
    
    # Create and test with sample data
    print("\n🧪 Testing with Sample Data")
    print("=" * 50)
    
    test_project = "TestSample01"
    
    # Create sample QC results
    sample_file = create_sample_qc_results(test_project)
    
    # Test worklist generation
    print(f"\n🔄 Testing worklist generation...")
    success = generate_worklist_for_project(test_project)
    
    if success:
        print("✅ Worklist generation test successful!")
        
        # Show the created worklist
        worklist_file = Path(f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/worklist/{test_project}/worklist_inputfile_{test_project}.csv")
        if worklist_file.exists():
            worklist_df = pd.read_csv(worklist_file)
            print(f"\n📋 Created Worklist ({len(worklist_df)} entries):")
            print(worklist_df.to_string(index=False))
    else:
        print("❌ Worklist generation test failed!")
    
    print("\n🎉 Test Suite Complete!")
    print("\n📋 Integration Instructions:")
    print("1. Import this module in Q_QC.py:")
    print("   from .qc_worklist_generator import generate_worklist_for_project")
    print("2. Add after qc_results() function:")
    print("   worklist_success = generate_worklist_for_project(project_name)")
    print("3. The worklist will be created automatically for failed QC results")


if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Run the test suite
    asyncio.run(main())
