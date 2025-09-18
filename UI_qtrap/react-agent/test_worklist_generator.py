#!/usr/bin/env python3
"""
Test script for QC Worklist Generator

Tests the worklist generation functionality with sample data.
"""

import sys
import asyncio
import logging
from pathlib import Path
import pandas as pd

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from react_agent.qc_worklist_generator import (
    parse_filename_components,
    create_worklist_from_failed_qc,
    generate_worklist_for_project
)

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

def test_filename_parsing():
    """Test filename parsing with various examples"""
    print("🧪 Testing Filename Parsing")
    print("=" * 60)
    
    test_cases = [
        {
            'filename': '20250916_21MeOHACN_BrainLipidExtract_LC-PC_R-1_Op-Tom_Proj-Solvent01_PC_withSPLASH-NIST_1950_But_100x_1',
            'expected': {
                'Date': '20250916_21MeOHACN_BrainLipidExtract_LC-PC',
                'Info': '21MeOHACN',
                'Info2': 'BrainLipidExtract',
                'Lipid': 'PC',
                'Operator': 'Tom',
                'Project': 'Solvent01',
                'Technical_Replicate': '3'
            }
        },
        {
            'filename': '20250917_30MeOH_PlasmaExtract_LC-PE_R-2_Op-Sarah_Proj-Test02_PE_withSPLASH',
            'expected': {
                'Date': '20250917_30MeOH_PlasmaExtract_LC-PE',
                'Info': '30MeOH',
                'Info2': 'PlasmaExtract',
                'Lipid': 'PE',
                'Operator': 'Sarah',
                'Project': 'Test02',
                'Technical_Replicate': '3'
            }
        },
        {
            'filename': '20250918_15ACN_SerumExtract_LC-PS_R-3_Op-Mike_Proj-Control01_PS_standard',
            'expected': {
                'Date': '20250918_15ACN_SerumExtract_LC-PS',
                'Info': '15ACN',
                'Info2': 'SerumExtract',
                'Lipid': 'PS',
                'Operator': 'Mike',
                'Project': 'Control01',
                'Technical_Replicate': '3'
            }
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        filename = test_case['filename']
        expected = test_case['expected']
        
        print(f"\n📄 Test Case {i}:")
        print(f"   Filename: {filename}")
        
        result = parse_filename_components(filename)
        
        # Check each expected component
        test_passed = True
        for key, expected_value in expected.items():
            actual_value = result.get(key, '')
            if actual_value == expected_value:
                print(f"   ✅ {key}: '{actual_value}'")
            else:
                print(f"   ❌ {key}: Expected '{expected_value}', got '{actual_value}'")
                test_passed = False
                all_passed = False
        
        if test_passed:
            print(f"   🎉 Test Case {i}: PASSED")
        else:
            print(f"   💥 Test Case {i}: FAILED")
    
    print(f"\n📊 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    return all_passed

def create_sample_qc_results(project_name: str) -> Path:
    """Create sample QC results file for testing"""
    print(f"\n🔧 Creating Sample QC Results for: {project_name}")
    print("=" * 60)
    
    # Sample data with mix of pass/fail results
    sample_data = [
        {
            'QC_Result': 'pass',
            'Filename': '20250916_21MeOHACN_BrainLipidExtract_LC-PC_R-1_Op-Tom_Proj-Solvent01_PC_withSPLASH_1',
            'TIC_RSD_TopGroupWindow': 3.2,
            'TIC_RSD_WindowBest': 2.8,
            'Summed_TIC': 1250000.5,
            'TIC_Time': 15.2
        },
        {
            'QC_Result': 'fail',
            'Filename': '20250916_21MeOHACN_BrainLipidExtract_LC-PC_R-2_Op-Tom_Proj-Solvent01_PC_withSPLASH_2',
            'TIC_RSD_TopGroupWindow': 8.7,
            'TIC_RSD_WindowBest': 7.2,
            'Summed_TIC': 890000.3,
            'TIC_Time': 14.8
        },
        {
            'QC_Result': 'fail',
            'Filename': '20250917_30MeOH_PlasmaExtract_LC-PE_R-1_Op-Sarah_Proj-Solvent01_PE_standard_1',
            'TIC_RSD_TopGroupWindow': 12.1,
            'TIC_RSD_WindowBest': 10.5,
            'Summed_TIC': 750000.8,
            'TIC_Time': 16.1
        },
        {
            'QC_Result': 'pass',
            'Filename': '20250918_15ACN_SerumExtract_LC-PS_R-3_Op-Mike_Proj-Solvent01_PS_control_1',
            'TIC_RSD_TopGroupWindow': 2.8,
            'TIC_RSD_WindowBest': 2.1,
            'Summed_TIC': 1450000.2,
            'TIC_Time': 15.7
        },
        {
            'QC_Result': 'fail',
            'Filename': '20250919_25MeOH_TissueExtract_LC-PI_R-1_Op-Lisa_Proj-Solvent01_PI_test_1',
            'TIC_RSD_TopGroupWindow': 15.3,
            'TIC_RSD_WindowBest': 13.8,
            'Summed_TIC': 620000.1,
            'TIC_Time': 14.2
        }
    ]
    
    # Create results directory
    results_dir = Path(f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results/{project_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample QC results file
    sample_df = pd.DataFrame(sample_data)
    results_file = results_dir / f"QC_{project_name}_RESULTS.csv"
    sample_df.to_csv(results_file, index=False)
    
    # Summary
    total_results = len(sample_data)
    failed_results = len([r for r in sample_data if r['QC_Result'] == 'fail'])
    passed_results = total_results - failed_results
    
    print(f"✅ Created sample QC results: {results_file}")
    print(f"📊 Total results: {total_results}")
    print(f"   ✅ Passed: {passed_results}")
    print(f"   ❌ Failed: {failed_results}")
    
    print(f"\n📋 Sample QC Results:")
    print(sample_df[['QC_Result', 'Filename', 'TIC_RSD_TopGroupWindow']].to_string(index=False))
    
    return results_file

async def test_worklist_creation(project_name: str):
    """Test worklist creation functionality"""
    print(f"\n🔄 Testing Worklist Creation for: {project_name}")
    print("=" * 60)
    
    # Test the async worklist creation function
    worklist_file = await create_worklist_from_failed_qc(project_name)
    
    if worklist_file and worklist_file.exists():
        print(f"✅ Worklist created successfully: {worklist_file}")
        
        # Read and display the worklist
        worklist_df = pd.read_csv(worklist_file)
        print(f"\n📋 Generated Worklist ({len(worklist_df)} entries):")
        print(worklist_df.to_string(index=False))
        
        # Validate worklist structure
        expected_columns = ['Date', 'Info', 'Lipid', 'Technical_Replicate', 'Operator', 'Project', 'Info2']
        actual_columns = list(worklist_df.columns)
        
        if actual_columns == expected_columns:
            print(f"\n✅ Worklist structure is correct")
        else:
            print(f"\n❌ Worklist structure mismatch:")
            print(f"   Expected: {expected_columns}")
            print(f"   Actual: {actual_columns}")
        
        # Check that all entries have Technical_Replicate = 3
        if all(worklist_df['Technical_Replicate'] == '3'):
            print(f"✅ All Technical_Replicate values are correctly set to 3")
        else:
            print(f"❌ Some Technical_Replicate values are not 3")
        
        return True
    else:
        print(f"❌ Worklist creation failed or no failed files found")
        return False

def test_integration_function(project_name: str):
    """Test the integration function"""
    print(f"\n🔗 Testing Integration Function for: {project_name}")
    print("=" * 60)
    
    success = generate_worklist_for_project(project_name)
    
    if success:
        print(f"✅ Integration function successful")
        
        # Check if worklist file was created
        worklist_file = Path(f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/worklist/{project_name}/worklist_inputfile_{project_name}.csv")
        if worklist_file.exists():
            print(f"✅ Worklist file exists: {worklist_file}")
            
            # Show file size and modification time
            stat = worklist_file.stat()
            print(f"📊 File size: {stat.st_size} bytes")
            print(f"🕐 Modified: {pd.Timestamp.fromtimestamp(stat.st_mtime)}")
        else:
            print(f"❌ Worklist file not found: {worklist_file}")
    else:
        print(f"ℹ️ Integration function returned False (no failed files or error)")
    
    return success

async def main():
    """Main test function"""
    print("🚀 QC Worklist Generator - Test Suite")
    print("=" * 80)
    
    # Setup logging
    setup_logging()
    
    # Test 1: Filename Parsing
    parsing_success = test_filename_parsing()
    
    # Test 2: Create Sample Data and Test Worklist Creation
    test_project = "TestSample01"
    
    # Create sample QC results
    sample_file = create_sample_qc_results(test_project)
    
    # Test worklist creation
    worklist_success = await test_worklist_creation(test_project)
    
    # Test 3: Integration Function
    integration_success = test_integration_function(test_project)
    
    # Final Summary
    print(f"\n🎉 Test Suite Summary")
    print("=" * 80)
    print(f"📄 Filename Parsing: {'✅ PASSED' if parsing_success else '❌ FAILED'}")
    print(f"📋 Worklist Creation: {'✅ PASSED' if worklist_success else '❌ FAILED'}")
    print(f"🔗 Integration Function: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    
    overall_success = parsing_success and worklist_success and integration_success
    print(f"\n🏆 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print(f"\n🎯 Ready for Production!")
        print(f"📁 Sample worklist location:")
        print(f"   /home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/worklist/{test_project}/")
        print(f"\n🔧 Integration Instructions:")
        print(f"   The worklist generator is now integrated into Q_QC.py")
        print(f"   It will automatically create worklists after QC results generation")
        print(f"   Worklist files will be saved as: worklist_inputfile_{{project_name}}.csv")

if __name__ == "__main__":
    asyncio.run(main())
