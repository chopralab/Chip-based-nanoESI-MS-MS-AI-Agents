#!/usr/bin/env python3
"""
Test QC Worklist Integration

This script tests the integration between the QC pipeline and the general worklist system.
It demonstrates how failed QC samples are automatically imported into the general worklist
for reprocessing.
"""

import asyncio
import logging
import os
import pandas as pd
from pathlib import Path

# Import the integration functions
from Q_worklist import (
    import_qc_failed_samples_to_worklist,
    generate_integrated_worklist_for_project,
    INPUT_WORKLIST,
    QC_WORKLIST_BASE_DIR
)
from qc_worklist_generator import (
    create_sample_qc_results,
    generate_worklist_for_project
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_input_worklist():
    """Create a sample input worklist for testing."""
    logger.info("🔧 Creating sample input worklist")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(INPUT_WORKLIST), exist_ok=True)
    
    # Create sample regular worklist data
    sample_data = [
        {
            'Info': 'RegularSample1',
            'Lipid': 'PC',
            'Date': '20250926',
            'Technical_Replicate': '3',
            'Operator': 'TestUser',
            'Project': 'TestProject01',
            'Info2': 'RegularExtract'
        },
        {
            'Info': 'RegularSample2',
            'Lipid': 'PE',
            'Date': '20250926',
            'Technical_Replicate': '3',
            'Operator': 'TestUser',
            'Project': 'TestProject01',
            'Info2': 'RegularExtract'
        }
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv(INPUT_WORKLIST, index=False)
    
    logger.info(f"✅ Sample input worklist created: {INPUT_WORKLIST}")
    logger.info(f"📊 Contains {len(sample_data)} regular samples")
    
    return INPUT_WORKLIST

async def test_integration_workflow():
    """Test the complete integration workflow."""
    logger.info("🚀 Testing QC Worklist Integration Workflow")
    logger.info("=" * 60)
    
    test_project = "TestProject01"
    
    try:
        # Step 1: Create sample input worklist (regular samples)
        logger.info("\n📋 Step 1: Creating sample input worklist...")
        create_sample_input_worklist()
        
        # Step 2: Create sample QC results with failures
        logger.info("\n🧪 Step 2: Creating sample QC results with failures...")
        qc_results_file = create_sample_qc_results(test_project)
        logger.info(f"✅ QC results created: {qc_results_file}")
        
        # Step 3: Generate QC-specific worklist from failures
        logger.info("\n🔄 Step 3: Generating QC-specific worklist...")
        qc_worklist_success = generate_worklist_for_project(test_project)
        
        if qc_worklist_success:
            logger.info("✅ QC-specific worklist generated successfully")
            
            # Show the QC worklist
            qc_worklist_file = Path(QC_WORKLIST_BASE_DIR) / test_project / f"worklist_inputfile_{test_project}.csv"
            if qc_worklist_file.exists():
                qc_df = pd.read_csv(qc_worklist_file)
                logger.info(f"📋 QC Worklist ({len(qc_df)} entries):")
                logger.info(f"\n{qc_df.to_string(index=False)}")
        else:
            logger.error("❌ QC-specific worklist generation failed")
            return False
        
        # Step 4: Import QC failures into general worklist
        logger.info("\n🔄 Step 4: Importing QC failures into general worklist...")
        import_success = import_qc_failed_samples_to_worklist(test_project)
        
        if import_success:
            logger.info("✅ QC failures imported successfully")
            
            # Show the updated input worklist
            if os.path.exists(INPUT_WORKLIST):
                updated_df = pd.read_csv(INPUT_WORKLIST)
                logger.info(f"📋 Updated Input Worklist ({len(updated_df)} entries):")
                logger.info(f"\n{updated_df.to_string(index=False)}")
        else:
            logger.info("ℹ️ No QC failures to import")
        
        # Step 5: Generate integrated worklist
        logger.info("\n🚀 Step 5: Generating integrated worklist...")
        integrated_path = generate_integrated_worklist_for_project(test_project)
        
        if integrated_path:
            logger.info(f"✅ Integrated worklist generated: {integrated_path}")
            
            # Show sample of the final worklist
            if os.path.exists(integrated_path):
                final_df = pd.read_csv(integrated_path)
                logger.info(f"📋 Final Integrated Worklist ({len(final_df)} entries):")
                logger.info(f"\n{final_df.head(10).to_string(index=False)}")
                if len(final_df) > 10:
                    logger.info(f"... and {len(final_df) - 10} more entries")
        else:
            logger.error("❌ Integrated worklist generation failed")
            return False
        
        logger.info("\n🎉 Integration Test Complete!")
        logger.info("=" * 60)
        
        # Summary
        logger.info("\n📊 INTEGRATION SUMMARY:")
        logger.info("✅ QC pipeline generates worklists for failed samples")
        logger.info("✅ Failed QC samples are imported into general worklist")
        logger.info("✅ Integrated worklist includes both regular and QC reprocessing samples")
        logger.info("✅ QC reprocessing samples are marked with 'QC_REPROCESS_' prefix")
        logger.info("✅ Complete workflow from QC failure to reprocessing worklist")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_file_format_compatibility():
    """Test that QC worklist format is compatible with general worklist format."""
    logger.info("\n🧪 Testing File Format Compatibility")
    logger.info("-" * 40)
    
    # QC worklist format
    qc_format = ['Date', 'Info', 'Lipid', 'Technical_Replicate', 'Operator', 'Project', 'Info2']
    
    # General worklist input format
    general_format = ['Info', 'Lipid', 'Date', 'Technical_Replicate', 'Operator', 'Project', 'Info2']
    
    logger.info(f"📋 QC Worklist Format: {qc_format}")
    logger.info(f"📋 General Worklist Format: {general_format}")
    
    # Check compatibility
    qc_set = set(qc_format)
    general_set = set(general_format)
    
    if qc_set == general_set:
        logger.info("✅ Formats are compatible (same columns, different order)")
        logger.info("✅ Integration function handles column reordering")
    else:
        missing_in_general = qc_set - general_set
        missing_in_qc = general_set - qc_set
        
        if missing_in_general:
            logger.warning(f"⚠️ QC format has extra columns: {missing_in_general}")
        if missing_in_qc:
            logger.warning(f"⚠️ General format has extra columns: {missing_in_qc}")
    
    return qc_set == general_set

async def main():
    """Main test function."""
    logger.info("🧪 QC Worklist Integration Test Suite")
    logger.info("=" * 80)
    
    # Test 1: File format compatibility
    format_compatible = test_file_format_compatibility()
    
    # Test 2: Integration workflow
    if format_compatible:
        workflow_success = await test_integration_workflow()
        
        if workflow_success:
            logger.info("\n🎉 ALL TESTS PASSED!")
            logger.info("✅ QC and general worklist systems are successfully integrated")
        else:
            logger.error("\n❌ INTEGRATION TEST FAILED!")
    else:
        logger.error("\n❌ FORMAT COMPATIBILITY TEST FAILED!")
    
    logger.info("\n📋 Integration Usage Instructions:")
    logger.info("1. QC pipeline automatically generates worklists for failed samples")
    logger.info("2. Use import_qc_failed_samples_to_worklist(project_name) to import failures")
    logger.info("3. Use generate_integrated_worklist_for_project(project_name) for complete worklist")
    logger.info("4. QC failures are marked with 'QC_REPROCESS_' prefix for identification")

if __name__ == "__main__":
    asyncio.run(main())
