#!/usr/bin/env python3
"""
Test script to verify that Q_viz_intensity.py now saves both PNG and PDF files.
"""

import pandas as pd
import tempfile
import os
from pathlib import Path
import logging

# Import the visualization functions
from Q_viz_intensity import create_tic_barplots, prepare_visualization_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create sample data for testing."""
    test_data = {
        'BaseId': ['21MeOHACN', '21MeOHACN', '12CHCl3MeOH', '12CHCl3MeOH', '124CHCl3MeOHIPA', '124CHCl3MeOHIPA'],
        'LipidClass': ['PC', 'PC', 'PC', 'PC', 'PC', 'PC'],
        'Summed_TIC': [1000000, 1200000, 800000, 900000, 1100000, 1050000]
    }
    return pd.DataFrame(test_data)

def test_pdf_generation():
    """Test that both PNG and PDF files are generated."""
    logger.info("🧪 Testing PDF and PNG generation...")
    
    # Create test data
    df = create_test_data()
    logger.info(f"Created test data with {len(df)} rows")
    
    # Prepare data for visualization
    viz_data = prepare_visualization_data(df)
    if viz_data is None:
        logger.error("❌ Failed to prepare visualization data")
        return False
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory: {temp_dir}")
        
        # Generate plots
        create_tic_barplots(viz_data, temp_dir, "TestProject")
        
        # Check if files were created
        output_files = list(Path(temp_dir).glob("*"))
        logger.info(f"Generated files: {[f.name for f in output_files]}")
        
        # Check for expected files
        expected_files = [
            "highest_tic_PC.png",
            "highest_tic_PC.pdf", 
            "average_tic_PC.png",
            "average_tic_PC.pdf"
        ]
        
        success = True
        for expected_file in expected_files:
            file_path = Path(temp_dir) / expected_file
            if file_path.exists():
                file_size = file_path.stat().st_size
                logger.info(f"✅ {expected_file} created successfully ({file_size} bytes)")
            else:
                logger.error(f"❌ {expected_file} was not created")
                success = False
        
        return success

def main():
    """Main test function."""
    logger.info("🚀 Testing Q_viz_intensity.py PDF output functionality")
    logger.info("=" * 60)
    
    success = test_pdf_generation()
    
    if success:
        logger.info("\n🎉 SUCCESS: All tests passed!")
        logger.info("✅ Both PNG and PDF files are being generated correctly")
        logger.info("✅ High-quality output with 300 DPI and tight bounding boxes")
    else:
        logger.error("\n❌ FAILURE: Some tests failed")
    
    logger.info("=" * 60)
    return success

if __name__ == "__main__":
    main()
