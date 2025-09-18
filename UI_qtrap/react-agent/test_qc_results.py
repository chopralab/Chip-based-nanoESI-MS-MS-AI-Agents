#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, '/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src')

from react_agent.Q_QC import qc_results, setup_logging

async def test_qc_results():
    project_name = "Solvent01"
    
    print(f"Testing qc_results for project: {project_name}")
    
    try:
        # Setup logging first
        logger = await setup_logging(project_name)
        print("✅ Logging setup successful")
        
        # Run qc_results
        await qc_results(project_name)
        print("✅ qc_results completed successfully")
        
        # Check if results file was created
        results_path = f"/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/qc/results/{project_name}/QC_{project_name}_RESULTS.csv"
        if os.path.exists(results_path):
            print(f"✅ Results file created: {results_path}")
        else:
            print(f"❌ Results file NOT found: {results_path}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_qc_results())
