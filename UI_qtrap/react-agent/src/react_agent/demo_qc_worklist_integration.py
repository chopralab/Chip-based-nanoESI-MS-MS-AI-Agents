#!/usr/bin/env python3
"""
Demo: QC Worklist Integration

This script demonstrates how the integrated QC worklist system works.
It shows the complete workflow from QC failure detection to reprocessing worklist generation.
"""

import asyncio
import logging
from pathlib import Path

# Import the integration functions
from Q_worklist import generate_integrated_worklist_for_project
from qc_worklist_generator import generate_worklist_for_project

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

async def demo_integration(project_name: str):
    """
    Demonstrate the QC worklist integration for a specific project.
    
    Args:
        project_name: The project name (e.g., 'Solvent01')
    """
    logger.info(f"🚀 QC Worklist Integration Demo for Project: {project_name}")
    logger.info("=" * 70)
    
    try:
        # Step 1: Generate QC-specific worklist (this happens automatically in QC pipeline)
        logger.info(f"\n📋 Step 1: Generating QC-specific worklist for failed samples...")
        qc_success = generate_worklist_for_project(project_name)
        
        if qc_success:
            logger.info(f"✅ QC-specific worklist created for project {project_name}")
            logger.info(f"📁 Location: /data/qc/worklist/{project_name}/worklist_inputfile_{project_name}.csv")
        else:
            logger.info(f"ℹ️ No failed QC samples found for project {project_name}")
        
        # Step 2: Generate integrated worklist (includes QC failures + regular samples)
        logger.info(f"\n🔄 Step 2: Generating integrated worklist...")
        integrated_path = generate_integrated_worklist_for_project(project_name)
        
        if integrated_path:
            logger.info(f"✅ Integrated worklist generated successfully!")
            logger.info(f"📁 Location: {integrated_path}")
            logger.info(f"🔄 This worklist includes:")
            logger.info(f"   • Regular samples from input worklist")
            logger.info(f"   • Failed QC samples for reprocessing (marked with QC_REPROCESS_)")
            logger.info(f"   • Proper method assignments from methods.csv")
            logger.info(f"   • Appropriate blanks and replicates")
        else:
            logger.warning(f"⚠️ Could not generate integrated worklist for project {project_name}")
        
        logger.info(f"\n🎉 Demo Complete for Project {project_name}!")
        
        return integrated_path is not None
        
    except Exception as e:
        logger.error(f"❌ Demo failed for project {project_name}: {e}")
        return False

def show_usage_examples():
    """Show usage examples for the integrated system."""
    logger.info("\n📋 USAGE EXAMPLES:")
    logger.info("=" * 50)
    
    logger.info("\n1. 🔄 Automatic Integration (in QC Pipeline):")
    logger.info("   The QC pipeline (Q_QC.py) automatically calls both:")
    logger.info("   • generate_worklist_for_project(project_name)  # QC-specific")
    logger.info("   • generate_integrated_worklist_for_project(project_name)  # Complete")
    
    logger.info("\n2. 🛠️ Manual Integration:")
    logger.info("   from Q_worklist import generate_integrated_worklist_for_project")
    logger.info("   worklist_path = generate_integrated_worklist_for_project('Solvent01')")
    
    logger.info("\n3. 📊 LangGraph Integration:")
    logger.info("   Use the qc_integrated_worklist_node in your LangGraph workflow")
    logger.info("   Message: 'generate integrated worklist for project Solvent01'")
    
    logger.info("\n4. 🔍 File Locations:")
    logger.info("   • QC Results: /data/qc/results/{project}/QC_{project}_RESULTS.csv")
    logger.info("   • QC Worklist: /data/qc/worklist/{project}/worklist_inputfile_{project}.csv")
    logger.info("   • Input Worklist: /data/worklist/input/input_worklist.csv")
    logger.info("   • Final Worklist: /data/worklist/generated/aggregated_worklist_YYYYMMDD_HHMMSS.csv")

def show_integration_benefits():
    """Show the benefits of the integrated system."""
    logger.info("\n🎯 INTEGRATION BENEFITS:")
    logger.info("=" * 50)
    
    benefits = [
        "✅ Automatic QC failure detection and worklist generation",
        "✅ Seamless integration between QC and general worklist systems",
        "✅ Failed samples automatically marked for reprocessing",
        "✅ No manual intervention required for QC failure handling",
        "✅ Unified worklist format for all sample types",
        "✅ Proper method assignments and replicate handling",
        "✅ Comprehensive logging and error handling",
        "✅ Compatible with existing LangGraph workflow",
        "✅ Maintains separation of concerns (QC vs general worklist)",
        "✅ Scalable for multiple projects and sample types"
    ]
    
    for benefit in benefits:
        logger.info(f"   {benefit}")

async def main():
    """Main demo function."""
    logger.info("🚀 QC Worklist Integration Demo")
    logger.info("=" * 80)
    
    # Demo with example projects
    test_projects = ["Solvent01", "TestProject01"]
    
    for project in test_projects:
        success = await demo_integration(project)
        if not success:
            logger.warning(f"⚠️ Demo incomplete for project {project} (may not have QC results)")
    
    # Show usage examples and benefits
    show_usage_examples()
    show_integration_benefits()
    
    logger.info("\n🎉 QC Worklist Integration Demo Complete!")
    logger.info("=" * 80)
    
    logger.info("\n📋 NEXT STEPS:")
    logger.info("1. Run your QC pipeline - worklists will be generated automatically")
    logger.info("2. Check /data/worklist/generated/ for integrated worklists")
    logger.info("3. Use the generated worklists for sample reprocessing")
    logger.info("4. Monitor QC results to track reprocessing success")

if __name__ == "__main__":
    asyncio.run(main())
