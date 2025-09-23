#!/usr/bin/env python3
"""
Test script for the new QC monitoring every 1 minute for 10 minutes functionality.
This script demonstrates how to use the new monitoring function.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path so we can import the QC module
sys.path.insert(0, str(Path(__file__).parent / "src" / "react_agent"))

from Q_QC import (
    run_qc_monitoring_every_minute_for_10_minutes,
    run_qc_monitoring_session,
    extract_interval_and_duration_from_messages,
    extract_project_from_messages
)

async def test_monitoring_function():
    """Test the main monitoring function with a sample project."""
    print("🧪 Testing QC Monitoring Every 1 Minute for 10 Minutes")
    print("=" * 60)
    
    # Test project name
    project_name = "Solvent01"
    
    print(f"📊 Starting monitoring test for project: {project_name}")
    print(f"⏱️ This will run for exactly 10 minutes with 1-minute intervals")
    print(f"🔍 The system will check for WIFF files every minute")
    print(f"📝 All activities will be logged with detailed timestamps")
    print()
    
    try:
        # Run the monitoring session
        result = await run_qc_monitoring_every_minute_for_10_minutes(project_name)
        
        # Display results
        print("\n🎉 QC Monitoring Test Complete!")
        print("=" * 60)
        print(f"✅ Success: {result['success']}")
        print(f"📊 Project: {result['project_name']}")
        print(f"⏱️ Total Duration: {result['total_duration_seconds']} seconds")
        print(f"🔄 Completed Loops: {result['completed_loops']}")
        print(f"📁 Files Processed: {result['files_processed_total']}")
        print(f"⏭️ Files Skipped: {result['files_skipped_total']}")
        print(f"❌ Errors: {result['errors_encountered']}")
        print(f"📈 Success Rate: {result['success_rate']:.1f}%")
        
        if result.get('shutdown_requested'):
            print(f"🛑 Session was interrupted by user")
        
        print(f"\n📋 Monitoring Results:")
        for i, result_msg in enumerate(result['monitoring_results'][-5:], 1):  # Show last 5 results
            print(f"  {i}. {result_msg}")
            
    except Exception as e:
        print(f"❌ Error during monitoring test: {e}")
        import traceback
        traceback.print_exc()

def test_message_parsing():
    """Test the message parsing functionality for the new patterns."""
    print("\n🧪 Testing Message Parsing")
    print("=" * 40)
    
    # Test messages
    test_messages = [
        {'type': 'human', 'content': 'run QC monitoring every 1 minute for 10 minutes for project Solvent01'},
        {'type': 'human', 'content': 'QC monitoring every 1 minute for 10 minutes Solvent01'},
        {'type': 'human', 'content': 'monitor project Solvent01 every 1 minute for 10 minutes'},
        {'type': 'human', 'content': 'start QC for Solvent01 every 60 seconds for 600 seconds'},
    ]
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\nTest {i}: '{msg['content']}'")
        
        # Test interval and duration extraction
        interval, duration = extract_interval_and_duration_from_messages([msg])
        print(f"  Interval: {interval}")
        print(f"  Duration: {duration}")
        
        # Test project extraction
        project = extract_project_from_messages([msg])
        print(f"  Project: {project}")

async def test_convenience_function():
    """Test the convenience wrapper function."""
    print("\n🧪 Testing Convenience Function")
    print("=" * 40)
    
    project_name = "TestProject01"
    
    print(f"📊 Testing run_qc_monitoring_session() with project: {project_name}")
    
    try:
        # This should work the same as the main function
        result = await run_qc_monitoring_session(project_name)
        print(f"✅ Convenience function test completed")
        print(f"📊 Result: {result['success']}")
        
    except Exception as e:
        print(f"❌ Error in convenience function test: {e}")

def print_usage_examples():
    """Print usage examples for the new functionality."""
    print("\n📚 Usage Examples")
    print("=" * 40)
    
    print("1. Direct function call:")
    print("   result = await run_qc_monitoring_every_minute_for_10_minutes('Solvent01')")
    
    print("\n2. Convenience wrapper:")
    print("   result = await run_qc_monitoring_session('Solvent01')")
    
    print("\n3. Message-based triggering:")
    print("   - 'run QC monitoring every 1 minute for 10 minutes for project Solvent01'")
    print("   - 'QC monitoring every 1 minute for 10 minutes Solvent01'")
    print("   - 'monitor project Solvent01 every 1 minute for 10 minutes'")
    
    print("\n4. Expected log output structure:")
    print("   - Session start with 80 '=' characters")
    print("   - Loop iterations with 60 '-' characters")
    print("   - Step-by-step processing logs")
    print("   - File status summaries")
    print("   - Cumulative statistics")
    print("   - Session completion with final summary")

async def main():
    """Main test function."""
    print("🚀 QC Monitoring Every 1 Minute for 10 Minutes - Test Suite")
    print("=" * 70)
    
    # Test message parsing (synchronous)
    test_message_parsing()
    
    # Print usage examples
    print_usage_examples()
    
    # Ask user if they want to run the actual monitoring test
    print(f"\n⚠️  WARNING: The following tests will run actual monitoring sessions!")
    print(f"   - Each test runs for 10 minutes")
    print(f"   - Tests will check for real WIFF files")
    print(f"   - Logs will be written to project directories")
    
    response = input(f"\n❓ Do you want to run the actual monitoring tests? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print(f"\n🏃 Running actual monitoring tests...")
        
        # Test the main monitoring function
        await test_monitoring_function()
        
        # Test the convenience function
        # await test_convenience_function()  # Commented out to avoid double 10-minute runs
        
    else:
        print(f"\n✅ Skipping actual monitoring tests (they take 10 minutes each)")
        print(f"🧪 Message parsing and usage examples completed successfully")
    
    print(f"\n🎉 Test suite completed!")

if __name__ == "__main__":
    asyncio.run(main())
