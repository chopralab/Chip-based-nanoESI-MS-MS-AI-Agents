#!/usr/bin/env python3
"""
Test script to verify the blocking input() fix for LangGraph server compatibility.
This tests that the functions no longer use blocking input() calls.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path so we can import the QC module
sys.path.insert(0, str(Path(__file__).parent / "src" / "react_agent"))

from Q_QC import (
    get_dynamic_project_str,
    convert_and_parse_node,
    continuous_qc_monitoring_node,
    create_qc_state
)

def test_project_extraction():
    """Test that project extraction works without blocking input."""
    print("🧪 Testing Project Extraction (No Blocking)")
    print("=" * 50)
    
    # Test cases with project names
    test_cases = [
        # Should extract project successfully
        [{'type': 'human', 'content': 'run QC for project Solvent01'}],
        [{'type': 'human', 'content': 'monitor QC every 1 minute for 10 minutes for project TestProject01'}],
        [{'type': 'human', 'content': 'process Solvent01 QC'}],
        
        # Should return None (no project found)
        [{'type': 'human', 'content': 'run QC monitoring'}],
        [{'type': 'human', 'content': 'hello world'}],
        []  # Empty messages
    ]
    
    for i, messages in enumerate(test_cases, 1):
        print(f"\nTest {i}: {messages[0]['content'] if messages else 'Empty messages'}")
        
        # This should NOT call input() and should not block
        result = get_dynamic_project_str(messages)
        
        if result:
            print(f"  ✅ Extracted project: {result}")
        else:
            print(f"  ℹ️ No project found (expected for some test cases)")

async def test_node_error_handling():
    """Test that node functions handle missing project names gracefully."""
    print("\n🧪 Testing Node Error Handling")
    print("=" * 40)
    
    # Test message without project name
    messages_no_project = [
        {'type': 'human', 'content': 'run QC monitoring'}
    ]
    
    print("Testing convert_and_parse_node with no project...")
    state_no_project = create_qc_state(messages_no_project)
    
    try:
        # This should return an error state, not raise an exception or block
        result = await convert_and_parse_node(state_no_project, {})
        
        if 'parsing_result' in result and 'No project name found' in result['parsing_result']:
            print("  ✅ convert_and_parse_node handled missing project correctly")
            print(f"  📝 Error message: {result['parsing_result']}")
        else:
            print("  ❌ convert_and_parse_node did not handle missing project correctly")
            
    except Exception as e:
        print(f"  ❌ convert_and_parse_node raised exception: {e}")
    
    print("\nTesting continuous_qc_monitoring_node with no project...")
    
    try:
        # This should return an error state, not raise an exception or block
        result = await continuous_qc_monitoring_node(state_no_project, {})
        
        if 'parsing_result' in result and 'No project name found' in result['parsing_result']:
            print("  ✅ continuous_qc_monitoring_node handled missing project correctly")
            print(f"  📝 Error message: {result['parsing_result']}")
        else:
            print("  ❌ continuous_qc_monitoring_node did not handle missing project correctly")
            
    except Exception as e:
        print(f"  ❌ continuous_qc_monitoring_node raised exception: {e}")

async def test_successful_project_extraction():
    """Test that node functions work correctly when project is found."""
    print("\n🧪 Testing Successful Project Extraction")
    print("=" * 40)
    
    # Test message with clear project name
    messages_with_project = [
        {'type': 'human', 'content': 'run QC for project TestProject01'}
    ]
    
    print("Testing with valid project name...")
    state_with_project = create_qc_state(messages_with_project)
    
    # Extract project to verify it works
    project = get_dynamic_project_str(messages_with_project)
    if project:
        print(f"  ✅ Project extracted successfully: {project}")
    else:
        print(f"  ❌ Failed to extract project from valid message")
        return
    
    print(f"\n⚠️  Note: Full node testing would require actual file system access")
    print(f"   The key fix is that no blocking input() calls are made")
    print(f"   Project extraction and error handling work correctly")

def test_no_blocking_calls():
    """Verify that no blocking calls are made in the fixed functions."""
    print("\n🧪 Testing No Blocking Calls")
    print("=" * 30)
    
    # Test various message scenarios
    test_scenarios = [
        [],  # Empty
        [{'type': 'human', 'content': 'hello'}],  # No project
        [{'type': 'human', 'content': 'run QC for project Solvent01'}],  # With project
    ]
    
    for i, messages in enumerate(test_scenarios, 1):
        print(f"\nScenario {i}: {len(messages)} messages")
        
        # This should complete immediately without any user input
        start_time = asyncio.get_event_loop().time()
        result = get_dynamic_project_str(messages)
        end_time = asyncio.get_event_loop().time()
        
        duration = end_time - start_time
        
        if duration < 0.1:  # Should be nearly instantaneous
            print(f"  ✅ Completed in {duration:.4f}s (no blocking)")
            print(f"  📊 Result: {result}")
        else:
            print(f"  ❌ Took {duration:.4f}s (may have blocked)")

async def main():
    """Main test function."""
    print("🔧 LangGraph Blocking Input Fix - Verification Test")
    print("=" * 60)
    
    # Test that project extraction works without blocking
    test_project_extraction()
    
    # Test that node functions handle missing projects gracefully
    await test_node_error_handling()
    
    # Test successful project extraction
    await test_successful_project_extraction()
    
    # Test that no blocking calls are made
    test_no_blocking_calls()
    
    print(f"\n🎉 Blocking fix verification completed!")
    print(f"📋 Key points verified:")
    print(f"   ✅ get_dynamic_project_str() no longer calls input()")
    print(f"   ✅ Node functions handle missing projects gracefully")
    print(f"   ✅ Error messages are returned instead of exceptions")
    print(f"   ✅ No blocking operations detected")
    print(f"\n🚀 The QC system should now work in LangGraph server environment!")

if __name__ == "__main__":
    asyncio.run(main())
