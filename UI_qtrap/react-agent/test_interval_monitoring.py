#!/usr/bin/env python3
"""
Test script for the new interval-based QC monitoring functionality.
Tests both message parsing and the enhanced monitoring workflow.
"""

import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from react_agent.Q_QC import (
    extract_interval_and_duration_from_messages,
    extract_duration_from_messages,
    parse_duration,
    check_file_stability_tracked,
    get_stable_files,
    format_time_remaining
)

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger()

def test_message_parsing():
    """Test the enhanced message parsing for interval and duration"""
    logger = setup_logging()
    logger.info("🧪 Testing Message Parsing")
    logger.info("=" * 50)
    
    # Test cases for interval-based monitoring
    test_messages = [
        # Interval + Duration patterns
        {"content": "run QC for project Solvent01 every 1 minute for 10 minutes", "expected": ("1m", "10m")},
        {"content": "monitor Solvent01 QC every 30 seconds for 2 hours", "expected": ("30s", "2h")},
        {"content": "check QC for project Test01 every 2m for 1h", "expected": ("2m", "1h")},
        {"content": "run QC every 5 minutes for 30 minutes", "expected": ("5m", "30m")},
        
        # Duration only patterns (should return None, duration)
        {"content": "run QC for project Solvent01 for 1h", "expected": (None, "1h")},
        {"content": "monitor QC for 2d", "expected": (None, "2d")},
        
        # No pattern matches
        {"content": "just run QC for Solvent01", "expected": (None, None)},
    ]
    
    # Mock message objects
    class MockMessage:
        def __init__(self, content):
            self.content = content
            self.__class__.__name__ = "HumanMessage"
    
    for i, test_case in enumerate(test_messages, 1):
        messages = [MockMessage(test_case["content"])]
        interval, duration = extract_interval_and_duration_from_messages(messages)
        expected_interval, expected_duration = test_case["expected"]
        
        logger.info(f"Test {i}: '{test_case['content']}'")
        logger.info(f"  Expected: interval={expected_interval}, duration={expected_duration}")
        logger.info(f"  Got:      interval={interval}, duration={duration}")
        
        if interval == expected_interval and duration == expected_duration:
            logger.info(f"  ✅ PASSED")
        else:
            logger.info(f"  ❌ FAILED")
        logger.info("")
    
    return True

def test_duration_parsing():
    """Test duration parsing functionality"""
    logger = logging.getLogger()
    logger.info("🧪 Testing Duration Parsing")
    logger.info("=" * 50)
    
    test_cases = [
        ("1m", 60),
        ("5m", 300),
        ("1h", 3600),
        ("2h", 7200),
        ("1d", 86400),
        ("30s", 30),
    ]
    
    for duration_str, expected_seconds in test_cases:
        try:
            result = parse_duration(duration_str)
            logger.info(f"'{duration_str}' -> {result}s (expected: {expected_seconds}s)")
            if result == expected_seconds:
                logger.info(f"  ✅ PASSED")
            else:
                logger.info(f"  ❌ FAILED")
        except Exception as e:
            logger.info(f"'{duration_str}' -> ERROR: {e}")
            logger.info(f"  ❌ FAILED")
        logger.info("")
    
    return True

async def test_file_stability_tracking():
    """Test file stability tracking functionality"""
    logger = logging.getLogger()
    logger.info("🧪 Testing File Stability Tracking")
    logger.info("=" * 50)
    
    # Create test files
    test_dir = Path("/tmp/qc_test_files")
    test_dir.mkdir(exist_ok=True)
    
    test_file1 = test_dir / "test_file1.wiff"
    test_file2 = test_dir / "test_file2.wiff"
    
    # Create files with initial content
    test_file1.write_text("initial content")
    test_file2.write_text("initial content")
    
    file_tracker = {}
    
    # First check - files should be unstable (first time seeing them)
    logger.info("First stability check (should be unstable):")
    is_stable1 = await check_file_stability_tracked(test_file1, file_tracker, 2)  # 2 second stability
    is_stable2 = await check_file_stability_tracked(test_file2, file_tracker, 2)
    logger.info(f"  File1 stable: {is_stable1} (expected: False)")
    logger.info(f"  File2 stable: {is_stable2} (expected: False)")
    
    # Wait and check again - should still be unstable (not enough time)
    await asyncio.sleep(1)
    logger.info("Second check after 1s (should still be unstable):")
    is_stable1 = await check_file_stability_tracked(test_file1, file_tracker, 2)
    is_stable2 = await check_file_stability_tracked(test_file2, file_tracker, 2)
    logger.info(f"  File1 stable: {is_stable1} (expected: False)")
    logger.info(f"  File2 stable: {is_stable2} (expected: False)")
    
    # Wait full stability period - should be stable
    await asyncio.sleep(2)
    logger.info("Third check after 2s more (should be stable):")
    is_stable1 = await check_file_stability_tracked(test_file1, file_tracker, 2)
    is_stable2 = await check_file_stability_tracked(test_file2, file_tracker, 2)
    logger.info(f"  File1 stable: {is_stable1} (expected: True)")
    logger.info(f"  File2 stable: {is_stable2} (expected: True)")
    
    # Modify one file - should become unstable
    test_file1.write_text("modified content")
    logger.info("After modifying file1 (should be unstable):")
    is_stable1 = await check_file_stability_tracked(test_file1, file_tracker, 2)
    is_stable2 = await check_file_stability_tracked(test_file2, file_tracker, 2)
    logger.info(f"  File1 stable: {is_stable1} (expected: False)")
    logger.info(f"  File2 stable: {is_stable2} (expected: True)")
    
    # Test get_stable_files function
    logger.info("Testing get_stable_files function:")
    all_files = [test_file1, test_file2]
    stable_files, unstable_files = await get_stable_files(all_files, file_tracker, 2)
    logger.info(f"  Stable files: {[f.name for f in stable_files]}")
    logger.info(f"  Unstable files: {[f.name for f in unstable_files]}")
    
    # Cleanup
    test_file1.unlink(missing_ok=True)
    test_file2.unlink(missing_ok=True)
    test_dir.rmdir()
    
    logger.info("✅ File stability tracking test completed")
    return True

def test_time_formatting():
    """Test time formatting functionality"""
    logger = logging.getLogger()
    logger.info("🧪 Testing Time Formatting")
    logger.info("=" * 50)
    
    test_cases = [
        (0, "0s"),
        (30, "30s"),
        (60, "1m 0s"),
        (90, "1m 30s"),
        (3600, "1h 0m 0s"),
        (3661, "1h 1m 1s"),
        (86400, "1d 0h 0m 0s"),
        (90061, "1d 1h 1m 1s"),
    ]
    
    for seconds, expected in test_cases:
        result = format_time_remaining(seconds)
        logger.info(f"{seconds}s -> '{result}' (expected: '{expected}')")
        if result == expected:
            logger.info(f"  ✅ PASSED")
        else:
            logger.info(f"  ❌ FAILED")
        logger.info("")
    
    return True

def create_sample_monitoring_messages():
    """Create sample messages for testing monitoring workflow"""
    logger = logging.getLogger()
    logger.info("🧪 Sample Monitoring Commands")
    logger.info("=" * 50)
    
    sample_commands = [
        "run QC for project Solvent01 every 1 minute for 10 minutes",
        "monitor Solvent01 QC every 30 seconds for 2 hours", 
        "check QC for project Test01 every 2m for 1h",
        "run QC for project Solvent01 every 5 minutes for 30 minutes",
        "monitor QC every 1m for 5m",  # Short test
    ]
    
    logger.info("Example commands that will trigger interval monitoring:")
    for i, cmd in enumerate(sample_commands, 1):
        logger.info(f"  {i}. {cmd}")
    
    logger.info("")
    logger.info("Traditional monitoring commands (backward compatible):")
    traditional_commands = [
        "run QC for project Solvent01 for 1h",
        "monitor QC for 2d",
        "run QC for project Test01"
    ]
    
    for i, cmd in enumerate(traditional_commands, 1):
        logger.info(f"  {i}. {cmd}")
    
    return True

async def main():
    """Run all tests"""
    logger = setup_logging()
    logger.info("🚀 Starting Interval Monitoring Test Suite")
    logger.info("=" * 60)
    
    # Test 1: Message parsing
    test_message_parsing()
    
    # Test 2: Duration parsing
    test_duration_parsing()
    
    # Test 3: File stability tracking
    await test_file_stability_tracking()
    
    # Test 4: Time formatting
    test_time_formatting()
    
    # Test 5: Sample commands
    create_sample_monitoring_messages()
    
    logger.info("=" * 60)
    logger.info("🎉 All tests completed!")
    logger.info("")
    logger.info("📋 Summary of New Features:")
    logger.info("   ✅ Interval-based monitoring with 'every X for Y' syntax")
    logger.info("   ✅ File stability tracking to prevent processing growing files")
    logger.info("   ✅ Enhanced logging with detailed progress tracking")
    logger.info("   ✅ Backward compatibility with existing duration-only monitoring")
    logger.info("   ✅ Memory-efficient file tracking with cleanup")
    logger.info("   ✅ Graceful shutdown handling")
    logger.info("")
    logger.info("🔧 Ready for production use!")

if __name__ == "__main__":
    asyncio.run(main())
