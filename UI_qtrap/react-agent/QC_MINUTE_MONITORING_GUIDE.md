# QC Monitoring Every 1 Minute for 10 Minutes - Implementation Guide

## Overview

This implementation provides a specialized QC monitoring system that runs **every 1 minute for exactly 10 minutes** with comprehensive logging of all activities, file processing status, and detailed timestamps.

## Key Features

### ✅ Timing
- Runs for exactly 10 minutes (600 seconds)
- Checks every 60 seconds (1 minute intervals)
- Precise timing with loop duration tracking

### ✅ Comprehensive Logging
- Session start/end with 80 "=" characters
- Loop iterations with 60 "-" characters  
- Step-by-step processing logs
- File status summaries
- Cumulative statistics
- Detailed timestamps

### ✅ File Processing
- Only processes stable files (not still growing)
- Avoids reprocessing the same files
- Tracks file stability over 20 seconds
- Project-specific file filtering

### ✅ Error Handling
- Continues running despite individual errors
- Full stack trace logging
- Graceful shutdown on Ctrl+C
- Error counting and reporting

### ✅ Resource Management
- Prevents memory buildup from file tracking
- Cleans up processed file trackers
- Efficient file stability checking

## Usage Examples

### 1. Direct Function Call
```python
import asyncio
from Q_QC import run_qc_monitoring_every_minute_for_10_minutes

# Run monitoring for project Solvent01
result = await run_qc_monitoring_every_minute_for_10_minutes("Solvent01")
```

### 2. Convenience Wrapper
```python
from Q_QC import run_qc_monitoring_session

# Same functionality, cleaner interface
result = await run_qc_monitoring_session("Solvent01")
```

### 3. Message-Based Triggering
The system recognizes these message patterns:
- `"run QC monitoring every 1 minute for 10 minutes for project Solvent01"`
- `"QC monitoring every 1 minute for 10 minutes Solvent01"`
- `"monitor project Solvent01 every 1 minute for 10 minutes"`

### 4. Integration with Existing Pipeline
```python
# The main Q_QC.py file automatically detects the 1m/10m pattern
test_messages = [
    {'type': 'human', 'content': 'run QC monitoring every 1 minute for 10 minutes for project Solvent01'}
]

# Will automatically use the new monitoring function
python Q_QC.py
```

## Expected Log Output Structure

### Session Start
```
================================================================================
QC MONITORING SESSION STARTED
================================================================================
Project: Solvent01
Interval: Every 60 seconds (1 minute)
Total Duration: 600 seconds (10 minutes)
Start Time: 2025-01-23 16:45:01
End Time: 2025-01-23 16:55:01
File Stability Check: 20 seconds
================================================================================
```

### Loop Iterations
```
------------------------------------------------------------
LOOP 1 STARTING
Time: 16:45:01
Remaining: 600 seconds (10:00)
------------------------------------------------------------

Step 1: Scanning for WIFF files...
Found 15 total WIFF files
Found 3 WIFF files for project Solvent01

Step 2: Checking file stability...
File Status Summary:
  - Total files found: 3
  - Stable files: 2
  - Unstable files: 1
  - New stable files: 2
  - Already processed: 0

Step 3: Processing 2 new stable files...
Processing: 20250916_Sample1_Proj-Solvent01.wiff
  ✓ Successfully processed: 20250916_Sample1_Proj-Solvent01.wiff

LOOP 1 COMPLETED
Loop Duration: 45 seconds
Files Processed This Loop: 2
Cumulative Stats:
  - Total Files Processed: 2
  - Total Files Skipped: 1
  - Total Errors: 0
  - Unique Files Tracked: 2
```

### Session End
```
================================================================================
QC MONITORING SESSION COMPLETED
================================================================================
Project: Solvent01
End Time: 2025-01-23 16:55:01
Total Monitoring Time: 600 seconds (10 minutes)
Completed Loops: 10
Final Statistics:
  - Total Files Processed: 8
  - Total Files Skipped (unstable): 3
  - Total Errors Encountered: 0
  - Unique Files Tracked: 8
  - Success Rate: 100.0%
================================================================================
```

## Return Value Structure

The function returns a comprehensive dictionary:

```python
{
    'success': True,
    'project_name': 'Solvent01',
    'total_duration_seconds': 600,
    'interval_seconds': 60,
    'completed_loops': 10,
    'files_processed_total': 8,
    'files_skipped_total': 3,
    'errors_encountered': 0,
    'unique_files_tracked': 8,
    'success_rate': 100.0,
    'monitoring_results': [
        'Loop 1: Processed 2 files',
        'Loop 2: No new files',
        # ... more results
    ],
    'shutdown_requested': False
}
```

## Integration Points

### With LangGraph System ✅ FIXED
- **Special Case Detection**: `continuous_qc_monitoring_node` now detects 1m/10m pattern
- **Automatic Routing**: When interval=60s and duration=600s, automatically calls comprehensive monitoring
- **State Conversion**: Results properly converted to QCState format for LangGraph compatibility
- **Message Integration**: Works seamlessly with existing message parsing and graph structure

### With Existing QC Pipeline
- Uses existing `run_single_qc_check()` for complete 5-step QC workflow
- Integrates with `get_stable_files()` for file stability checking
- Uses existing `setup_logging()` for project-specific logging
- Calls existing `get_directories()` for path management (now includes 'wiff' directory)

### With Message Parsing
- Extends `extract_interval_and_duration_from_messages()` with new patterns
- Works with existing `extract_project_from_messages()` function
- Integrates with existing graph structure and state management

### With Error Handling
- Graceful shutdown on SIGINT/SIGTERM signals
- Continues processing despite individual loop errors
- Comprehensive error logging with stack traces
- Resource cleanup on completion

## Testing

Run the test scripts to verify functionality:

### 1. Basic Functionality Test
```bash
cd /home/qtrap/sciborg_dev/UI_qtrap/react-agent
python test_minute_monitoring.py
```

The test script includes:
- Message parsing tests
- Usage examples
- Optional full monitoring test (runs for 10 minutes)

### 2. LangGraph Integration Test ✅ NEW
```bash
cd /home/qtrap/sciborg_dev/UI_qtrap/react-agent
python test_integration_fix.py
```

The integration test verifies:
- Pattern detection for 1m/10m works correctly
- `continuous_qc_monitoring_node` properly routes to comprehensive monitoring
- QCState conversion works correctly
- Full LangGraph integration (optional 10-minute test)

## Success Criteria

✅ **Timing**: Runs for exactly 10 minutes with 1-minute intervals  
✅ **Logging**: All activities logged with timestamps and details  
✅ **File Processing**: Only processes stable, new files  
✅ **Error Handling**: Continues running despite individual errors  
✅ **Resource Management**: Avoids memory buildup from file tracking  
✅ **User Experience**: Clear progress indication and final summary  
✅ **Integration**: Works with existing QC pipeline components  

## File Locations

- **Main Implementation**: `/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/Q_QC.py`
- **Test Script**: `/home/qtrap/sciborg_dev/UI_qtrap/react-agent/test_minute_monitoring.py`
- **This Guide**: `/home/qtrap/sciborg_dev/UI_qtrap/react-agent/QC_MINUTE_MONITORING_GUIDE.md`

## Log File Location

Project-specific logs are written to:
```
/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/logs/qc/{project_name}/QC_{project_name}.log
```

Example: `/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/logs/qc/Solvent01/QC_Solvent01.log`
