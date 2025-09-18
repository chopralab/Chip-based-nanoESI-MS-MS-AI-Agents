# QC Interval-Based Monitoring Guide

## Overview

The QC pipeline now supports advanced **interval-based monitoring** that allows you to:
- Run QC checks at regular intervals (every X seconds/minutes)
- Monitor for a specific total duration 
- Track file stability to avoid processing growing files
- Maintain a history of processed files to avoid reprocessing

## New Features Added

### 1. Enhanced Message Parsing
- **Interval + Duration**: `"run QC for project Solvent01 every 1 minute for 10 minutes"`
- **Flexible Syntax**: Supports natural language and abbreviated formats
- **Backward Compatible**: Traditional duration-only monitoring still works

### 2. File Stability Tracking
- **Smart Detection**: Only processes files that haven't grown for 20+ seconds
- **Memory Efficient**: Tracks file sizes and timestamps
- **Prevents Errors**: Avoids processing incomplete/growing files

### 3. Processed File Tracking
- **No Duplicates**: Remembers which files have been processed
- **Efficient**: Uses sets for fast lookup
- **Clean Memory**: Automatically cleans up tracking data

## Usage Examples

### Interval-Based Monitoring

```python
# Every 1 minute for 10 minutes
"run QC for project Solvent01 every 1 minute for 10 minutes"

# Every 30 seconds for 2 hours  
"monitor Solvent01 QC every 30 seconds for 2 hours"

# Short format
"check QC for project Test01 every 2m for 1h"

# Very frequent checks for testing
"run QC every 30s for 5m"
```

### Traditional Monitoring (Backward Compatible)

```python
# Duration only (uses 5-minute intervals)
"run QC for project Solvent01 for 1h"
"monitor QC for 2d"
"run QC for project Test01"
```

## Supported Time Formats

### Intervals
- **Seconds**: `30s`, `45s`
- **Minutes**: `1m`, `2m`, `5m`, `10m`

### Durations  
- **Minutes**: `5m`, `10m`, `30m`
- **Hours**: `1h`, `2h`, `24h`
- **Days**: `1d`, `2d`, `7d`

### Natural Language
- `"every 1 minute"` → `1m`
- `"every 30 seconds"` → `30s`
- `"for 2 hours"` → `2h`
- `"for 10 minutes"` → `10m`

## Expected Behavior

### Example: "every 1 minute for 10 minutes"

```
2025-09-18 10:00:00 [INFO] 🚀 Starting QC monitoring: Solvent01, every 1m for 10m
2025-09-18 10:00:00 [INFO] 📅 Monitoring will run until: 2025-09-18 10:10:00
2025-09-18 10:00:00 [INFO] ⏱️ Check interval: 60s, File stability: 20s

2025-09-18 10:00:00 [INFO] 🔄 Loop 1, 10m 0s remaining
2025-09-18 10:00:00 [INFO] 🔍 Found 3 total files for project Solvent01
2025-09-18 10:00:00 [INFO] 📊 Files status: 2 stable, 1 still growing
2025-09-18 10:00:00 [INFO] 📊 Processing status: 2 new, 0 already done
2025-09-18 10:00:00 [INFO] ✅ Loop 1: Processed 2 new files

2025-09-18 10:01:00 [INFO] 🔄 Loop 2, 9m 0s remaining
2025-09-18 10:01:00 [INFO] 🔍 Found 4 total files for project Solvent01
2025-09-18 10:01:00 [INFO] 📊 Files status: 3 stable, 1 still growing
2025-09-18 10:01:00 [INFO] 📊 Processing status: 1 new, 2 already done
2025-09-18 10:01:00 [INFO] ✅ Loop 2: Processed 1 new file

...

2025-09-18 10:10:00 [INFO] ✅ QC monitoring completed after 10m
2025-09-18 10:10:00 [INFO] 📊 Final Summary:
2025-09-18 10:10:00 [INFO]    • Total loops: 10
2025-09-18 10:10:00 [INFO]    • Files processed: 15
2025-09-18 10:10:00 [INFO]    • Files skipped (unstable): 3
2025-09-18 10:10:00 [INFO]    • Errors encountered: 0
2025-09-18 10:10:00 [INFO]    • Unique files tracked: 18
```

## File Stability Implementation

### How It Works
1. **First Detection**: File is added to tracker with current size and timestamp
2. **Size Monitoring**: Each check compares current size to last recorded size
3. **Growth Detection**: If size changed, update tracker and mark as unstable
4. **Stability Check**: If size unchanged for ≥20 seconds, mark as stable
5. **Processing**: Only stable files are processed through QC pipeline

### Benefits
- **Prevents Corruption**: Avoids processing incomplete files
- **Reduces Errors**: Eliminates "file in use" errors
- **Improves Reliability**: Ensures complete data before analysis

## QC Pipeline Integration

### Complete 5-Step Workflow
Each monitoring iteration runs the full QC pipeline:

1. **Check for WIFF files** (`move_wiff_pairs`)
2. **Convert WIFF to TXT** (`Convert` class)
3. **Parse TXT to CSV** (`QTRAP_Parse` class)  
4. **Generate QC results** (`qc_results` function)
5. **Move files by pass/fail** (`qc_validated_move` function)

### Enhanced Logging
```
=== QC CHECK #1 STARTING ===
Step 1: Checking for new WIFF files...
✅ Step 1: Found 2 new WIFF file pairs
Step 2: Converting WIFF to TXT...
✅ Step 2: Conversion successful. Files moved: 2
Step 3: Parsing TXT to CSV...
✅ Step 3: Parsed 2 files to CSV
Step 4: Generating QC results...
✅ Step 4: QC results generated for Solvent01
Step 5: Moving validated files...
✅ Step 5: QC validated files moved to production and fail directories
=== QC CHECK #1 COMPLETED: 2 files processed ===
```

## Performance Considerations

### Memory Management
- **File Tracker Cleanup**: Removes processed files from tracker
- **Set-Based Tracking**: Efficient lookup for processed files
- **Bounded Growth**: Memory usage doesn't grow indefinitely

### Optimal Intervals
- **Short Projects**: 30s-1m intervals for 5-30m duration
- **Long Projects**: 5-15m intervals for hours/days
- **File Stability**: 20s default works for most file sizes

### Resource Usage
- **CPU**: Minimal overhead between checks
- **Disk I/O**: Only when files are detected
- **Network**: No additional network calls

## Error Handling

### Graceful Shutdown
- **SIGINT/SIGTERM**: Clean shutdown on Ctrl+C or kill signals
- **State Preservation**: Current progress is logged
- **Resource Cleanup**: File handles and trackers are cleaned up

### Error Recovery
- **Individual Loop Errors**: Don't stop entire monitoring
- **File Processing Errors**: Logged but don't affect other files
- **Network/Disk Errors**: Retry on next iteration

## Testing

### Run Test Suite
```bash
cd /home/qtrap/sciborg_dev/UI_qtrap/react-agent
conda activate qtrap_graph
python test_interval_monitoring.py
```

### Test Cases Covered
- ✅ Message parsing for all syntax variations
- ✅ Duration parsing and conversion
- ✅ File stability tracking logic
- ✅ Time formatting for user display
- ✅ Sample command validation

## Migration Guide

### Existing Code
No changes needed! Existing monitoring commands continue to work:
```python
# These still work exactly as before
"run QC for project Solvent01 for 1h"
"monitor QC for 2d"
```

### New Capabilities
Simply add interval syntax to enable new features:
```python
# Old way (5-minute intervals)
"run QC for project Solvent01 for 1h"

# New way (1-minute intervals)  
"run QC for project Solvent01 every 1 minute for 1 hour"
```

## Troubleshooting

### Common Issues

**Files Not Processing**
- Check file stability logs - files may still be growing
- Verify project name matches filename pattern `Proj-ProjectName`
- Ensure sufficient time between interval checks

**Memory Usage**
- Monitor `processed_files` set size in logs
- File tracker automatically cleans up processed files
- Restart monitoring for very long runs (days)

**Performance**
- Increase interval time for large numbers of files
- Use longer stability check periods for large files
- Monitor system resources during peak processing

### Debug Logging
Enable debug logging to see detailed file tracking:
```python
# In your monitoring setup
logging.getLogger().setLevel(logging.DEBUG)
```

## Summary

The enhanced QC monitoring system provides:

✅ **Flexible Scheduling** - Run checks at any interval
✅ **Smart File Handling** - Only process stable, complete files  
✅ **Efficient Tracking** - Avoid reprocessing the same data
✅ **Robust Error Handling** - Continue monitoring despite individual failures
✅ **Backward Compatibility** - Existing code works unchanged
✅ **Production Ready** - Memory efficient and thoroughly tested

Perfect for both development testing (frequent short checks) and production monitoring (longer intervals over days/weeks).
