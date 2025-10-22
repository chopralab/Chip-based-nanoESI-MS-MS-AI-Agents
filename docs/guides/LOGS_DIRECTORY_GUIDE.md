# Logs Directory - Complete Guide

**How each QTRAP script uses logging and what's in each log file**

---

## 📁 Directory Structure

```
data/logs/
├── convert/                    ← Q_convert.py logs
│   ├── convert_20250609.log
│   ├── convert_20251022.log
│   └── archive/
│
├── parse/                      ← Q_parse.py logs
│   ├── 20250326/              ← Organized by date
│   │   └── parse_20250326.log
│   ├── 20250616/
│   │   └── parse_20250616.log
│   └── ...
│
├── qc/                         ← Q_QC.py logs
│   ├── solventmatrix/         ← Organized by project
│   │   └── QC_solventmatrix.log
│   └── archive/
│
└── worklist/                   ← Q_worklist.py logs
    ├── worklist_20250609.log
    ├── worklist_20251022.log
    └── archive/
```

---

## 📝 Script-by-Script Logging

### **1. Q_worklist.py - Worklist Generation Logs**

**Log Location:**
```
data/logs/worklist/worklist_{YYYYMMDD}.log
```

**Organization:** By date (daily rotation)

**Code (Lines 39-44):**
```python
LOG_DIR = os.path.join(SCRIPT_DIR, "data", "logs", "worklist")
os.makedirs(LOG_DIR, exist_ok=True)
today_str = datetime.now().strftime("%Y%m%d")
LOG_FILE = os.path.join(LOG_DIR, f"worklist_{today_str}.log")
```

**What's Logged:**
```
2025-10-22 15:17:46,325 [INFO] Logging to /home/qtrap/.../worklist_20251022.log
2025-10-22 15:18:00,065 [INFO] Generated aggregated worklist: .../worklist_20251022_318pm_solventmatrix.csv
2025-10-22 15:18:01,121 [INFO] Copied to server output: /mnt/d_drive/.../QTRAP_worklist/solventmatrix
```

**Contents:**
- Worklist generation start/completion
- Input file processing
- Output file paths
- Server copy status
- Failed sample integration
- Errors and warnings

---

### **2. Q_parse.py - Parsing Logs**

**Log Location:**
```
data/logs/parse/{date}/parse_{YYYYMMDD}.log
```

**Organization:** By date (subdirectory per date)

**Code (Lines 50-53):**
```python
log_base_dir = ".../data/logs/parse"
log_dir = os.path.join(log_base_dir, date)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"parse_{date}.log")
```

**What's Logged:**
```
2025-06-16 13:45:39,139 INFO Starting batch parsing for date: 20250616
2025-06-16 13:45:39,140 INFO Found 34 .txt files to process
2025-06-16 13:45:39,156 INFO ✅ Parsed sample1.txt → sample1.csv (186 rows)
2025-06-16 13:45:39,249 ERROR ❌ Failed to parse sample2.txt
```

**Contents:**
- Batch parsing start
- Input/output directories
- File count
- Each file parsed (success/failure)
- Row counts
- Errors with stack traces
- Summary statistics

---

### **3. Q_QC.py - Quality Control Logs**

**Log Location:**
```
data/logs/qc/{project}/QC_{project}.log
```

**Organization:** By project name

**Code (Lines 536-539):**
```python
dirs = get_directories(project_name)
log_dir = dirs['log_dir']  # data/logs/qc/{project}/
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"QC_{project_name}.log"
```

**What's Logged:**
```
[INFO] Logging initialized for project solventmatrix
[INFO] 🔍 Scanning directory for project 'solventmatrix'
[INFO] ✓ Found complete pair: sample1.wiff (modified: 2025-09-24 21:52)
[INFO] 📊 Total files found for processing: 67
[INFO] 🚀 Executing command: cmd.exe /c cd /d C:\... && convert_files.bat
[INFO] ✅ Conversion successful. Files moved: ['file1.txt', 'file2.txt']
[INFO] ✅ Parsed 80 files successfully
[INFO] QC Results saved to: .../QC_solventmatrix_RESULTS.csv
[INFO] ✅ Moved 50 passing files to production
[INFO] ⚠️ Moved 17 failing files to qc_fail directory
```

**Contents:**
- Project initialization
- WIFF file scanning and copying
- MSConvert execution
- File conversion status
- Parsing results
- QC analysis metrics
- Pass/fail decisions
- File movements
- TIC generation
- Errors and warnings
- Complete workflow tracking

---

### **4. Q_convert.py - Conversion Logs (Standalone)**

**Log Location:**
```
data/logs/convert/convert_{YYYYMMDD}.log
```

**Organization:** By date (daily rotation)

**Code (Lines 42-45):**
```python
script_dir = Path(__file__).parent
logs_dir = script_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "convert.log"
```

**What's Logged:**
```
2025-10-22 12:44:12,002 [INFO] Logging initialized. Writing to .../convert_20251022.log
[INFO] Path check: /mnt/c/.../convert_raw exists? True
[INFO] Checking for batch file at: /mnt/c/.../convert_files.bat
[INFO] Executing conversion command: cmd.exe /c cd /d C:\... && convert_files.bat
[INFO] Conversion step completed successfully
[INFO] Found 5 .txt files in source: /mnt/c/.../converted_files
[INFO] Copied: sample1.txt -> /home/qtrap/.../convert/converted_files/sample1.txt
```

**Contents:**
- Initialization
- Path checks
- Batch file verification
- MSConvert execution
- File discovery
- File copying status
- Errors

**Note:** This is from standalone Q_convert.py (not actively used)

---

## 📊 Log File Comparison

| Script | Location | Organization | Rotation | Status |
|--------|----------|--------------|----------|--------|
| Q_worklist.py | `logs/worklist/` | By date | Daily | ✅ Active |
| Q_parse.py | `logs/parse/{date}/` | By date | Per run | ✅ Active |
| Q_QC.py | `logs/qc/{project}/` | By project | Per run | ✅ Active |
| Q_convert.py | `logs/convert/` | By date | Daily | ❌ Standalone |

---

## 🔍 What Each Log Contains

### **Common Elements (All Logs):**
- Timestamp
- Log level (INFO, ERROR, WARNING, DEBUG)
- Message content
- File paths
- Operation status

### **Q_worklist.py Specific:**
- Input worklist processing
- Method lookup results
- Lipid class aggregation
- Failed sample integration
- Output file generation
- Server copy operations

### **Q_parse.py Specific:**
- File discovery count
- Per-file parsing status
- Row counts per file
- Success/failure ratio
- Error messages with details

### **Q_QC.py Specific:**
- WIFF file scanning
- File pair matching
- MSConvert execution
- Conversion status
- Parsing results
- QC metric calculations
- Pass/fail decisions
- File movements (production vs qc_fail)
- TIC generation status
- Complete workflow tracking

### **Q_convert.py Specific:**
- Path accessibility checks
- Batch file verification
- MSConvert command execution
- File discovery and copying

---

## 📈 Log File Sizes

**Typical sizes:**
- **worklist logs:** 144 bytes - 4 KB (small, simple operations)
- **parse logs:** 1-5 KB (depends on file count)
- **QC logs:** 10-100 KB+ (comprehensive workflow tracking)
- **convert logs:** 1-5 KB (simple conversion tracking)

---

## 🗂️ Archive Strategy

Each log directory has an `archive/` subdirectory:

```
logs/
├── convert/archive/
├── qc/archive/
└── worklist/archive/
```

**When to archive:**
- Old date-based logs (> 30 days)
- Completed project logs
- To keep active directories clean

---

## 💡 How to Use Logs

### **Debugging Failed Runs:**
```bash
# Check latest QC log
tail -100 data/logs/qc/solventmatrix/QC_solventmatrix.log

# Check for errors
grep ERROR data/logs/qc/solventmatrix/QC_solventmatrix.log

# Check worklist generation
cat data/logs/worklist/worklist_20251022.log
```

### **Tracking Workflow:**
```bash
# See what files were processed
grep "✅ Parsed" data/logs/parse/20250616/parse_20250616.log

# See QC pass/fail
grep "Moved.*passing\|failing" data/logs/qc/solventmatrix/QC_solventmatrix.log

# See worklist output
grep "Generated" data/logs/worklist/worklist_20251022.log
```

### **Finding Issues:**
```bash
# All errors across all logs
find data/logs -name "*.log" -exec grep -l ERROR {} \;

# Recent activity
find data/logs -name "*.log" -mtime -1  # Last 24 hours
```

---

## 🎯 Best Practices

### **1. Log Rotation**
- Worklist: Daily (automatic via date in filename)
- Parse: Per date (automatic via subdirectory)
- QC: Per project (overwrites on each run)
- Convert: Daily (automatic via date in filename)

### **2. Log Retention**
- Keep current logs in main directory
- Move old logs to archive/ subdirectories
- Consider deleting logs > 90 days old

### **3. Debugging**
- Always check logs first when troubleshooting
- Look for ERROR and WARNING messages
- Check timestamps to understand workflow timing

### **4. Monitoring**
- QC logs show complete workflow status
- Worklist logs confirm file generation
- Parse logs show data processing success rate

---

## 📝 Quick Reference

| Need to... | Check this log |
|------------|----------------|
| Debug QC workflow | `logs/qc/{project}/QC_{project}.log` |
| See parsing errors | `logs/parse/{date}/parse_{date}.log` |
| Verify worklist generation | `logs/worklist/worklist_{date}.log` |
| Check conversion issues | `logs/convert/convert_{date}.log` |
| Find all errors | `grep -r ERROR logs/` |
| See recent activity | `find logs/ -mtime -1` |

---

## 🚫 What's NOT Logged

- Visualization generation (Q_viz_*.py) - No dedicated logs
- TIC generation (Q_QC_TIC.py) - Logged via Q_QC.py
- Helper queries (Q_helper.py) - No logging

---

## 💾 Git Status

**Logs are ignored by git:**
```gitignore
# From .gitignore
*.log
```

**Why?**
- Logs are temporary/runtime data
- Can be regenerated
- Would clutter repository
- Contain system-specific paths

---

**Logs provide complete traceability of the QTRAP workflow from raw data to final outputs! 📊**
