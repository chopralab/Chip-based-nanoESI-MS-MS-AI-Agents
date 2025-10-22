# QTRAP Workflow - Complete Summary

---

## 🎯 One-Sentence Summary

QTRAP automates mass spectrometry quality control by parsing raw instrument data, analyzing quality metrics, generating TIC plots, creating visualizations, and automatically generating reprocessing worklists for failed samples.

---

## 📋 Concise Outline (One Sentence Per Step)

1. **Raw Data** - MS instrument generates txt files in data/text/project
2. **Parsing** - Q_parse converts raw text to CSV in data/csv/date (optional archive step)
3. **QC Input** - Files organized in data/qc/text/project for quality control
4. **QC Analysis** - Q_QC analyzes data and writes results to data/qc/results/project
5. **TIC Plots** - Q_QC_TIC creates chromatograms in data/qc/TIC/project
6. **Pass/Fail** - Passing samples go to production, failing samples to data/worklist/qc_fail/project
7. **Visualization** - Q_viz creates plots in data/viz/intensity/project
8. **Worklists** - Q_worklist generates reprocessing lists in data/worklist/generated/project
9. **Logging** - All operations logged in data/logs for tracking

---

## 📖 Detailed Workflow

### Step 1: Raw Data Collection
- MS instrument runs samples
- Generates txt files with time-series data
- Stored in data/text/project
- Example: 67 files for solventmatrix project

### Step 2: Data Parsing (Optional)
- Q_parse reads from data/text/date
- Converts to structured CSV
- Writes to data/csv/date
- Purpose: Archive and reference

### Step 3: QC Input Preparation
- Files placed in data/qc/text/project
- Organized by project name
- Ready for quality control analysis

### Step 4: Quality Control Analysis
- Q_QC reads from data/qc/text/project
- Calculates QC metrics (CV, intensity, etc)
- Writes results to data/qc/results/project
- Creates QC_project_RESULTS.csv

### Step 5: TIC Generation
- Q_QC_TIC extracts Total Ion Current
- Creates chromatogram plots (PNG/PDF)
- Saves to data/qc/TIC/project/chromatograms
- Failed samples in hidden_chromatograms

### Step 6: Pass/Fail Decision
- PASS: Files move to data/text/project (production ready)
- FAIL: Files move to data/worklist/qc_fail/project
- Automatic decision based on QC thresholds

### Step 7: Visualization
- Q_viz_intensity reads QC results
- Creates intensity plots, heatmaps, statistical analysis
- Saves to data/viz/intensity/project
- Includes faceted panels and normalized data

### Step 8: Worklist Generation
- Q_worklist reads from worklist/input
- Combines with failed samples from qc_fail
- Generates optimized sample lists
- Writes to worklist/generated/project

### Step 9: Logging
- All scripts log to data/logs/script
- Daily log files for tracking
- Includes parse, qc, worklist logs

---

## 🗂️ Directory Map

```
data/
├── text/project          [Step 1: Raw input]
├── csv/date             [Step 2: Parsed archive]
├── qc/
│   ├── text/project     [Step 3: QC input]
│   ├── csv/project      [Step 4: QC parsed]
│   ├── results/project  [Step 4: QC results]
│   └── TIC/project      [Step 5: Chromatograms]
├── viz/
│   └── intensity/project [Step 7: Visualizations]
├── worklist/
│   ├── input            [Step 8: User input]
│   ├── generated/project [Step 8: Output lists]
│   └── qc_fail/project  [Step 6: Failed samples]
└── logs/                [Step 9: All logging]
```

---

## 🔄 Example: Solventmatrix Project

### Input
- 67 raw txt files in data/text/solventmatrix

### Processing
- QC analysis creates 80 text + 80 CSV files
- 5 result files with metrics
- 2 TIC data files

### Output
- 27 visualization files
- 29 worklist files
- Total: 293 files across pipeline

---

## 🎯 Key Scripts

- Q_parse: text to CSV conversion
- Q_QC: Main quality control orchestrator
- Q_QC_TIC: TIC extraction and plotting
- Q_viz_intensity: Intensity visualizations
- Q_worklist: Worklist generation
- Q_helper: Literature Q&A (separate)

---

## 💡 Summary

The QTRAP workflow is a fully automated pipeline that takes raw MS data through quality control, visualization, and worklist generation with minimal manual intervention.
