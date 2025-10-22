# QTRAP Data Directory Structure Guide

**Complete guide to how the `data/` directory works across all QTRAP scripts**

---

## 📁 Overview

The `data/` directory is the central hub for all QTRAP workflow data. Each script reads from and writes to specific subdirectories in an organized pipeline.

**Location:** `UI_qtrap/react-agent/src/react_agent/data/`

---

## 🔄 Complete Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    QTRAP DATA PIPELINE                          │
└─────────────────────────────────────────────────────────────────┘

1. RAW DATA INPUT
   └─> text/{project}/              [Raw MS .txt files from instrument]
       
2. PARSING (Q_parse.py)
   └─> csv/{date}/                  [Parsed CSV data]
       └─> logs/parse/{date}/       [Parse logs]

3. QUALITY CONTROL (Q_QC.py)
   ├─> qc/text/{project}/           [QC input text files]
   ├─> qc/csv/{project}/            [QC parsed CSV]
   ├─> qc/results/{project}/        [QC analysis results]
   ├─> qc/TIC/{project}/            [TIC plots & data]
   ├─> worklist/qc_fail/{project}/  [Failed samples]
   └─> logs/qc/{project}/           [QC logs]

4. VISUALIZATION (Q_viz_*.py)
   ├─> viz/intensity/{project}/     [Intensity plots & data]
   └─> viz/QC/{project}/            [QC visualizations]

5. WORKLIST GENERATION (Q_worklist.py)
   ├─> worklist/input/              [User input worklists]
   ├─> worklist/generated/{project}/ [Generated worklists]
   └─> logs/worklist/               [Worklist logs]
```

---

## 📂 Directory Structure by Function

### **1. `text/` - Raw MS Data Input**

```
text/
├── {project_name}/              ← Raw .txt files from MS instrument
│   ├── sample1.txt
│   ├── sample2.txt
│   └── ...
└── archive/                     ← Old projects
    └── {old_date}/
```

**Used by:**
- **Q_parse.py** - Reads raw text files for parsing
- **Q_QC.py** - Moves production-ready files here

**File format:** `.txt`, `.dam.txt` (MS instrument output)

---

### **2. `csv/` - Parsed Data**

```
csv/
└── {date}/                      ← Organized by date (YYYYMMDD)
    ├── sample1.csv
    ├── sample2.csv
    └── ...
```

**Used by:**
- **Q_parse.py** - Writes parsed CSV files here
- **Q_QC.py** - May read for analysis

**File format:** `.csv` (structured data tables)

---

### **3. `qc/` - Quality Control Data**

```
qc/
├── text/{project}/              ← QC input text files
│   ├── sample1.txt
│   └── ...
│
├── csv/{project}/               ← QC parsed CSV data
│   ├── sample1.csv
│   └── ...
│
├── results/{project}/           ← QC analysis results
│   ├── QC_{project}_RESULTS.csv
│   └── organized/
│       └── organized_QC_{project}_RESULTS.csv
│
├── TIC/{project}/               ← TIC plots and data
│   ├── chromatograms/
│   │   ├── png/
│   │   └── pdf/
│   └── hidden_chromatograms/
│
└── worklist/{project}/          ← QC-generated worklists
    └── worklist_{project}.csv
```

**Used by:**
- **Q_QC.py** - Main QC workflow (reads/writes all)
- **Q_QC_TIC.py** - Generates TIC plots
- **Q_viz_QC.py** - Creates QC visualizations

**File formats:** `.txt`, `.csv`, `.png`, `.pdf`

---

### **4. `viz/` - Visualization Outputs**

```
viz/
├── intensity/{project}/         ← Intensity visualizations
│   ├── average_tic_*.csv
│   ├── average_tic_*.png
│   ├── normalized_data.csv
│   ├── faceted_panel_plots/
│   └── statistical_plots/
│
└── QC/{project}/                ← QC visualizations
    ├── qc_summary_plot.png
    └── ...
```

**Used by:**
- **Q_viz_intensity.py** - Main intensity visualizations
- **Q_viz_intensity_advanced.py** - Advanced plots
- **Q_viz_intensity_advanced_part2.py** - Statistical plots
- **Q_viz_QC.py** - QC-specific visualizations

**File formats:** `.png`, `.pdf`, `.csv` (plot data)

---

### **5. `worklist/` - Worklist Management**

```
worklist/
├── input/                       ← User-editable input
│   ├── input_worklist.csv      ← Main input file
│   ├── all_methods.csv         ← All available methods
│   └── archive/
│
├── generated/{project}/         ← Auto-generated worklists
│   ├── worklist_{date}.csv
│   └── ...
│
├── qc_fail/{project}/           ← Failed QC samples
│   ├── failed_sample1.txt
│   └── ...
│
└── methods.csv                  ← Lipid→Method lookup table
```

**Used by:**
- **Q_worklist.py** - Main worklist generation
- **Q_QC.py** - Writes failed samples to qc_fail/

**File formats:** `.csv`, `.txt`

---

### **6. `logs/` - Logging System**

```
logs/
├── parse/{date}/                ← Parse operation logs
│   └── parse_{date}.log
│
├── qc/{project}/                ← QC operation logs
│   └── QC_{project}.log
│
├── worklist/                    ← Worklist operation logs
│   └── worklist_{date}.log
│
└── convert/                     ← Conversion logs
    └── convert.log
```

**Used by:**
- All scripts for debugging and tracking
- Daily log rotation

**File format:** `.log`

---

## 🔄 Script-by-Script Data Usage

### **Q_parse.py - Data Parsing**

**Reads from:**
- `text/{date}/` - Raw MS text files

**Writes to:**
- `csv/{date}/` - Parsed CSV files
- `logs/parse/{date}/` - Parse logs

**Purpose:** Convert raw MS instrument output to structured CSV

---

### **Q_QC.py - Quality Control**

**Reads from:**
- `qc/text/{project}/` - QC input files
- `qc/csv/{project}/` - Parsed QC data

**Writes to:**
- `qc/results/{project}/` - QC analysis results
- `qc/TIC/{project}/` - TIC data (via Q_QC_TIC.py)
- `worklist/qc_fail/{project}/` - Failed samples
- `text/{project}/` - Production-ready files
- `logs/qc/{project}/` - QC logs

**Purpose:** Automated quality control with continuous monitoring

---

### **Q_QC_TIC.py - TIC Analysis**

**Reads from:**
- `qc/text/{project}/` - QC text files
- `qc/results/{project}/` - QC results for filtering

**Writes to:**
- `qc/TIC/{project}/chromatograms/` - TIC plots (PNG/PDF)
- `qc/TIC/{project}/hidden_chromatograms/` - Failed sample plots

**Purpose:** Extract and plot Total Ion Current data

---

### **Q_viz_intensity.py - Intensity Visualization**

**Reads from:**
- `qc/results/{project}/organized/` - Organized QC results

**Writes to:**
- `viz/intensity/{project}/` - Intensity plots and data
  - Average TIC plots
  - Normalized data
  - Faceted panels
  - Statistical plots

**Purpose:** Create comprehensive intensity visualizations

---

### **Q_viz_QC.py - QC Visualization**

**Reads from:**
- `qc/results/{project}/` - QC results

**Writes to:**
- `viz/QC/{project}/` - QC-specific visualizations

**Purpose:** Create QC summary visualizations

---

### **Q_worklist.py - Worklist Generation**

**Reads from:**
- `worklist/input/input_worklist.csv` - User input
- `worklist/methods.csv` - Method lookup
- `worklist/qc_fail/{project}/` - Failed samples (optional)

**Writes to:**
- `worklist/generated/{project}/` - Generated worklists
- `logs/worklist/` - Worklist logs

**Purpose:** Generate optimized sample worklists

---

### **Q_helper.py - RAG Helper**

**Reads from:**
- `notebooks/papers/qtrap_nano/` - PDF papers (outside data/)
- FAISS index (outside data/)

**Writes to:**
- None (read-only)

**Purpose:** Literature Q&A using RAG

---

## 📊 Project Organization

### **Project-Based Structure**

Most directories use `{project}` subdirectories:

```
data/
├── text/{project}/
├── qc/
│   ├── text/{project}/
│   ├── csv/{project}/
│   ├── results/{project}/
│   └── TIC/{project}/
├── viz/
│   ├── intensity/{project}/
│   └── QC/{project}/
└── worklist/
    ├── generated/{project}/
    └── qc_fail/{project}/
```

**Example projects:**
- `solventmatrix` - Solvent matrix optimization study
- `Solvent01` - Solvent test 01
- `20250916` - Date-based project

---

## 🔄 Typical Workflow Example

### **Complete Pipeline for "solventmatrix" Project:**

```
1. Raw Data Collection
   └─> data/text/solventmatrix/*.txt
       [67 raw MS files]

2. Quality Control (Q_QC.py)
   ├─> data/qc/text/solventmatrix/*.txt
   ├─> data/qc/csv/solventmatrix/*.csv
   ├─> data/qc/results/solventmatrix/QC_solventmatrix_RESULTS.csv
   ├─> data/qc/TIC/solventmatrix/chromatograms/*.png
   └─> data/logs/qc/solventmatrix/QC_solventmatrix.log

3. Visualization (Q_viz_intensity.py)
   └─> data/viz/intensity/solventmatrix/
       ├── average_tic_*.png
       ├── normalized_data.csv
       └── faceted_panels/

4. Failed Sample Handling
   ├─> data/worklist/qc_fail/solventmatrix/*.txt
   └─> Q_worklist.py generates reprocessing worklist
       └─> data/worklist/generated/solventmatrix/worklist_*.csv
```

---

## 📝 File Naming Conventions

### **Text Files:**
```
{date}_{operator}_{sample}_{lipid}_{replicate}_{project}.txt
Example: 20250916_21MeOHACN_BrainLipidEx_LC-PC_R-1_Op-TGL_Proj-solventmatrix_PC_withSPLASH.dam.txt
```

### **CSV Files:**
```
{date}_{operator}_{sample}_{lipid}_{replicate}_{project}.csv
```

### **Results Files:**
```
QC_{project}_RESULTS.csv
organized_QC_{project}_RESULTS.csv
```

### **Log Files:**
```
{script}_{date}.log
Example: worklist_20251022.log, QC_solventmatrix.log
```

---

## 🗂️ Archive Strategy

Each major directory has an `archive/` subdirectory:

```
data/
├── text/archive/           ← Old projects
├── qc/
│   ├── text/archive/
│   ├── csv/archive/
│   ├── results/archive/
│   └── TIC/archive/
├── worklist/
│   ├── input/archive/
│   └── generated/archive/
└── logs/
    ├── qc/archive/
    └── worklist/archive/
```

**Purpose:** Keep old data organized without cluttering active directories

---

## 💾 Storage Considerations

### **File Sizes:**
- **Text files:** 10-100 KB each
- **CSV files:** 5-50 KB each
- **PNG plots:** 50-500 KB each
- **PDF plots:** 100 KB - 2 MB each
- **Log files:** 1-10 KB each

### **Typical Project:**
- **solventmatrix:** ~293 files, ~10-50 MB total
- Includes all stages: raw → QC → viz → worklist

---

## 🎯 Best Practices

### **1. Project Naming**
- Use descriptive names: `solventmatrix`, not `test1`
- Use consistent naming across all directories
- Avoid spaces and special characters

### **2. Archive Old Data**
- Move completed projects to `archive/` subdirectories
- Keep active directories clean

### **3. Log Rotation**
- Logs are automatically dated
- Archive old logs periodically

### **4. Backup Strategy**
- Raw data (`text/`) is most critical
- QC results can be regenerated
- Visualizations can be recreated

---

## 🔍 Quick Reference

| Data Type | Location | Script | Purpose |
|-----------|----------|--------|---------|
| Raw MS data | `text/{project}/` | Q_parse | Input |
| Parsed data | `csv/{date}/` | Q_parse | Structured data |
| QC input | `qc/text/{project}/` | Q_QC | QC analysis |
| QC results | `qc/results/{project}/` | Q_QC | QC output |
| TIC plots | `qc/TIC/{project}/` | Q_QC_TIC | Chromatograms |
| Intensity viz | `viz/intensity/{project}/` | Q_viz_intensity | Plots |
| Worklists | `worklist/generated/{project}/` | Q_worklist | Sample lists |
| Failed samples | `worklist/qc_fail/{project}/` | Q_QC | Reprocessing |
| Logs | `logs/{script}/` | All | Debugging |

---

## 📖 Related Documentation

- **[Git Data Files Summary](GIT_DATA_FILES_SUMMARY.md)** - What's tracked in git
- **[Gitignore Guide](GITIGNORE_GUIDE.md)** - File ignore strategy
- **[Setup Guide](SETUP_GUIDE.md)** - Environment setup

---

**The data directory is the heart of the QTRAP workflow - organized, automated, and scalable! 🎯**
