# data/csv/ Directory - Complete Guide

**Detailed explanation of the `data/csv/` directory and its role in the QTRAP workflow**

---

## 📁 Location

```
/home/qtrap/sciborg_dev/UI_qtrap/react-agent/src/react_agent/data/csv/
```

---

## 🎯 Purpose

The `csv/` directory stores **parsed and structured MS data** after conversion from raw text files. It serves as an intermediate storage between raw instrument output and quality control analysis.

---

## 🔄 Data Flow Position

```
MS Instrument
      ↓
data/text/{date}/          ← Raw .txt files
      ↓
   Q_parse.py              ← PARSING SCRIPT
      ↓
data/csv/{date}/           ← PARSED CSV FILES (THIS DIRECTORY)
      ↓
   Q_QC.py                 ← Quality Control (may read from here)
      ↓
data/qc/...                ← QC analysis
```

---

## 📂 Directory Structure

```
data/csv/
├── 20250326/              ← Date-organized folders (YYYYMMDD)
│   ├── sample1.csv
│   ├── sample2.csv
│   └── ...
├── 20250327/
│   ├── sample1.csv
│   └── ...
├── 20250421/
│   └── archive/
├── 20250609/
├── 20250612/
│   ├── 1/
│   └── 7pm/
└── 20250616/
    ├── file1.csv
    ├── file2.csv
    └── ... (68 files)
```

**Organization:** By date (YYYYMMDD format)

---

## 🔧 Which Script Uses It?

### **PRIMARY USER: Q_parse.py**

**Purpose:** Converts raw MS text files to structured CSV format

**Code Reference:**
```python
# Q_parse.py lines 40-47
async def setup_directories(date: str):
    # Input: raw text files
    input_dir = f".../data/text/{date}"
    
    # Output: parsed CSV files
    output_base_dir = ".../data/csv"
    output_dir = os.path.join(output_base_dir, date)
    os.makedirs(output_dir, exist_ok=True)
    
    return input_dir, output_dir, log_dir, log_file
```

**What it does:**
1. Reads raw `.txt` files from `data/text/{date}/`
2. Parses MS data (peaks, intensities, retention times)
3. Writes structured CSV files to `data/csv/{date}/`

---

## 📄 File Format

### **CSV Structure:**

Each CSV file contains parsed MS data with columns like:

```csv
Time,Intensity,Q1,Q3,Collision_Energy,Polarity,...
0.1,1234.5,400.2,184.1,35,Positive,...
0.2,2345.6,400.2,184.1,35,Positive,...
...
```

**Example filename:**
```
20250616_20250326_TGL_Splashmix_SolvenTests_Chloroform_r1-1_2_4 Chloroform_MeOH(20mM NH4 Formate)_IPA.csv
```

**Naming convention:**
```
{parse_date}_{original_date}_{operator}_{sample}_{method}.csv
```

### **File Contents:**

Looking at an actual file:
- **Columns:** Time, Intensity values, Q1 mass, Q3 mass, parameters
- **Data:** Time-series intensity measurements
- **Size:** ~69-106 KB per file (typical)
- **Rows:** Hundreds to thousands of data points

---

## 🔍 Current Contents

### **Example: 20250616/ directory**

**Total files:** 68 CSV files  
**Total size:** ~2.2 MB  
**File types:**
- Splashmix solvent tests (6 files)
- Serum lipid extracts (62 files)
  - Various lipid classes: PC, PE, PS, TG, DG, FFA, Ceramides, etc.

**Sample files:**
```
20250616_20250326_TGL_Splashmix_SolvenTests_Chloroform_r1.csv
20250616_20250327_TGL_SerumLipidExtract_PC_wSPLASH.csv
20250616_20250327_TGL_SerumLipidExtract_TG_18-1_wSPLASH.csv
...
```

---

## 🚫 NOT Used By Other Scripts (Currently)

### **Q_QC.py - Does NOT read from csv/**

**Q_QC.py reads from:**
- `data/qc/text/{project}/` - QC-specific text files
- `data/qc/csv/{project}/` - QC-specific CSV files (different location!)

**Note:** The QC workflow has its own separate CSV directory:
- `data/csv/{date}/` ← Parsed data (from Q_parse.py)
- `data/qc/csv/{project}/` ← QC-specific CSV (from Q_QC.py)

These are **different directories** with **different purposes**!

---

## 📊 Comparison: csv/ vs qc/csv/

| Aspect | `data/csv/` | `data/qc/csv/` |
|--------|-------------|----------------|
| **Created by** | Q_parse.py | Q_QC.py |
| **Organization** | By date | By project |
| **Purpose** | General parsing | QC-specific |
| **Input from** | data/text/{date}/ | data/qc/text/{project}/ |
| **Used by** | (Archive/reference) | Q_QC.py, Q_viz_*.py |

---

## 🎯 Use Cases

### **1. Initial Data Parsing**
```
Raw MS data → Q_parse.py → data/csv/{date}/
```
- Convert instrument output to structured format
- Organize by date
- Create searchable, analyzable data

### **2. Archive/Reference**
- Historical parsed data
- Can be referenced if needed
- Not actively used in current QC workflow

### **3. Potential Future Use**
- Could be used for batch reprocessing
- Historical data analysis
- Comparison across dates

---

## 🔄 Typical Workflow

### **Step 1: Data Collection**
```bash
# Raw data arrives from MS instrument
data/text/20250616/sample1.txt
data/text/20250616/sample2.txt
```

### **Step 2: Run Parser**
```python
# Q_parse.py is executed with date parameter
date = "20250616"
Q_parse.py processes all files in data/text/20250616/
```

### **Step 3: CSV Output**
```bash
# Parsed files written to
data/csv/20250616/sample1.csv
data/csv/20250616/sample2.csv
```

### **Step 4: QC Workflow (Separate Path)**
```bash
# QC uses its own directories
data/qc/text/{project}/
data/qc/csv/{project}/
```

---

## 💾 Storage Considerations

### **Typical Directory:**
- **Files:** 50-100 CSV files per date
- **Size:** 1-5 MB per date directory
- **Total:** Multiple date directories

### **Example (20250616):**
- **68 files**
- **2.2 MB total**
- **Average:** ~32 KB per file

---

## 🗂️ Archive Strategy

### **When to Archive:**
- After QC workflow is complete
- After data is no longer actively used
- To keep active directories clean

### **How to Archive:**
```bash
# Move old date directories to archive
mv data/csv/20250326 data/csv/archive/
mv data/csv/20250327 data/csv/archive/
```

---

## 📝 File Naming Details

### **Pattern:**
```
{parse_date}_{original_date}_{operator}_{sample}_{lipid_class}_{details}.csv
```

### **Example Breakdown:**
```
20250616_20250327_TGL_SerumLipidExtract_PC_wSPLASH-Serum,Human,Male,ABPlasma,SigmaAldrich.csv

20250616        = Date parsed
20250327        = Original data date
TGL             = Operator initials
SerumLipidExtract = Sample type
PC_wSPLASH      = Lipid class (Phosphatidylcholine with SPLASH standard)
Serum,Human,... = Additional sample details
```

---

## 🔍 How to Check Contents

### **List all dates:**
```bash
ls -la data/csv/
```

### **Count files in a date:**
```bash
ls data/csv/20250616/ | wc -l
```

### **Check file sizes:**
```bash
du -sh data/csv/20250616/
```

### **View file structure:**
```bash
head -5 data/csv/20250616/sample.csv
```

---

## ⚠️ Important Notes

### **1. Date Organization**
- Files are organized by **parse date**, not original data date
- One date directory may contain files from multiple original dates

### **2. Not Used in Current QC Workflow**
- QC workflow uses `data/qc/csv/{project}/` instead
- This directory is primarily for archival/reference

### **3. Different from qc/csv/**
- `data/csv/` = General parsed data (by date)
- `data/qc/csv/` = QC-specific data (by project)

### **4. Created Automatically**
- Q_parse.py creates date directories automatically
- No manual setup required

---

## 🚀 Quick Reference

| Question | Answer |
|----------|--------|
| **What is it?** | Parsed MS data in CSV format |
| **Created by?** | Q_parse.py |
| **Organized by?** | Date (YYYYMMDD) |
| **Used by?** | Q_parse.py (writes), Archive/reference |
| **File format?** | CSV with time-series MS data |
| **Typical size?** | 1-5 MB per date directory |
| **Related to QC?** | No - QC uses data/qc/csv/ instead |

---

## 📖 Related Documentation

- **[Data Directory Guide](DATA_DIRECTORY_GUIDE.md)** - Complete data structure
- **[Data Flow Diagram](DATA_FLOW_DIAGRAM.md)** - Visual pipeline
- **[Q_parse.py Documentation](guides/)** - Parser details

---

## 💡 Summary

**`data/csv/` is:**
- ✅ Output directory for Q_parse.py
- ✅ Organized by date
- ✅ Contains structured MS data
- ✅ Archive/reference storage
- ❌ NOT used by current QC workflow (uses data/qc/csv/ instead)

**Purpose:** Intermediate storage between raw data and analysis, primarily for archival and reference.
