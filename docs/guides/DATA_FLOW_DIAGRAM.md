# QTRAP Data Flow Diagram

**Visual representation of how data flows through the QTRAP system**

---

## 🔄 Complete Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         QTRAP WORKFLOW PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘

                         MS INSTRUMENT
                              │
                              │ .txt files
                              ▼
                    ┌──────────────────┐
                    │  data/text/      │
                    │  {project}/      │  ◄─── Raw MS Data
                    └────────┬─────────┘
                             │
                             │ Q_parse.py
                             ▼
                    ┌──────────────────┐
                    │  data/csv/       │
                    │  {date}/         │  ◄─── Parsed CSV
                    └────────┬─────────┘
                             │
                             │ Q_QC.py
                             ▼
        ┌────────────────────────────────────────────┐
        │         QUALITY CONTROL STAGE              │
        │                                            │
        │  ┌──────────────────────────────────────┐ │
        │  │ data/qc/text/{project}/              │ │
        │  │ data/qc/csv/{project}/               │ │
        │  │ data/qc/results/{project}/           │ │
        │  └──────────────┬───────────────────────┘ │
        │                 │                          │
        │                 │ Q_QC_TIC.py              │
        │                 ▼                          │
        │  ┌──────────────────────────────────────┐ │
        │  │ data/qc/TIC/{project}/               │ │
        │  │   ├─ chromatograms/                  │ │
        │  │   └─ hidden_chromatograms/           │ │
        │  └──────────────────────────────────────┘ │
        └────────────────┬───────────────────────────┘
                         │
                         ├──────────────────────────┐
                         │                          │
                         │ PASS                     │ FAIL
                         ▼                          ▼
            ┌─────────────────────┐    ┌─────────────────────────┐
            │ data/text/          │    │ data/worklist/qc_fail/  │
            │ {project}/          │    │ {project}/              │
            │ (Production Ready)  │    │ (Reprocess Queue)       │
            └──────────┬──────────┘    └────────┬────────────────┘
                       │                        │
                       │                        │ Q_worklist.py
                       │                        ▼
                       │               ┌─────────────────────────┐
                       │               │ data/worklist/          │
                       │               │ generated/{project}/    │
                       │               │ (Reprocessing List)     │
                       │               └─────────────────────────┘
                       │
                       │ Q_viz_intensity.py
                       │ Q_viz_QC.py
                       ▼
            ┌─────────────────────┐
            │ data/viz/           │
            │ ├─ intensity/       │
            │ └─ QC/              │
            │ (Plots & Analysis)  │
            └─────────────────────┘
```

---

## 📊 Script Interaction Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCRIPT INTERACTIONS                          │
└─────────────────────────────────────────────────────────────────┘

Q_parse.py
  ├─ READS:  data/text/{project}/
  └─ WRITES: data/csv/{date}/
             data/logs/parse/{date}/

Q_QC.py (Main Orchestrator)
  ├─ READS:  data/qc/text/{project}/
  │          data/qc/csv/{project}/
  ├─ WRITES: data/qc/results/{project}/
  │          data/text/{project}/              (production files)
  │          data/worklist/qc_fail/{project}/  (failed samples)
  │          data/logs/qc/{project}/
  └─ CALLS:  Q_QC_TIC.py
             Q_worklist.py (for failed samples)

Q_QC_TIC.py
  ├─ READS:  data/qc/text/{project}/
  │          data/qc/results/{project}/
  └─ WRITES: data/qc/TIC/{project}/chromatograms/
             data/qc/TIC/{project}/hidden_chromatograms/

Q_viz_intensity.py
  ├─ READS:  data/qc/results/{project}/organized/
  └─ WRITES: data/viz/intensity/{project}/
             ├─ average_tic_*.png
             ├─ normalized_data.csv
             ├─ faceted_panels/
             └─ statistical_plots/

Q_viz_QC.py
  ├─ READS:  data/qc/results/{project}/
  └─ WRITES: data/viz/QC/{project}/

Q_worklist.py
  ├─ READS:  data/worklist/input/input_worklist.csv
  │          data/worklist/methods.csv
  │          data/worklist/qc_fail/{project}/  (optional)
  └─ WRITES: data/worklist/generated/{project}/
             data/logs/worklist/

Q_helper.py (RAG)
  ├─ READS:  notebooks/papers/qtrap_nano/  (PDFs)
  │          faiss_index_qtrap_nano/        (vector DB)
  └─ WRITES: None (read-only)
```

---

## 🗂️ Directory Hierarchy

```
data/
│
├── text/                          ◄─── RAW INPUT
│   ├── {project}/                      Raw MS .txt files
│   └── archive/                        Old projects
│
├── csv/                           ◄─── PARSED DATA
│   └── {date}/                         Structured CSV files
│
├── qc/                            ◄─── QUALITY CONTROL
│   ├── text/{project}/                 QC input text
│   ├── csv/{project}/                  QC parsed CSV
│   ├── results/{project}/              QC analysis results
│   │   ├── QC_{project}_RESULTS.csv
│   │   └── organized/
│   ├── TIC/{project}/                  TIC chromatograms
│   │   ├── chromatograms/
│   │   │   ├── png/
│   │   │   └── pdf/
│   │   └── hidden_chromatograms/
│   └── worklist/{project}/             QC-generated worklists
│
├── viz/                           ◄─── VISUALIZATIONS
│   ├── intensity/{project}/            Intensity plots
│   │   ├── average_tic_*.png
│   │   ├── normalized_data.csv
│   │   ├── faceted_panels/
│   │   └── statistical_plots/
│   └── QC/{project}/                   QC visualizations
│
├── worklist/                      ◄─── WORKLIST MANAGEMENT
│   ├── input/                          User input
│   │   ├── input_worklist.csv
│   │   └── all_methods.csv
│   ├── generated/{project}/            Auto-generated
│   ├── qc_fail/{project}/              Failed samples
│   └── methods.csv                     Lipid→Method lookup
│
└── logs/                          ◄─── LOGGING
    ├── parse/{date}/                   Parse logs
    ├── qc/{project}/                   QC logs
    ├── worklist/                       Worklist logs
    └── convert/                        Conversion logs
```

---

## 🎯 Data Flow by Stage

### **Stage 1: Data Acquisition**
```
MS Instrument → data/text/{project}/ → Q_parse.py
```

### **Stage 2: Parsing**
```
Q_parse.py → data/csv/{date}/
```

### **Stage 3: Quality Control**
```
data/qc/text/{project}/ → Q_QC.py → data/qc/results/{project}/
                            ↓
                       Q_QC_TIC.py → data/qc/TIC/{project}/
```

### **Stage 4: Decision Point**
```
QC Results → PASS → data/text/{project}/ (production)
          ↓
          FAIL → data/worklist/qc_fail/{project}/ → Q_worklist.py
```

### **Stage 5: Visualization**
```
data/qc/results/{project}/ → Q_viz_intensity.py → data/viz/intensity/{project}/
                           → Q_viz_QC.py → data/viz/QC/{project}/
```

### **Stage 6: Worklist Generation**
```
data/worklist/input/ + data/worklist/qc_fail/{project}/
                ↓
           Q_worklist.py
                ↓
    data/worklist/generated/{project}/
```

---

## 📈 Example: Solventmatrix Project Flow

```
1. Raw Data (67 files)
   data/text/solventmatrix/*.txt

2. Quality Control
   data/qc/text/solventmatrix/        (80 files)
   data/qc/csv/solventmatrix/         (80 files)
   data/qc/results/solventmatrix/     (5 files)
   data/qc/TIC/solventmatrix/         (2 files)

3. Visualization
   data/viz/intensity/solventmatrix/  (27 files)

4. Worklist Management
   data/worklist/generated/solventmatrix/ (29 files)

TOTAL: 293 files across complete pipeline
```

---

## 🔄 Continuous Monitoring Flow

```
┌─────────────────────────────────────────────────────┐
│         Q_QC.py CONTINUOUS MONITORING               │
└─────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │  Monitor Mode    │
    │  (continuous/    │
    │   interval/      │
    │   minute)        │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Scan data/qc/    │
    │ text/{project}/  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Analyze Files    │
    │ (QC metrics)     │
    └────────┬─────────┘
             │
             ├─ PASS ──► data/text/{project}/
             │
             └─ FAIL ──► data/worklist/qc_fail/{project}/
                         ↓
                    Generate Worklist
                         ↓
                    data/worklist/generated/{project}/
```

---

## 💡 Key Concepts

### **Project-Based Organization**
- Each project has its own subdirectories
- Consistent naming across all stages
- Easy to track complete workflow

### **Separation of Concerns**
- Raw data (`text/`) separate from processed (`qc/`, `viz/`)
- Logs separate from data
- Input separate from output

### **Automated Pipeline**
- Scripts read from expected locations
- Write to organized destinations
- Minimal manual intervention

### **Archive Strategy**
- Each directory has `archive/` subdirectory
- Old projects moved but not deleted
- Clean active directories

---

## 📝 Quick Reference Table

| Stage | Input | Script | Output | Purpose |
|-------|-------|--------|--------|---------|
| Parse | `text/` | Q_parse | `csv/` | Structure data |
| QC | `qc/text/` | Q_QC | `qc/results/` | Quality check |
| TIC | `qc/text/` | Q_QC_TIC | `qc/TIC/` | Chromatograms |
| Viz | `qc/results/` | Q_viz_* | `viz/` | Plots |
| Worklist | `worklist/input/` | Q_worklist | `worklist/generated/` | Sample lists |
| Reprocess | `qc_fail/` | Q_worklist | `worklist/generated/` | Failed samples |

---

**The QTRAP data directory is a well-organized, automated pipeline from raw data to final visualizations! 🚀**
