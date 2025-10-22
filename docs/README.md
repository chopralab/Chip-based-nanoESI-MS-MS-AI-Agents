# QTRAP Documentation

This directory contains all documentation, guides, environment setup files, and project reports for the QTRAP workflow.

---

## 📁 Directory Structure

```
docs/
├── README.md                           (This file)
├── environment/                        Environment setup files
│   ├── requirements.txt               Python dependencies (pip)
│   └── environment.yml                Conda environment
├── guides/                            User guides & implementation docs
│   ├── INTERVAL_MONITORING_GUIDE.md   Interval-based QC monitoring
│   ├── QC_MINUTE_MONITORING_GUIDE.md  Minute-based QC monitoring
│   ├── ADVANCED_VIZ_INTEGRATION_GUIDE.md  Advanced visualization
│   ├── FACETED_PANEL_PLOT_ADDED.md    Faceted panel plots
│   └── IMPLEMENTATION_SUMMARY.md      Implementation details
├── CLEANUP_SUMMARY.md                 Repository cleanup report
├── GIT_LARGE_FILES_REPORT.md          Git repository analysis
└── MISSING_FILES_AND_FIXES.md         Dependencies & fixes report
```

---

## 🚀 Quick Start

### Environment Setup

**Using pip:**
```bash
pip install -r docs/environment/requirements.txt
```

**Using conda:**
```bash
conda env create -f docs/environment/environment.yml
conda activate QTRAP_Agents
```

### LangGraph Project Setup
For the LangGraph QTRAP agents:
```bash
cd UI_qtrap/react-agent
pip install -e .
```

---

## 📚 Documentation

### User Guides

#### **QC Monitoring Guides**
- **[Interval Monitoring Guide](guides/INTERVAL_MONITORING_GUIDE.md)** - Set up interval-based QC monitoring
- **[Minute Monitoring Guide](guides/QC_MINUTE_MONITORING_GUIDE.md)** - Configure minute-by-minute QC monitoring

#### **Visualization Guides**
- **[Advanced Visualization Integration](guides/ADVANCED_VIZ_INTEGRATION_GUIDE.md)** - Advanced plotting features
- **[Faceted Panel Plots](guides/FACETED_PANEL_PLOT_ADDED.md)** - Multi-panel visualization setup

#### **Implementation Details**
- **[Implementation Summary](guides/IMPLEMENTATION_SUMMARY.md)** - Technical implementation overview

---

### Project Reports

#### **Repository Maintenance**
- **[Cleanup Summary](CLEANUP_SUMMARY.md)** - Complete repository cleanup documentation
- **[Git Large Files Report](GIT_LARGE_FILES_REPORT.md)** - Git repository size analysis
- **[Missing Files & Fixes](MISSING_FILES_AND_FIXES.md)** - Dependency resolution report

---

## 🔧 Environment Details

### Python Version
- **Required:** Python 3.11+

### Key Dependencies
- **LangGraph** - Agent orchestration framework
- **LangChain** - LLM framework
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **NumPy/SciPy** - Scientific computing

### Environment Files
- **`environment/requirements.txt`** - Pip-installable packages
- **`environment/environment.yml`** - Full conda environment specification

---

## 📖 Additional Resources

### Main QTRAP Scripts
Located in `/UI_qtrap/react-agent/src/react_agent/`:
- `Q_worklist.py` - Worklist generation
- `Q_parse.py` - Data parsing
- `Q_QC.py` - Quality control analysis
- `Q_helper.py` - RAG helper agent
- `Q_viz_*.py` - Visualization scripts

### LangGraph Configuration
- **Config:** `/UI_qtrap/react-agent/langgraph.json`
- **Agents:** 4 active agents (worklist, parse, QC, helper)

---

## 🎯 For Presentation

This documentation is organized for the **ACS Analytical Chemistry** presentation.

**Key Highlights:**
- Clean, organized structure
- Comprehensive guides for all features
- Easy environment setup
- Well-documented workflows

---

## 📝 Notes

- All guides are in Markdown format for easy viewing on GitHub
- Environment files support both pip and conda workflows
- Project reports document the cleanup and optimization process
- For LangGraph-specific dependencies, see `UI_qtrap/react-agent/pyproject.toml`
