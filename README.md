# Memory-Based AI Agents Integrated with Real-Time Quality Control Automates Chip-based nanoESI-MS/MS Platform 

---

## 🎯 Overview

This repository implements the first LLM-driven AI agent framework for chip-based nanoESI-MS/MS automation, specifically designed for TriVersa NanoMate coupled to SCIEX 4000 QTRAP systems. The framework leverages large language models to interpret natural language instructions and autonomously execute analytical workflows—from worklist generation and real-time quality control to data conversion, visualization, and literature retrieval—without requiring programming expertise. Unlike traditional vendor software or custom scripts, the system integrates persistent memory to preserve both lab-specific protocols and published knowledge, while maintaining a modular, instrument-agnostic architecture that can evolve as experimental needs change across diverse analytical platforms beyond mass spectrometry.

---

## ✨ Key Features

### 🤖 **AI-Powered Agents**
- **QC Agent** - Automated real-time quality control
- **Worklist Agent** - Intelligent sample worklist generation
- **Parse Agent** - Automated data extraction and organization
- **Visualization Agent** - Automated data visualization
- **Helper Agent** - Retrival Augmented Generation (RAG) for scientific literature and persistent memory for knowledge retention

---

## 🚀 Quick Start

### 1. **Environment Setup**

```bash
# Using conda (recommended)
conda env create -f docs/environment/environment.yml
conda activate QTRAP_Agents

# Or using pip
pip install -r docs/environment/requirements.txt
```

### 2. **Install LangGraph Project**

```bash
cd UI_qtrap/react-agent
pip install -e .
```

### 3. **Configure Environment**

Create a `.env` file in `UI_qtrap/react-agent/`:
```bash
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=false  # Set to false for local dev
```

### 4. **Start LangGraph UI**

```bash
cd UI_qtrap/react-agent
langgraph dev
```

Open browser to `http://localhost:8123` to access the LangGraph Studio UI.

---

## 📁 Repository Structure

```
sciborg_dev/
├── README.md                    # This file
├── .gitignore                   # Git configuration
│
├── UI_qtrap/                    # Main QTRAP workflow
│   └── react-agent/            # LangGraph agents
│       ├── src/react_agent/    # Core scripts
│       │   ├── Q_QC.py         # QC analysis (main)
│       │   ├── Q_worklist.py   # Worklist generation
│       │   ├── Q_parse.py      # Data parsing
│       │   ├── Q_helper.py     # RAG helper
│       │   └── Q_viz_*.py      # Visualization scripts
│       └── langgraph.json      # Agent configuration
│
├── notebooks/                   # Papers & supplemental info
│   ├── papers/                 # Scientific literature
│   └── SI/                     # Supplemental information
│
└── docs/                        # Documentation
    ├── README.md               # Documentation index
    ├── environment/            # Setup files
    ├── guides/                 # User guides
    ├── CLEANUP_SUMMARY.md      # Repo maintenance
    └── GIT_LARGE_FILES_REPORT.md
```

---

## 🤖 LangGraph Agents

### **agent_QC** - Quality Control
- Monitors MS data quality in real-time
- Detects failed samples automatically
- Generates TIC plots and QC metrics
- Triggers reprocessing workflows

### **agent_worklist** - Worklist Generation
- Creates optimized sample worklists
- Integrates failed sample reprocessing
- Organizes by lipid class and project

### **agent_parse** - Data Parsing
- Extracts data from raw MS files
- Converts to structured CSV format
- Organizes by date and project

### **agent_helper** - Literature Assistant
- RAG-based Q&A on scientific papers
- FAISS vector database integration
- Answers questions about MS methods

---

## 📚 Documentation

### **User Guides**
- [Interval Monitoring Guide](docs/guides/INTERVAL_MONITORING_GUIDE.md)
- [Minute Monitoring Guide](docs/guides/QC_MINUTE_MONITORING_GUIDE.md)
- [Advanced Visualization](docs/guides/ADVANCED_VIZ_INTEGRATION_GUIDE.md)
- [Faceted Panel Plots](docs/guides/FACETED_PANEL_PLOT_ADDED.md)

### **Technical Documentation**
- [Implementation Summary](docs/guides/IMPLEMENTATION_SUMMARY.md)
- [Cleanup Summary](docs/CLEANUP_SUMMARY.md)
- [Git Analysis](docs/GIT_LARGE_FILES_REPORT.md)

**Full documentation:** See [`docs/README.md`](docs/README.md)

---

## 🔧 Core Scripts

### **Quality Control**
- `Q_QC.py` (98 KB) - Main QC workflow with monitoring
- `Q_QC_TIC.py` (17 KB) - TIC extraction and plotting
- `Q_viz_QC.py` (29 KB) - QC visualization

### **Data Processing**
- `Q_parse.py` (15 KB) - Data parsing and extraction
- `Q_worklist.py` (22 KB) - Worklist generation
- `Q_convert.py` (5 KB) - MSConvert integration

### **Visualization**
- `Q_viz_intensity.py` (90 KB) - Intensity visualization
- `Q_viz_intensity_advanced.py` (16 KB) - Advanced plots
- `Q_viz_intensity_advanced_part2.py` (18 KB) - Statistical plots

### **AI Helper**
- `Q_helper.py` (5 KB) - RAG-based literature assistant

---

## 🎯 Workflow Examples

### **Continuous QC Monitoring**
```python
# Monitors data directory continuously
# Automatically detects and flags failed samples
# Generates TIC plots and QC metrics
# Triggers reprocessing workflows
```

### **Automated Worklist Generation**
```python
# Scans QC results for failed samples
# Creates optimized reprocessing worklists
# Organizes by lipid class and project
# Integrates with MS instrument software
```

### **Intelligent Data Parsing**
```python
# Extracts data from raw MS files
# Converts to structured CSV format
# Organizes by date and project structure
# Handles multiple file formats
```

---

## 📊 Key Technologies

- **LangGraph** - Agent orchestration and workflow management
- **LangChain** - LLM integration and tool calling
- **OpenAI GPT** - Language model for agent reasoning
- **FAISS** - Vector database for literature search
- **Pandas** - Data manipulation and analysis
- **Matplotlib/Seaborn** - Scientific visualization
- **NumPy/SciPy** - Statistical analysis

---

## 🎓 Citation

If you use this workflow in your research, please cite:

```
QTRAP: AI-Powered Mass Spectrometry Quality Control Workflow
Presented at ACS Analytical Chemistry, 2025
```

---

## 📝 License

[Add your license here]

---

## 👥 Contributors

[Add contributors here]

---

## 📧 Contact

For questions or collaboration:
- [Add contact information]

---

## 🌟 Acknowledgments

This project uses:
- LangGraph by LangChain
- OpenAI API
- Scientific Python ecosystem

---

**Status:** ✅ Presentation Ready  
**Last Updated:** October 22, 2025
