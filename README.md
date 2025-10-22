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
- Creates worklists formatted for Analyst v1.6.3
- Formats for windows 7 and moves files to QTRAP worklist directory automatically 
- Operator initials for each experiemnt

### **agent_parse** - Data Parsing
- Extracts data from raw .wiff files
- Converts to structured CSV format
- Organizes by project

### **agent_helper** - Literature Assistant
- RAG-based Q&A on scientific papers
- Persistent memory for knowledge retention

---

## 📚 Documentation

### **User Guides**
- [`docs/README.md`](docs/README.md)

---

## 🔧 Core Scripts

### **Quality Control**
- `Q_QC.py` - Main QC workflow with monitoring
- `Q_QC_TIC.py` - TIC extraction and plotting
- `Q_viz_QC.py` - QC visualization

### **Data Processing**
- `Q_parse.py` - Data parsing and extraction
- `Q_worklist.py` - Worklist generation
- `Q_convert.py` - MSConvert integration

### **Visualization**
- `Q_viz_intensity.py` - Intensity visualization
- `Q_viz_intensity_advanced.py` - Advanced plots
- `Q_viz_intensity_advanced_part2.py` - Statistical plots

### **AI Helper**
- `Q_helper.py` - RAG-based literature assistant and persistent memory for knowledge retention

---

## 📊 Key Dependencies

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
Memory-Based AI Agents Integrated with Real-Time Quality Control Automates Chip-based nanoESI-MS/MS Platform
(In Preparation)
```

---

## 📧 Contact

For questions or collaboration:
- iyer95@purdue.edu

---

## 🌟 Acknowledgments

This project uses:
- LangGraph by LangChain
- OpenAI API
- Scientific Python ecosystem

---

**Last Updated:** October 22, 2025
