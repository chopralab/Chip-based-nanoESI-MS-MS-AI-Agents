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
# Create conda environment
conda create --name qtrap_paper --clone sciborg_dev
# Or create from scratch:
conda env create -f docs/environment/environment.yml

# Activate environment
conda activate qtrap_paper

# Install additional dependencies
pip install python-dotenv  # For secure API key loading
```

### 2. **Configure API Keys** (Secure Method)

```bash
# Create .env file for Helper Agent
cd helper_agent
cp .env.example .env
nano .env  # Add your OPENAI_API_KEY
```

See [`helper_agent/API_KEY_SETUP.md`](helper_agent/API_KEY_SETUP.md) for detailed security setup.

### 3. **Using Helper Agent (Standalone)**

```python
import sys
sys.path.insert(0, "helper_agent/drivers")

from config_loader import setup_environment
from qtrap_utils import setup_helper, ask_agent, tell_agent

# Load API key securely
setup_environment()

# Setup agent
helper = setup_helper(model="gpt-4o", temperature=0)

# Ask questions (searches literature)
answer = ask_agent("What solvent for lipidomics?")

# Store your experimental findings
tell_agent("Project SolventMatrix: 2:1 MeOH/ACN is best...")

# Ask again (now uses YOUR data!)
answer = ask_agent("What solvent for lipidomics?")
```

### 4. **Using LangGraph Agents** (Optional)

```bash
# Install LangGraph
cd UI_qtrap/react-agent
pip install -e .

# Configure .env
cp .env.example .env
# Add: OPENAI_API_KEY, TAVILY_API_KEY

# Start LangGraph UI
langgraph dev
```

Open browser to `http://localhost:8123` to access LangGraph Studio.

---

## 📁 Repository Structure

```
Chip-based-nanoESI-MS-MS-AI-Agents/
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
├── helper_agent/                # 🆕 Standalone Helper AI Agent
│   ├── drivers/                # Core modules
│   │   ├── qtrap_utils.py      # Main API (use this!)
│   │   ├── helper_agent_core.py # Agent implementation
│   │   ├── config_loader.py    # Secure API key loading
│   │   ├── archive/            # Old/unused files
│   │   └── README.md           # Module documentation
│   │
│   ├── logs/                   # Organized log files
│   │   ├── ask_agent/          # Memory-first queries
│   │   ├── tell_agent/         # Memory storage
│   │   ├── query/              # Full RAG queries
│   │   └── README.md           # Log documentation
│   │
│   ├── papers/                 # Scientific literature corpus
│   │   └── recentlipids7/      # Lipidomics papers
│   │       └── faiss_index/    # Vector embeddings
│   │
│   ├── helper_agent_notebooks/ # Example notebooks
│   ├── .env                    # API keys (gitignored)
│   ├── .env.example            # API key template
│   ├── API_KEY_SETUP.md        # Security guide
│   └── ORGANIZATION_SUMMARY.md # Complete organization guide
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
- Memory-first queries (checks stored data before searching literature)
- Automatic logging of all interactions
- Secure API key management

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
- `Q_helper.py` - RAG-based literature assistant (LangGraph integration)
- `helper_agent/` - Standalone Helper AI Agent system:
  - `qtrap_utils.py` - Main API for agent operations
  - `helper_agent_core.py` - Core RAG and memory implementation
  - `config_loader.py` - Secure API key management
  - Organized logging system (ask/tell/query logs)
  - Example notebooks with Project SolventMatrix demo

---

## 📊 Key Dependencies

### **Core AI Framework**
- **LangGraph** - Agent orchestration and workflow management
- **LangChain** - LLM integration and tool calling
- **OpenAI GPT-4o** - Language model for agent reasoning
- **FAISS** - Vector database for RAG and literature search

### **Data Processing**
- **Pandas** - Data manipulation and analysis
- **NumPy/SciPy** - Statistical analysis
- **python-dotenv** - Secure environment variable management

### **Visualization**
- **Matplotlib/Seaborn** - Scientific visualization

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
