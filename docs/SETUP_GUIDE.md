# QTRAP_Agents Environment Setup Guide

## Quick Setup Instructions

### Step 1: Create the Conda Environment

```bash
cd /home/qtrap/sciborg_dev
conda env create -f docs/environment/environment.yml
```

This will create a new conda environment named **`QTRAP_Agents`** with all required dependencies.

### Step 2: Activate the Environment

```bash
conda activate QTRAP_Agents
```

### Step 3: Install LangGraph Project

```bash
cd UI_qtrap/react-agent
pip install -e .
```

### Step 4: Configure Environment Variables

Create a `.env` file in `UI_qtrap/react-agent/`:

```bash
cd /home/qtrap/sciborg_dev/UI_qtrap/react-agent
nano .env
```

Add your API keys:
```
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
LANGCHAIN_TRACING_V2=false
```

### Step 5: Start LangGraph UI

```bash
cd /home/qtrap/sciborg_dev/UI_qtrap/react-agent
langgraph dev
```

Open browser to: `http://localhost:8123`

---

## Environment Details

- **Name:** QTRAP_Agents
- **Python Version:** 3.11.10
- **Location:** `/home/qtrap/anaconda3/envs/QTRAP_Agents`

### Key Packages Included:
- LangGraph & LangChain - AI agent framework
- OpenAI - LLM integration
- FAISS - Vector database
- Pandas, NumPy - Data processing
- Matplotlib, Seaborn - Visualization

---

## Troubleshooting

### If environment already exists:
```bash
conda env remove -n QTRAP_Agents
conda env create -f docs/environment/environment.yml
```

### If you need to update the environment:
```bash
conda activate QTRAP_Agents
conda env update -f docs/environment/environment.yml --prune
```

### Verify installation:
```bash
conda activate QTRAP_Agents
python -c "import langchain; import langgraph; print('Success!')"
```

---

## Alternative: Using pip only

If you prefer not to use conda:

```bash
cd /home/qtrap/sciborg_dev
python3.11 -m venv qtrap_venv
source qtrap_venv/bin/activate
pip install -r docs/environment/requirements.txt
cd UI_qtrap/react-agent
pip install -e .
```

---

## Next Steps

After setup, see:
- [Main README](../README.md) - Project overview
- [Documentation Index](README.md) - All guides
- [User Guides](guides/) - Feature-specific guides
