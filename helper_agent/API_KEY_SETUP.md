# How to Setup API Key (Secure Method)

## ✅ Method 1: Using .env File (RECOMMENDED)

### Step 1: Install python-dotenv
```bash
pip install python-dotenv
```

### Step 2: Create `.env` file
```bash
cd /home/sanjay/QTRAP_paper/Chip-based-nanoESI-MS-MS-AI-Agents/helper_agent
nano .env
```

### Step 3: Add your API key to `.env`
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

### Step 4: Use in your notebook
```python
# CELL 1: Load API key securely
import sys
sys.path.insert(0, "/home/sanjay/QTRAP_paper/Chip-based-nanoESI-MS-MS-AI-Agents/helper_agent/drivers")

from config_loader import setup_environment

# Load API key from .env file (not visible in notebook!)
setup_environment()

# Now use qtrap_utils as normal
from qtrap_utils import setup_helper, ask_agent, tell_agent

helper = setup_helper(model="gpt-4o", temperature=0)
```

---

## ✅ Method 2: System Environment Variable

### Set in your shell (add to ~/.bashrc for persistence)
```bash
export OPENAI_API_KEY="sk-proj-your-actual-key-here"
```

### Use in notebook
```python
import os

# API key is already in environment - nothing to do!
from qtrap_utils import setup_helper

helper = setup_helper(model="gpt-4o", temperature=0)
```

---

## ✅ Method 3: Using qtrap_utils with auto-loading

Update `qtrap_utils.py` to auto-load the API key:

```python
# At the top of qtrap_utils.py
from config_loader import setup_environment

# Auto-load API key when module is imported
try:
    setup_environment()
except ValueError:
    print("⚠️  API key not found - please set it before using the module")
```

Then your notebooks just need:
```python
from qtrap_utils import setup_helper  # API key loads automatically!
helper = setup_helper()
```

---

## 🔒 Security Best Practices

1. ✅ **DO**: Use `.env` file (already in .gitignore)
2. ✅ **DO**: Use environment variables
3. ✅ **DO**: Use config_loader module
4. ❌ **DON'T**: Put API keys directly in notebooks
5. ❌ **DON'T**: Commit `.env` file to git (it's already ignored)
6. ❌ **DON'T**: Share notebooks with API keys visible

---

## 📁 File Structure

```
helper_agent/
├── .env                    # Your API key (GITIGNORED - safe!)
├── .env.example           # Template (safe to commit)
├── drivers/
│   ├── config_loader.py   # Secure loader
│   └── qtrap_utils.py     # Main utilities
└── helper_agent_notebooks/
    └── your_notebook.ipynb
```
