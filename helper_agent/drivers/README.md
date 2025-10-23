# Drivers Directory

This directory contains the core modules for the QTRAP Helper AI Agent system.

## 📁 Active Files (Currently Used)

### **1. `qtrap_utils.py`** - Main Utility Module ⭐
**Purpose**: High-level API for all agent operations  
**Used by**: All notebooks  
**Functions**:
- `setup_helper()` - Initialize the complete system
- `ask_agent()` - Memory-first queries
- `tell_agent()` - Store experimental data
- `run_helper_query()` - Full RAG queries with logging
- `clear_memory()` - Clear stored memories
- `list_memories()` - View stored memories

**Usage**:
```python
from qtrap_utils import setup_helper, ask_agent, tell_agent

helper = setup_helper(model="gpt-4o", temperature=0)
tell_agent("Your experimental data...")
answer = ask_agent("Your question?")
```

---

### **2. `helper_agent_core.py`** - Core Agent Implementation
**Purpose**: Base HelperAIAgent class with RAG and memory functionality  
**Used by**: `qtrap_utils.py`  
**Key Classes**:
- `HelperAIAgent` - Main agent class with RAG, memory, and LLM

**Note**: This is imported by `qtrap_utils.py`. Users typically don't need to import this directly.

---

### **3. `config_loader.py`** - Secure API Key Management
**Purpose**: Load API keys securely from environment or .env file  
**Used by**: Notebooks (for API key setup)  
**Functions**:
- `load_api_key()` - Load from .env or environment
- `setup_environment()` - Configure environment variables

**Usage**:
```python
from config_loader import setup_environment
setup_environment()  # Loads OPENAI_API_KEY from .env
```

---

## 📦 Module Dependencies

```
Notebooks
    ↓
config_loader.py (API keys)
    ↓
qtrap_utils.py (High-level API)
    ↓
helper_agent_core.py (Core agent)
    ↓
LangChain + OpenAI + FAISS
```

---

## 🗂️ Archived Files (`archive/`)

These files are **not currently used** and have been archived:

### **`HelperAgent_v2.py`**
- **Status**: Replaced by `helper_agent_core.py`
- **Why archived**: Renamed for clarity

### **`HelperAgent_v2_1022_822pm.py`**
- **Status**: Old version
- **Why archived**: Superseded by newer version

### **`core2.py` & `core3.py`**
- **Status**: From old sciborg_dev system
- **Why archived**: Not used in current QTRAP system
- **Note**: These were part of a different project

---

## 🎯 Naming Convention

**Current naming standard** (clear and descriptive):

| File | Purpose | Naming Logic |
|------|---------|--------------|
| `qtrap_utils.py` | Main utilities | `{project}_{type}.py` |
| `helper_agent_core.py` | Core agent class | `{module}_core.py` |
| `config_loader.py` | Configuration loading | `{purpose}_loader.py` |

**Old naming** (archived):
- `HelperAgent_v2.py` → CamelCase, version number unclear
- `core2.py`, `core3.py` → Generic names, unclear purpose

---

## 🔧 Adding New Modules

When adding new functionality:

1. **Name clearly**: `{purpose}_{type}.py`
   - Example: `data_processor.py`, `lipid_analyzer.py`

2. **Add docstring at top**:
```python
"""
Module Purpose Here
===================
Brief description of what this module does.

Usage:
    from module_name import function_name
"""
```

3. **Update this README** with module description

4. **If replacing old module**: Move old version to `archive/`

---

## 📊 Module Sizes

```bash
config_loader.py         2.9 KB   (Lightweight)
qtrap_utils.py          21 KB    (Main API)
helper_agent_core.py    29 KB    (Core agent)
```

---

## 🧪 Testing Imports

To verify all imports work:

```python
# Test from notebooks directory
import sys
sys.path.insert(0, "../drivers")

from config_loader import setup_environment
from qtrap_utils import setup_helper, ask_agent, tell_agent
from helper_agent_core import HelperAIAgent

print("✓ All imports successful!")
```

---

## 🔄 Migration Guide

If you have old notebooks using `HelperAgent_v2`:

**Old code:**
```python
from HelperAgent_v2 import HelperAIAgent
```

**New code:**
```python
from qtrap_utils import setup_helper, ask_agent, tell_agent
# helper_agent_core is imported automatically by qtrap_utils
```

---

## 📝 Maintenance

### **Cleaning Archive**
```bash
# If you're sure you don't need archived files
rm -rf drivers/archive/
```

### **Checking What's Imported**
```bash
# Find all imports in notebooks
grep -r "from.*import" ../helper_agent_notebooks/*.ipynb
```

---

## ✅ Summary

**Active (3 files)**:
1. ✅ `qtrap_utils.py` - Main API (use this!)
2. ✅ `helper_agent_core.py` - Core agent (imported by qtrap_utils)
3. ✅ `config_loader.py` - API key loading

**Archived (4 files)**:
1. 📦 `HelperAgent_v2.py` - Old name
2. 📦 `HelperAgent_v2_1022_822pm.py` - Old version
3. 📦 `core2.py` - From different project
4. 📦 `core3.py` - From different project
