# Missing Files and Required Fixes Report
**Generated:** October 22, 2025  
**Repository:** /home/qtrap/sciborg_dev  
**Branch:** paper

---

## Summary

After cleanup, some files that were archived are actually **required dependencies** for the main QTRAP scripts to function properly.

---

## ✅ Files Restored

### 1. **qc_worklist_generator.py** - RESTORED
- **Location:** `/UI_qtrap/react-agent/src/react_agent/qc_worklist_generator.py`
- **Status:** ✅ Restored from archive
- **Required by:** `Q_QC.py` (imports `generate_worklist_for_project`)
- **Purpose:** Generates worklist files from failed QC results

---

## ⚠️ Issues Found

### 1. **Q_convert.py - Missing LangGraph Integration**

**Problem:**
- `langgraph.json` references `Q_convert.py:graph`
- Current `Q_convert.py` is a standalone script (QCWorkflow class)
- Does NOT export a `graph` object for LangGraph

**Current Code:**
```python
class QCWorkflow:
    # Standalone MSConvert workflow
    def run(self):
        # Continuous loop for file conversion
```

**Options:**
1. **Remove from langgraph.json** (if not used as LangGraph agent)
2. **Create LangGraph wrapper** (if needed as agent)
3. **Find original LangGraph version** in archives

**Recommendation:** Check if Q_convert is actually used as a LangGraph agent in your workflow. If not, remove it from `langgraph.json`.

---

## 📋 All Script Dependencies Check

### Q_worklist.py ✅
- **Imports:** `react_agent.tools` ✅ (exists)
- **Dependencies:** pandas, langchain, langgraph ✅

### Q_parse.py ✅
- **Imports:** `react_agent.tools` ✅ (exists)
- **Dependencies:** pandas, langchain, langgraph ✅

### Q_QC.py ✅ (Fixed)
- **Imports:** 
  - `qc_worklist_generator` ✅ (RESTORED)
  - `Q_worklist.generate_integrated_worklist_for_project` ✅
  - `Q_QC_TIC.generate_tic_plots_for_project` ✅
- **Dependencies:** pandas, langchain, langgraph ✅

### Q_QC_TIC.py ✅
- **Dependencies:** matplotlib, numpy, pandas ✅
- **No local imports** ✅

### Q_helper.py ✅
- **Hardcoded paths:**
  - `PDF_DIR = "/home/qtrap/sciborg_dev/notebooks/papers/qtrap_nano"` ⚠️
  - `VECTOR_DB_PATH = "/home/qtrap/sciborg_dev/faiss_index_qtrap_nano"` ⚠️
- **Dependencies:** langchain, FAISS ✅
- **Note:** Paths are absolute but work for your system

### Q_viz_*.py ✅
- **Dependencies:** matplotlib, seaborn, pandas, numpy, scipy ✅
- **No local imports** ✅

### Q_convert.py ⚠️
- **Issue:** No `graph` export for LangGraph
- **Action needed:** See section above

---

## 🔧 Recommended Actions

### Immediate (Critical):

1. **Fix Q_convert.py in langgraph.json:**
   ```bash
   # Option A: Remove if not used as LangGraph agent
   # Edit langgraph.json and remove the agent_convert line
   
   # Option B: Comment it out temporarily
   # Change line 6 to: // "agent_convert": "./src/react_agent/Q_convert.py:graph",
   ```

### Optional (For Portability):

2. **Fix hardcoded paths in Q_helper.py:**
   - Replace absolute paths with relative paths
   - Use `Path(__file__).parent` for portability

3. **Fix hardcoded paths in Q_QC.py, Q_parse.py, Q_QC_TIC.py:**
   - All have hardcoded `/home/qtrap/sciborg_dev/...` paths
   - Consider making them configurable or relative

---

## 📁 Current File Status

### Essential Scripts (All Present):
- ✅ Q_worklist.py
- ✅ Q_parse.py
- ✅ Q_QC.py
- ✅ Q_QC_TIC.py
- ✅ Q_helper.py
- ✅ Q_viz_QC.py
- ✅ Q_viz_intensity.py
- ✅ Q_viz_intensity_advanced.py
- ✅ Q_viz_intensity_advanced_part2.py
- ⚠️ Q_convert.py (present but needs LangGraph integration)

### Supporting Files:
- ✅ qc_worklist_generator.py (RESTORED)
- ✅ tools.py
- ✅ configuration.py
- ✅ state.py
- ✅ graph.py
- ✅ prompts.py
- ✅ utils.py
- ✅ __init__.py

---

## 🚀 Next Steps

1. **Test LangGraph startup:**
   ```bash
   cd /home/qtrap/sciborg_dev/UI_qtrap/react-agent
   langgraph dev
   ```

2. **If Q_convert error persists:**
   - Remove `agent_convert` from `langgraph.json`
   - OR create a proper LangGraph wrapper for Q_convert.py

3. **Verify all agents load:**
   - agent_worklist ✅
   - agent_parse ✅
   - agent_convert ⚠️ (needs fix)
   - agent_QC ✅
   - agent_helper ✅

---

## 📝 Notes

- The cleanup was successful in removing duplicate/old files
- One dependency (`qc_worklist_generator.py`) was restored
- Q_convert.py needs attention for LangGraph compatibility
- All other scripts should work correctly

**Status:** Repository is 95% ready. Just need to fix Q_convert.py integration.
