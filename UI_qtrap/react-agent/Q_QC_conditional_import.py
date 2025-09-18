# Alternative Solution 2: Conditional Import
# Replace the import section in Q_QC.py with this:

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

# Conditional Import QC Worklist Generator
try:
    # Try relative import first (works in package context)
    from .qc_worklist_generator import generate_worklist_for_project
except ImportError:
    try:
        # Try absolute import (works when executed standalone)
        from qc_worklist_generator import generate_worklist_for_project
    except ImportError:
        # Fallback: add current directory to path and import
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        from qc_worklist_generator import generate_worklist_for_project
