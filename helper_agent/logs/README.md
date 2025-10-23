# Logs Directory Structure

This directory contains all logs from the Helper AI Agent system, organized by function type.

## 📁 Directory Structure

```
logs/
├── ask_agent/          # Memory-first query logs
├── tell_agent/         # Memory storage logs
├── query/              # Full RAG query logs (run_helper_query)
└── archive/            # Optional: old/archived logs
```

---

## 📊 Log Types

### **1. `ask_agent/` - Memory-First Queries**
- **Function**: `ask_agent()`
- **Purpose**: Queries that check memory first, then fall back to RAG
- **Files**: `ask_agent_log_YYYYMMDD_HHMMSS.{json,txt}`
- **Contains**:
  - Question asked
  - Whether memory was used
  - Number of memory entries found
  - Answer (from memory or RAG)
  - Citations (if from RAG)

**Example:**
```
ask_agent/
├── ask_agent_log_20251022_212527.json
└── ask_agent_log_20251022_212527.txt
```

---

### **2. `tell_agent/` - Memory Storage**
- **Function**: `tell_agent()`
- **Purpose**: Logs of experimental data stored in memory
- **Files**: `tell_agent_log_YYYYMMDD_HHMMSS.{json,txt}`
- **Contains**:
  - Timestamp
  - Memory key assigned
  - Full observation/data stored
  - Observation length
  - Namespace used

**Example:**
```
tell_agent/
├── tell_agent_log_20251022_212527.json
└── tell_agent_log_20251022_212527.txt
```

---

### **3. `query/` - Full RAG Queries**
- **Function**: `run_helper_query()`
- **Purpose**: Full RAG queries with detailed step-by-step logging
- **Files**: `query_log_YYYYMMDD_HHMMSS.{json,txt}`
- **Contains**:
  - Detailed search process
  - Document retrieval steps
  - Context retrieved
  - LLM prompting
  - Final answer with citations
  - Performance metrics

**Example:**
```
query/
├── query_log_20251022_202838.json
└── query_log_20251022_202838.txt
```

---

## 🔍 Finding Logs

### **View recent ask_agent logs:**
```bash
ls -lt logs/ask_agent/ | head
```

### **View recent tell_agent logs:**
```bash
ls -lt logs/tell_agent/ | head
```

### **View recent query logs:**
```bash
ls -lt logs/query/ | head
```

### **Search logs by date:**
```bash
# Find all logs from October 22, 2025
find logs/ -name "*20251022*.txt"
```

### **Search logs by content:**
```bash
# Find all logs mentioning "SolventMatrix"
grep -r "SolventMatrix" logs/
```

---

## 📝 Log Formats

Each log type creates **two files**:

1. **`.json`** - Machine-readable format for analysis
2. **`.txt`** - Human-readable format for review

---

## 🧹 Maintenance

### **Clean old logs:**
```bash
# Delete logs older than 30 days
find logs/ -name "*.json" -mtime +30 -delete
find logs/ -name "*.txt" -mtime +30 -delete
```

### **Archive logs by month:**
```bash
# Create monthly archive
mkdir -p logs/archive/2025_10/
mv logs/*/\*202510*.* logs/archive/2025_10/
```

---

## ⚙️ Configuration

Logging is automatic by default. To disable:

```python
# Disable logging for specific calls
ask_agent("question", save_log=False)
tell_agent("data", save_log=False)

# Custom log directory for run_helper_query
run_helper_query("question", log_dir="/custom/path")
```

---

## 📊 Log Statistics

View log counts:
```bash
echo "ask_agent logs:  $(ls logs/ask_agent/ 2>/dev/null | wc -l) files"
echo "tell_agent logs: $(ls logs/tell_agent/ 2>/dev/null | wc -l) files"
echo "query logs:      $(ls logs/query/ 2>/dev/null | wc -l) files"
```

View total size:
```bash
du -sh logs/*/
```
