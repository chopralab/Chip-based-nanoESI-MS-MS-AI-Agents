#!/bin/bash
# Script to organize existing logs into subdirectories

LOG_DIR="/home/sanjay/QTRAP_paper/Chip-based-nanoESI-MS-MS-AI-Agents/helper_agent/logs"

# Create subdirectories
mkdir -p "$LOG_DIR/ask_agent"
mkdir -p "$LOG_DIR/tell_agent"
mkdir -p "$LOG_DIR/query"

# Move existing logs to appropriate subdirectories
echo "Organizing logs..."

# Move ask_agent logs
if ls "$LOG_DIR"/ask_agent_log_*.json >/dev/null 2>&1; then
    mv "$LOG_DIR"/ask_agent_log_*.json "$LOG_DIR/ask_agent/" 2>/dev/null
    echo "✓ Moved ask_agent JSON logs"
fi

if ls "$LOG_DIR"/ask_agent_log_*.txt >/dev/null 2>&1; then
    mv "$LOG_DIR"/ask_agent_log_*.txt "$LOG_DIR/ask_agent/" 2>/dev/null
    echo "✓ Moved ask_agent TXT logs"
fi

# Move tell_agent logs
if ls "$LOG_DIR"/tell_agent_log_*.json >/dev/null 2>&1; then
    mv "$LOG_DIR"/tell_agent_log_*.json "$LOG_DIR/tell_agent/" 2>/dev/null
    echo "✓ Moved tell_agent JSON logs"
fi

if ls "$LOG_DIR"/tell_agent_log_*.txt >/dev/null 2>&1; then
    mv "$LOG_DIR"/tell_agent_log_*.txt "$LOG_DIR/tell_agent/" 2>/dev/null
    echo "✓ Moved tell_agent TXT logs"
fi

# Move query logs
if ls "$LOG_DIR"/query_log_*.json >/dev/null 2>&1; then
    mv "$LOG_DIR"/query_log_*.json "$LOG_DIR/query/" 2>/dev/null
    echo "✓ Moved query JSON logs"
fi

if ls "$LOG_DIR"/query_log_*.txt >/dev/null 2>&1; then
    mv "$LOG_DIR"/query_log_*.txt "$LOG_DIR/query/" 2>/dev/null
    echo "✓ Moved query TXT logs"
fi

echo ""
echo "Log organization complete!"
echo ""
echo "Directory structure:"
tree -L 2 "$LOG_DIR" 2>/dev/null || ls -la "$LOG_DIR"

echo ""
echo "Summary:"
echo "  ask_agent logs:  $(ls "$LOG_DIR/ask_agent" 2>/dev/null | wc -l) files"
echo "  tell_agent logs: $(ls "$LOG_DIR/tell_agent" 2>/dev/null | wc -l) files"
echo "  query logs:      $(ls "$LOG_DIR/query" 2>/dev/null | wc -l) files"
