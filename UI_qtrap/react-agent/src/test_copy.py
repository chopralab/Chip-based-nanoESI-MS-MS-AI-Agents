import shutil
import os

# Source files
src1 = "/home/qtrap/Chip-based-nanoESI-MS-MS-AI-Agents/UI_qtrap/react-agent/src/react_agent/data/worklist/generated/worklist_20250421_v3.csv"
src2 = "/home/qtrap/Chip-based-nanoESI-MS-MS-AI-Agents/UI_qtrap/react-agent/src/react_agent/data/worklist/generated/worklist_20250609_407pm.csv"

# Destination directory
server_dir = "/mnt/d_drive/Analyst Data/Projects/API Instrument/Batch/QTRAP_worklist"

# Ensure destination exists
os.makedirs(server_dir, exist_ok=True)

# Copy files
for src in [src1, src2]:
    dest = os.path.join(server_dir, os.path.basename(src))
    shutil.copy2(src, dest)
    print(f"Copied {src} to {dest}")
