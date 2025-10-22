# Git Ignore Strategy for Data Files

## 🎯 Current Setup

The `.gitignore` is configured to:
1. **Ignore all data files by default** (*.txt, *.csv, *.wiff, etc.)
2. **Keep specific example/demo files** for presentation
3. **Keep all documentation files**

---

## 📁 How to Include Example Data Files

### **Option 1: Use `examples/` Directory**

Create an `examples/` folder anywhere in your repo:

```bash
# Example structure:
UI_qtrap/react-agent/examples/
├── sample_worklist.csv
├── example_qc_output.txt
└── demo_results.csv
```

**These files WILL be tracked** because of the whitelist pattern:
```gitignore
!**/examples/**/*.txt
!**/examples/**/*.csv
```

---

### **Option 2: Use `demo/` Directory**

```bash
# Example structure:
UI_qtrap/react-agent/demo/
├── demo_data.txt
└── demo_analysis.csv
```

**These files WILL be tracked** because of:
```gitignore
!**/demo/**/*.txt
!**/demo/**/*.csv
```

---

### **Option 3: Use `sample_data/` Directory**

```bash
# Example structure:
notebooks/sample_data/
├── example_lipid_data.csv
└── sample_qc_results.txt
```

**These files WILL be tracked** because of:
```gitignore
!**/sample_data/**/*.txt
!**/sample_data/**/*.csv
```

---

### **Option 4: Whitelist Specific Files**

If you want to keep a specific file outside these directories, add it explicitly:

```gitignore
# In .gitignore, add:
!UI_qtrap/react-agent/src/react_agent/important_example.csv
!notebooks/key_dataset.txt
```

---

## 📝 What Gets Ignored vs. Kept

### ❌ **Ignored (Not Tracked):**
```
UI_qtrap/react-agent/src/react_agent/data/*.txt
UI_qtrap/react-agent/src/react_agent/data/*.csv
any_directory/random_data.txt
any_directory/results.csv
*.wiff files (raw MS data)
*.log files
```

### ✅ **Kept (Tracked):**
```
UI_qtrap/react-agent/examples/*.txt
UI_qtrap/react-agent/examples/*.csv
UI_qtrap/react-agent/demo/*.txt
notebooks/sample_data/*.csv
docs/**/*.txt
docs/**/*.csv
README.txt (anywhere)
requirements.txt
```

---

## 🚀 Quick Setup for Presentation

### Step 1: Create Example Data Directory

```bash
cd /home/qtrap/sciborg_dev/UI_qtrap/react-agent
mkdir -p examples
```

### Step 2: Copy Representative Files

```bash
# Copy a few small example files
cp src/react_agent/data/qc/results/example_project/QC_results.csv examples/example_qc_results.csv
cp src/react_agent/data/worklist/example_worklist.csv examples/sample_worklist.csv
```

### Step 3: Add to Git

```bash
git add examples/
git commit -m "Add example data files for presentation"
```

---

## 🔍 Testing What Will Be Ignored

### Check if a file will be ignored:
```bash
git check-ignore -v path/to/file.txt
```

### See what would be added:
```bash
git add --dry-run .
```

### List all tracked .txt/.csv files:
```bash
git ls-files | grep -E '\.(txt|csv)$'
```

---

## 💡 Best Practices

### **For Presentation:**
1. Create `examples/` directory with small, representative datasets
2. Include 1-2 example outputs from each workflow stage:
   - Example worklist (CSV)
   - Example QC results (CSV)
   - Example parsed data (TXT)
   - Example visualization data (CSV)

### **File Size Guidelines:**
- Keep example files **< 1 MB** each
- Use **< 100 rows** for CSV examples
- Show **structure**, not full datasets

### **Recommended Examples:**
```
UI_qtrap/react-agent/examples/
├── README.md                          # Explain what each file shows
├── example_worklist.csv              # Sample worklist format
├── example_qc_results.csv            # QC output format
├── example_parsed_data.txt           # Parsed MS data format
└── example_tic_data.csv              # TIC analysis format
```

---

## 🛠️ Customization

### To add more whitelisted directories:

Edit `.gitignore` and add:
```gitignore
!**/your_custom_dir/**/*.txt
!**/your_custom_dir/**/*.csv
```

### To whitelist specific file patterns:

```gitignore
# Keep all files starting with "example_"
!**/example_*.txt
!**/example_*.csv

# Keep all files in "presentation" folders
!**/presentation/**/*.txt
!**/presentation/**/*.csv
```

---

## 📊 Current Whitelist Patterns

```gitignore
# Ignore all data files
*.txt
*.csv
*.wiff
*.wiff.scan
*.dam.csv

# But keep these:
!**/examples/**/*.txt          # Any examples/ directory
!**/examples/**/*.csv
!**/demo/**/*.txt              # Any demo/ directory
!**/demo/**/*.csv
!**/sample_data/**/*.txt       # Any sample_data/ directory
!**/sample_data/**/*.csv
!README.txt                    # All README.txt files
!**/README.txt
!docs/**/*.txt                 # All files in docs/
!docs/**/*.csv
!requirements.txt              # Config files
!environment.yml
!pyproject.toml
!langgraph.json
```

---

## ⚠️ Important Notes

1. **Order matters**: Negative patterns (`!`) must come **after** the ignore pattern
2. **Directory patterns**: Use `**/` to match in any subdirectory
3. **Already tracked files**: Files already in git won't be removed by `.gitignore`
4. **Test first**: Use `git check-ignore -v` to test before committing

---

## 🎯 Summary

**Default:** All `.txt`, `.csv`, `.wiff` files are ignored  
**Exception:** Files in `examples/`, `demo/`, `sample_data/`, and `docs/` are kept  
**Result:** Clean repo with representative examples for presentation
