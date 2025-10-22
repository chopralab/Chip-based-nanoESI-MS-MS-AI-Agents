# Git Repository - Large Files Report
**Generated:** October 22, 2025  
**Repository:** /home/qtrap/sciborg_dev  
**Branch:** paper

---

## Repository Size Summary

- **Total .git directory size:** 249 MB
- **Git object count:** 2,019 objects
- **Size in pack files:** 103.44 MiB
- **Loose objects:** 142.91 MiB

---

## Largest Files Currently Tracked (Top 20)

### FAISS Indexes (Large ML/Vector Indexes)
1. **26.6 MB** - `faiss_index_qtrap_nano/index.faiss`
2. **15.7 MB** - `notebooks/papers/faiss_index/index.faiss`
3. **3.8 MB** - `faiss_index_qtrap_nano/index.pkl`
4. **2.6 MB** - `notebooks/papers/faiss_index/index.pkl`

**Total FAISS files: ~48 MB**

### PDF Papers (Duplicated Across Multiple Directories)
Each of these papers appears in 4-5 locations:
- `notebooks/papers/qtrap_nano/`
- `notebooks/papers/all/`
- `notebooks/papers/Nanomate/`
- `notebooks/papers/QTRAP/`
- `UI_qtrap/react-agent/archive/data/helper/papers/`

**Duplicate PDFs:**
- **8.7 MB** × 5 copies - Hall_et_al-2016-Cancer_Research-AM.pdf = **43.5 MB**
- **7.7 MB** × 4 copies - 4000-api-hardware-guide.pdf = **30.8 MB**
- **4.5 MB** × 4 copies - elife-63252-v2.pdf = **18 MB**
- **4.2 MB** × 4 copies - sciadv.aay3240.pdf = **16.8 MB**
- **4.2 MB** × 4 copies - PIIS2589004220308956.pdf = **16.8 MB**
- **4.1 MB** × 4 copies - 1-s2.0-S0009308419300519-main.pdf = **16.4 MB**
- **4.0 MB** × 5 copies - almeida-et-al-2015... .pdf = **20 MB**
- **3.7 MB** × 4 copies - randall-et-al-2014... .pdf = **14.8 MB**
- **3.5 MB** × 4 copies - 1-s2.0-S0022227521000304-main.pdf = **14 MB**
- **3.1 MB** × 4 copies - stegemann-et-al-2011... .pdf = **12.4 MB**
- **3.0 MB** × 4 copies - JCB_200901145.pdf = **12 MB**

**Total duplicate PDFs: ~215 MB** (could be reduced to ~50 MB if deduplicated)

---

## Problem Areas

### 1. **FAISS Indexes (~48 MB)**
- These are large binary ML/vector index files
- Currently tracked by git
- Should be in `.gitignore` (already added in new .gitignore)
- **Action:** Remove from git tracking

### 2. **Duplicate PDF Papers (~215 MB)**
- Same papers stored in 4-5 different locations
- Massive duplication in git history
- **Recommendation:** Keep only in `notebooks/papers/` and remove duplicates

### 3. **Archive Directory**
- `UI_qtrap/react-agent/archive/data/helper/papers/` contains duplicate PDFs
- Already in `.gitignore` but still tracked in git history
- **Action:** Remove from git tracking

---

## Recommended Actions

### Immediate Actions (Remove from Git Tracking)

```bash
# Remove FAISS indexes from git (keep on disk)
git rm --cached faiss_index_qtrap_nano/index.faiss
git rm --cached faiss_index_qtrap_nano/index.pkl
git rm --cached faiss_index_blanksby_FA/index.faiss
git rm --cached faiss_index_blanksby_FA/index.pkl
git rm --cached notebooks/papers/faiss_index/index.faiss
git rm --cached notebooks/papers/faiss_index/index.pkl

# Remove embeddings from git (keep on disk)
git rm --cached -r embeddings/

# Remove archive from git (keep on disk)
git rm --cached -r archive/

# Remove data directory from git (keep on disk)
git rm --cached -r UI_qtrap/react-agent/src/react_agent/data/

# Commit the changes
git commit -m "Remove large files and generated data from git tracking"
```

### Optional: Clean Git History (Advanced)

If you want to completely remove these files from git history to reduce repository size:

```bash
# Use git filter-repo (recommended) or BFG Repo-Cleaner
# WARNING: This rewrites history - coordinate with team first!

# Install git-filter-repo
pip install git-filter-repo

# Remove large files from history
git filter-repo --path faiss_index_qtrap_nano/ --invert-paths
git filter-repo --path faiss_index_blanksby_FA/ --invert-paths
git filter-repo --path embeddings/ --invert-paths
git filter-repo --path archive/ --invert-paths
git filter-repo --path UI_qtrap/react-agent/src/react_agent/data/ --invert-paths
```

### Deduplicate PDFs

Keep papers only in `notebooks/papers/` and remove from:
- `UI_qtrap/react-agent/archive/data/helper/papers/` (already archived)
- Consider consolidating `notebooks/papers/qtrap_nano/`, `all/`, `Nanomate/`, `QTRAP/` subdirectories

---

## Impact of Cleanup

### Current State:
- Git repo: 249 MB
- Many duplicate files tracked

### After Removing from Tracking:
- Will prevent future growth
- Existing files stay on disk
- New .gitignore prevents re-adding

### After History Cleanup (Optional):
- Could reduce git repo to ~50-100 MB
- Requires force push (coordinate with team)
- One-time operation

---

## Files Protected by New .gitignore

The new `.gitignore` file now protects:
- ✅ `faiss_index_*/`
- ✅ `embeddings/`
- ✅ `archive/`
- ✅ `UI_qtrap/react-agent/src/react_agent/data/`
- ✅ `ai/`, `core/`, `server_data/`, `utils/`

These won't be tracked in future commits.

---

## Next Steps

1. **Review this report** and decide which actions to take
2. **Remove from tracking** (safe, keeps files on disk)
3. **Optionally clean history** (advanced, requires coordination)
4. **Deduplicate PDFs** if needed for presentation

**Note:** The `.gitignore` is already in place and working. The files listed above are already tracked in git history, so they need to be explicitly removed.
