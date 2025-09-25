#!/usr/bin/env python3
"""
copy_blank_files.py - Copy BLANK files and add project name

This script finds BLANK_LC-* files, copies them, and renames the copies
to include the project name (Proj-solventmatrix) after the R-# pattern.

Usage:
    python copy_blank_files.py --source-dir "/mnt/d_drive/Analyst Data/Projects/API Instrument/Data" --project-name "solventmatrix"
"""

import os
import shutil
import re
import argparse
from pathlib import Path
from typing import List, Tuple

def find_blank_files(source_dir: str) -> List[Path]:
    """Find all BLANK_LC-* files in the source directory."""
    source_path = Path(source_dir)
    blank_files = []
    
    # Find all files matching BLANK_LC-* pattern
    for file_path in source_path.glob("BLANK_LC-*"):
        blank_files.append(file_path)
    
    return sorted(blank_files)

def generate_new_filename(original_filename: str, project_name: str) -> str:
    """
    Generate new filename with project name inserted after R-# pattern.
    
    Example:
    BLANK_LC-PC_R-1_PC_withSPLASH.wiff
    -> BLANK_LC-PC_R-1_Proj-solventmatrix_PC_withSPLASH.wiff
    """
    # Pattern to match R-# and capture the parts
    pattern = r'(BLANK_LC-[^_]+_R-\d+)(_.*)'
    match = re.match(pattern, original_filename)
    
    if match:
        prefix = match.group(1)  # BLANK_LC-PC_R-1
        suffix = match.group(2)  # _PC_withSPLASH.wiff
        new_filename = f"{prefix}_Proj-{project_name}{suffix}"
        return new_filename
    else:
        # Fallback: just add project name before the extension
        name_parts = original_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            return f"{name_parts[0]}_Proj-{project_name}.{name_parts[1]}"
        else:
            return f"{original_filename}_Proj-{project_name}"

def copy_and_rename_files(source_dir: str, project_name: str, dry_run: bool = False) -> List[Tuple[str, str]]:
    """
    Copy BLANK files and rename them with project name.
    
    Returns:
        List of (original_path, new_path) tuples
    """
    blank_files = find_blank_files(source_dir)
    copied_files = []
    
    print(f"Found {len(blank_files)} BLANK_LC-* files")
    
    for file_path in blank_files:
        original_filename = file_path.name
        new_filename = generate_new_filename(original_filename, project_name)
        new_path = file_path.parent / new_filename
        
        print(f"  {original_filename}")
        print(f"  -> {new_filename}")
        
        if not dry_run:
            try:
                shutil.copy2(file_path, new_path)
                copied_files.append((str(file_path), str(new_path)))
                print(f"  ✅ Copied successfully")
            except Exception as e:
                print(f"  ❌ Error copying: {e}")
        else:
            print(f"  🔍 DRY RUN - would copy")
            copied_files.append((str(file_path), str(new_path)))
        
        print()
    
    return copied_files

def main():
    parser = argparse.ArgumentParser(description="Copy BLANK files and add project name")
    parser.add_argument("--source-dir", required=True, help="Source directory containing BLANK files")
    parser.add_argument("--project-name", required=True, help="Project name to add (e.g., 'solventmatrix')")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually copying")
    
    args = parser.parse_args()
    
    # Validate source directory
    source_path = Path(args.source_dir)
    if not source_path.exists():
        print(f"❌ Source directory does not exist: {args.source_dir}")
        return 1
    
    if not source_path.is_dir():
        print(f"❌ Source path is not a directory: {args.source_dir}")
        return 1
    
    print(f"🔍 Searching for BLANK_LC-* files in: {args.source_dir}")
    print(f"📝 Project name: {args.project_name}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be copied")
    
    print("=" * 60)
    
    copied_files = copy_and_rename_files(args.source_dir, args.project_name, args.dry_run)
    
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   Files processed: {len(copied_files)}")
    
    if not args.dry_run:
        print(f"   Files copied successfully: {len(copied_files)}")
        print(f"✅ All BLANK files copied with project name added!")
    else:
        print(f"   Files that would be copied: {len(copied_files)}")
        print(f"🔍 Run without --dry-run to actually copy the files")
    
    return 0

if __name__ == "__main__":
    exit(main())
