#!/usr/bin/env python3
"""
copy_blank_files_advanced.py - Advanced BLANK file copying with multiple options

This script provides advanced options for copying and renaming BLANK files:
- Support for different project names
- Option to copy to different directory
- Support for different file patterns
- Batch processing for multiple projects

Usage Examples:
    # Basic usage
    python copy_blank_files_advanced.py --source-dir "/path/to/data" --project-name "solventmatrix"
    
    # Copy to different directory
    python copy_blank_files_advanced.py --source-dir "/path/to/data" --target-dir "/path/to/output" --project-name "solventmatrix"
    
    # Use different pattern
    python copy_blank_files_advanced.py --source-dir "/path/to/data" --project-name "solventmatrix" --pattern "BLANK_*"
    
    # Process multiple projects
    python copy_blank_files_advanced.py --source-dir "/path/to/data" --project-names "solventmatrix,project2,project3"
"""

import os
import shutil
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import json

def find_files_by_pattern(source_dir: str, pattern: str) -> List[Path]:
    """Find all files matching the given pattern in the source directory."""
    source_path = Path(source_dir)
    files = []
    
    for file_path in source_path.glob(pattern):
        if file_path.is_file():
            files.append(file_path)
    
    return sorted(files)

def generate_new_filename(original_filename: str, project_name: str, insertion_pattern: str = None) -> str:
    """
    Generate new filename with project name inserted.
    
    Args:
        original_filename: Original filename
        project_name: Project name to insert
        insertion_pattern: Custom regex pattern for insertion point
    
    Returns:
        New filename with project name inserted
    """
    if insertion_pattern:
        # Use custom pattern
        pattern = insertion_pattern
    else:
        # Default pattern: insert after R-# 
        pattern = r'(.*_R-\d+)(_.*)'
    
    match = re.match(pattern, original_filename)
    
    if match:
        prefix = match.group(1)
        suffix = match.group(2) if len(match.groups()) > 1 else ""
        new_filename = f"{prefix}_Proj-{project_name}{suffix}"
        return new_filename
    else:
        # Fallback: add project name before extension
        name_parts = original_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            return f"{name_parts[0]}_Proj-{project_name}.{name_parts[1]}"
        else:
            return f"{original_filename}_Proj-{project_name}"

def copy_and_rename_files(source_dir: str, target_dir: str, project_name: str, 
                         pattern: str = "BLANK_LC-*", dry_run: bool = False,
                         insertion_pattern: str = None) -> List[Tuple[str, str]]:
    """
    Copy files and rename them with project name.
    
    Args:
        source_dir: Source directory
        target_dir: Target directory (can be same as source)
        project_name: Project name to add
        pattern: File pattern to match
        dry_run: If True, don't actually copy files
        insertion_pattern: Custom regex pattern for project name insertion
    
    Returns:
        List of (original_path, new_path) tuples
    """
    files = find_files_by_pattern(source_dir, pattern)
    copied_files = []
    
    print(f"Found {len(files)} files matching pattern '{pattern}'")
    
    # Ensure target directory exists
    target_path = Path(target_dir)
    if not dry_run:
        target_path.mkdir(parents=True, exist_ok=True)
    
    for file_path in files:
        original_filename = file_path.name
        new_filename = generate_new_filename(original_filename, project_name, insertion_pattern)
        new_path = target_path / new_filename
        
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

def process_multiple_projects(source_dir: str, target_dir: str, project_names: List[str],
                            pattern: str = "BLANK_LC-*", dry_run: bool = False) -> Dict[str, List[Tuple[str, str]]]:
    """Process multiple projects in batch."""
    results = {}
    
    for project_name in project_names:
        print(f"\n{'='*60}")
        print(f"Processing project: {project_name}")
        print(f"{'='*60}")
        
        # Create project-specific target directory
        project_target_dir = Path(target_dir) / project_name if target_dir != source_dir else target_dir
        
        copied_files = copy_and_rename_files(
            source_dir, str(project_target_dir), project_name, pattern, dry_run
        )
        results[project_name] = copied_files
    
    return results

def save_results_log(results: Dict[str, List[Tuple[str, str]]], log_file: str):
    """Save processing results to a log file."""
    log_data = {
        'timestamp': str(Path().cwd()),
        'results': results
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"📝 Results saved to: {log_file}")

def main():
    parser = argparse.ArgumentParser(description="Advanced BLANK file copying with project names")
    parser.add_argument("--source-dir", required=True, help="Source directory containing files")
    parser.add_argument("--target-dir", help="Target directory (default: same as source)")
    parser.add_argument("--project-name", help="Single project name to add")
    parser.add_argument("--project-names", help="Comma-separated list of project names for batch processing")
    parser.add_argument("--pattern", default="BLANK_LC-*", help="File pattern to match (default: BLANK_LC-*)")
    parser.add_argument("--insertion-pattern", help="Custom regex pattern for project name insertion")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually copying")
    parser.add_argument("--log-file", help="Save results to log file")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.project_name and not args.project_names:
        print("❌ Must specify either --project-name or --project-names")
        return 1
    
    if args.project_name and args.project_names:
        print("❌ Cannot specify both --project-name and --project-names")
        return 1
    
    # Validate source directory
    source_path = Path(args.source_dir)
    if not source_path.exists():
        print(f"❌ Source directory does not exist: {args.source_dir}")
        return 1
    
    if not source_path.is_dir():
        print(f"❌ Source path is not a directory: {args.source_dir}")
        return 1
    
    # Set target directory
    target_dir = args.target_dir if args.target_dir else args.source_dir
    
    print(f"🔍 Source directory: {args.source_dir}")
    print(f"📁 Target directory: {target_dir}")
    print(f"🔍 File pattern: {args.pattern}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be copied")
    
    # Process files
    if args.project_name:
        # Single project
        print(f"📝 Project name: {args.project_name}")
        print("=" * 60)
        
        copied_files = copy_and_rename_files(
            args.source_dir, target_dir, args.project_name, 
            args.pattern, args.dry_run, args.insertion_pattern
        )
        
        results = {args.project_name: copied_files}
    else:
        # Multiple projects
        project_names = [name.strip() for name in args.project_names.split(',')]
        print(f"📝 Project names: {', '.join(project_names)}")
        
        results = process_multiple_projects(
            args.source_dir, target_dir, project_names, args.pattern, args.dry_run
        )
    
    # Summary
    total_files = sum(len(files) for files in results.values())
    print("=" * 60)
    print(f"📊 Summary:")
    for project_name, files in results.items():
        print(f"   {project_name}: {len(files)} files")
    print(f"   Total files processed: {total_files}")
    
    if not args.dry_run:
        print(f"✅ All files copied successfully!")
    else:
        print(f"🔍 Run without --dry-run to actually copy the files")
    
    # Save log if requested
    if args.log_file:
        save_results_log(results, args.log_file)
    
    return 0

if __name__ == "__main__":
    exit(main())
