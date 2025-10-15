#!/usr/bin/env python3
"""
Script to automatically integrate advanced visualization functions into Q_viz_intensity.py
"""

def integrate_functions():
    # Read the main file
    with open('Q_viz_intensity.py', 'r') as f:
        main_content = f.read()
    
    # Read the advanced functions
    with open('advanced_functions_to_insert.py', 'r') as f:
        advanced_content = f.read()
    
    # Extract just the functions (skip the header comment)
    advanced_functions = '\n'.join(advanced_content.split('\n')[5:])  # Skip first 5 lines
    
    # Find the insertion point (after generate_std_dev_report)
    insertion_marker = "# --- Main Controller ---"
    
    if insertion_marker in main_content:
        parts = main_content.split(insertion_marker)
        
        # Insert the advanced functions before the main controller
        new_content = parts[0] + advanced_functions + "\n\n" + insertion_marker + parts[1]
        
        # Write back
        with open('Q_viz_intensity.py', 'w') as f:
            f.write(new_content)
        
        print("✅ Step 1 Complete: Advanced functions inserted into Q_viz_intensity.py")
        return True
    else:
        print("❌ Error: Could not find insertion marker")
        return False

if __name__ == "__main__":
    integrate_functions()
