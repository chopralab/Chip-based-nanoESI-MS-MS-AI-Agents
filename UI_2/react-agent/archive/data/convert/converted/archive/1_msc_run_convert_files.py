#!/usr/bin/env python3 
import subprocess 
import os 
 
# Define the new Windows path in WSL format 
windows_path = "/mnt/c/Users/iyer95/OneDrive - purdue.edu/Desktop/MSConvert" 
 
# Change to the Windows directory 
os.chdir(windows_path) 
 
# Execute the batch file using cmd.exe 
print("Running conversion script...") 
subprocess.run(["cmd.exe", "/c", "convert_files.bat"], check=True) 
print("Conversion complete!")