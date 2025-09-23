import os
import shutil

# Source directory where your CSV files are located
source_directory = '/home/qtrap/Chip-based-nanoESI-MS-MS-AI-Agents/UI_2/react-agent/src/react_agent/data/worklist/generated'  # Change this to your source directory

# Destination directory where you want to copy the files
destination_directory = '/home/qtrap/Chip-based-nanoESI-MS-MS-AI-Agents/server_test'  # Update if necessary

# Ensure the destination directory exists
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Iterate over all files in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith('.csv'):  # Check if the file is a CSV
        # Create full file paths
        source_file = os.path.join(source_directory, filename)
        destination_file = os.path.join(destination_directory, filename)
        
        try:
            # Copy the CSV file to the destination
            shutil.copy(source_file, destination_file)
            print(f"Copied {filename} to {destination_directory}")
        except Exception as e:
            print(f"Failed to copy {filename}: {e}")
