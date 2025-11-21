import os
import shutil
import glob
from typing import List

# --- CONFIGURATION: ONLY MODIFY THESE BASE PATHS AND THE NUMBER RANGE ---

# Main path of the dataset up to SkiTB/JP/
# ALL JPXXXX folders are located here.
BASE_DATASET_PATH = r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiTB dataset\SkiTB\JP"

# The main folder that will contain all results (e.g., JP0001, JP0002, ...)
# It will be created inside BASE_DATASET_PATH.
DESTINATION_FOLDER_NAME = "outof_five" 

# Selection interval: 1 file every STEP_SIZE
STEP_SIZE = 5 

# Range of JPXXXX folders to process (from 1 to 100)
START_NUM = 1
END_NUM = 100

# -----------------------------------------------------------------------------------


def generate_folder_names(start: int, end: int) -> List[str]:
    """Generates a list of folder names in the JPXXXX format (e.g., JP0001 to JP0100)."""
    return [f"JP{i:04d}" for i in range(start, end + 1)]

def select_and_copy_files(source_directory: str, destination_directory: str, step_size: int):
    """
    Selects and copies one JPG file every 'step_size' from the source directory 
    to the destination directory.
    """
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Get a list of all JPG/jpeg files in the source
    search_path_jpg = os.path.join(source_directory, '*.jpg')
    search_path_jpeg = os.path.join(source_directory, '*.jpeg')

    # Use glob to find full paths and sort the list for consistent selection
    all_files = sorted(glob.glob(search_path_jpg) + glob.glob(search_path_jpeg))
    
    if not all_files:
        print(f"   ⚠️ No .jpg or .jpeg files found in {os.path.basename(source_directory)}")
        return
    
    files_copied = 0

    # Iterate through the list and select one file every 'step_size'
    for i in range(0, len(all_files), step_size):
        source_path = all_files[i]
        file_name = os.path.basename(source_path)
        destination_path = os.path.join(destination_directory, file_name)

        try:
            # Copy the file, preserving metadata (copy2)
            shutil.copy2(source_path, destination_path)
            files_copied += 1
        except Exception as e:
            print(f"   ❌ Error while copying file {file_name}: {e}")

    print(f"   ✨ Copied {files_copied} files (1 every {step_size}).")

# --- Main Logic ---
def run_dataset_selection():
    
    # 1. Define root paths
    parent_destination_path = os.path.join(BASE_DATASET_PATH, DESTINATION_FOLDER_NAME)
    
    # 2. Generate the list of folders to process
    folders_to_process = generate_folder_names(START_NUM, END_NUM)
    
    print(f"*** Starting processing of {len(folders_to_process)} folders ***")
    print(f"Parent Destination Directory: {parent_destination_path}")
    print(f"Selection Step: 1 every {STEP_SIZE}")

    # Create the parent destination folder if it doesn't exist
    if not os.path.exists(parent_destination_path):
        os.makedirs(parent_destination_path)
        print(f"✅ Created parent destination directory: {DESTINATION_FOLDER_NAME}")

    # 3. Iteration and Processing
    for folder_name in folders_to_process:
        print(f"\n--- Processing folder: **{folder_name}** ---")
        
        # Construct Source path: .../JP/JPXXXX/frames
        source = os.path.join(BASE_DATASET_PATH, folder_name, "frames")
        
        # Construct Destination path: .../JP/outof_five/JPXXXX
        destination = os.path.join(parent_destination_path, folder_name)

        # Safety check: verify that the source folder exists
        if not os.path.isdir(source):
            print(f"❌ Source folder '{source}' not found. Skipping.")
            continue

        # Call the copy function
        select_and_copy_files(source, destination, STEP_SIZE)

    print("\n*** Processing complete! ✨ ***")

# --- EXECUTION ---
# Make sure you have verified BASE_DATASET_PATH, DESTINATION_FOLDER_NAME, and the numeric ranges.
run_dataset_selection()