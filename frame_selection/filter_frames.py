import os
import shutil
import glob
from typing import List, Set

# --- CONFIGURATION: ONLY MODIFY THESE BASE PATHS AND THE NUMBER RANGE ---

# Main path of the dataset up to SkiTB/JP/
# ALL JPXXXX folders are located here.
BASE_DATASET_PATH = r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiTB dataset\SkiTB\JP"

# The main folder that will contain all results (e.g., JP0001, JP0002, ...)
# It will be created inside BASE_DATASET_PATH.
DESTINATION_FOLDER_NAME = "filtered_frames" 

# Selection interval: 1 file every STEP_SIZE
STEP_SIZE = 10

# Range of JPXXXX folders to process (from 1 to 100)
START_NUM = 8
END_NUM = 20

# -----------------------------------------------------------------------------------


def generate_folder_names(start: int, end: int) -> List[str]:
    """Generates a list of folder names in the JPXXXX format (e.g., JP0001 to JP0100)."""
    return [f"JP{i:04d}" for i in range(start, end + 1)]


def read_camera_ids(cameras_file_path: str) -> List[int]:
    """
    Reads the cameras.txt file and returns a list of camera IDs.
    """
    try:
        with open(cameras_file_path, 'r') as f:
            camera_ids = [int(line.strip()) for line in f if line.strip()]
        return camera_ids
    except FileNotFoundError:
        print(f"   ⚠️ cameras.txt not found at {cameras_file_path}")
        return []
    except Exception as e:
        print(f"   ❌ Error reading cameras.txt: {e}")
        return []


def detect_camera_transitions(camera_ids: List[int]) -> Set[int]:
    """
    Detects camera transitions and returns a set of frame indices that must be included:
    - Last frame of each camera segment
    - First frame of each camera segment (except the very first)
    """
    transition_indices = set()
    
    for i in range(1, len(camera_ids)):
        if camera_ids[i] != camera_ids[i-1]:
            # Camera change detected
            transition_indices.add(i-1)  # Last frame of previous camera
            transition_indices.add(i)    # First frame of new camera
    
    return transition_indices


def select_and_copy_files(source_directory: str, destination_directory: str, 
                          step_size: int, camera_ids: List[int]):
    """
    Selects and copies JPG files based on:
    1. Regular sampling: one file every 'step_size'
    2. Camera transitions: always include first and last frame of each camera
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
    
    # Check if number of files matches number of camera IDs
    if camera_ids and len(all_files) != len(camera_ids):
        print(f"   ⚠️ Warning: {len(all_files)} files found but {len(camera_ids)} camera IDs in cameras.txt")
    
    # Detect camera transitions
    transition_indices = detect_camera_transitions(camera_ids) if camera_ids else set()
    
    # Create set of indices to copy
    indices_to_copy = set()
    
    # Add regular sampling indices (every step_size)
    for i in range(0, len(all_files), step_size):
        indices_to_copy.add(i)
    
    # Add transition indices
    indices_to_copy.update(transition_indices)
    
    # Sort indices for ordered copying
    indices_to_copy = sorted(indices_to_copy)
    
    files_copied = 0
    transition_frames = 0

    # Copy selected files
    for i in indices_to_copy:
        if i >= len(all_files):
            continue
            
        source_path = all_files[i]
        file_name = os.path.basename(source_path)
        destination_path = os.path.join(destination_directory, file_name)

        try:
            # Copy the file, preserving metadata (copy2)
            shutil.copy2(source_path, destination_path)
            files_copied += 1
            if i in transition_indices:
                transition_frames += 1
        except Exception as e:
            print(f"   ❌ Error while copying file {file_name}: {e}")

    print(f"   ✨ Copied {files_copied} files total")
    print(f"      - Regular sampling (1 every {step_size}): {files_copied - transition_frames} files")
    print(f"      - Camera transitions: {transition_frames} additional files")


# --- Main Logic ---
def run_dataset_selection():
    
    # 1. Define root paths
    parent_destination_path = os.path.join(BASE_DATASET_PATH, DESTINATION_FOLDER_NAME)
    
    # 2. Generate the list of folders to process
    folders_to_process = generate_folder_names(START_NUM, END_NUM)
    
    print(f"*** Starting processing of {len(folders_to_process)} folders ***")
    print(f"Parent Destination Directory: {parent_destination_path}")
    print(f"Selection Step: 1 every {STEP_SIZE}")
    print(f"Camera transitions: Always included")

    # Create the parent destination folder if it doesn't exist
    if not os.path.exists(parent_destination_path):
        os.makedirs(parent_destination_path)
        print(f"✅ Created parent destination directory: {DESTINATION_FOLDER_NAME}")

    # 3. Iteration and Processing
    for folder_name in folders_to_process:
        print(f"\n--- Processing folder: **{folder_name}** ---")
        
        # Construct Source path: .../JP/JPXXXX/frames
        source = os.path.join(BASE_DATASET_PATH, folder_name, "frames")
        
        # Construct cameras.txt path: .../JP/JPXXXX/cameras.txt
        cameras_file = os.path.join(BASE_DATASET_PATH, folder_name, "MC", "cameras.txt")
        
        # Construct Destination path: .../JP/filtered_frames/JPXXXX
        destination = os.path.join(parent_destination_path, folder_name)

        # Safety check: verify that the source folder exists
        if not os.path.isdir(source):
            print(f"❌ Source folder '{source}' not found. Skipping.")
            continue

        # Read camera IDs
        camera_ids = read_camera_ids(cameras_file)
        
        # Call the copy function
        select_and_copy_files(source, destination, STEP_SIZE, camera_ids)

    print("\n*** Processing complete! ✨ ***")

# --- EXECUTION ---
# Make sure you have verified BASE_DATASET_PATH, DESTINATION_FOLDER_NAME, and the numeric ranges.
if __name__ == "__main__":
    run_dataset_selection()