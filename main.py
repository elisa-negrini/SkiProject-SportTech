import subprocess
import sys
import os
import glob
from pathlib import Path  # Added for Path
import json
import re

# --- Path Configuration ---

# Scripts in the project root
EXTRACT_SCRIPT = "extract_annotations.py"
INTERPOLATE_SCRIPT = "interpolation.py"

# Scripts in the subfolder (Assuming 'visualize_annotations' based on context)
VISUALIZE_DIR = "visualize_annotations"
VISUALIZE_SCRIPT = os.path.join(VISUALIZE_DIR, "visualize_interpolation.py")
CREATE_VIDEO_SCRIPT = os.path.join(VISUALIZE_DIR, "create_video.py")

# Fixed base path for the 'frames' directory (DEPENDS ON filter_boxes)
ROOT_FRAMES_PATH = Path(r"dataset\frames")
FILE_EXT = "*.jpg"  # DEPENDS ON filter_boxes

# --- Support Functions for BOXES FILTERING ---

def get_indices_from_dir(directory, file_extension="*.jpg"):
    """ Retrieves frame numbers from a directory using glob and name extraction. """
    indices = set()
    try:
        pattern = os.path.join(directory, file_extension)
        file_paths = glob.glob(pattern)
        for file_path in file_paths:
            full_file_name = os.path.basename(file_path)
            # Assume the name is just the number without extension
            name_without_extension, _ = os.path.splitext(full_file_name)
            # Use int(name_without_extension) for sorting, but keep it as a string here
            indices.add(name_without_extension)
        return indices
    except Exception as e:
        # In this context, an error here is critical
        print(f"⚠️ Warning while retrieving indices from {directory}: {e}")
        return indices

def filtra_boxes_txt_local(jump_number):
    """
    Executes the boxes.txt filtering logic, adapted for local use in main_annotation.py.
    Returns True if filtering was successful, False otherwise.
    """
    JUMP_ID = f"JP{jump_number:04d}"
    BASE_PATH = ROOT_FRAMES_PATH / JUMP_ID
    
    if not BASE_PATH.exists():
        print(f"--- ❌ Error: The frames directory for jump {JUMP_ID} does not exist: {BASE_PATH} ---")
        return False

    # --- Path Definition (Local) ---
    main_dir = BASE_PATH
    dir_removed = BASE_PATH / 'removed'
    dir_occluded = BASE_PATH / 'occluded'
    
    input_file_path = main_dir / 'boxes.txt'
    output_file_path = main_dir / 'boxes_filtered.txt'
    
    if not input_file_path.exists():
        print(f"--- ❌ Error: boxes.txt file not found at path: {input_file_path} ---")
        return False

    # 1. Calculate Indices and Skip Positions
    main_indices = get_indices_from_dir(str(main_dir), FILE_EXT)
    removed_indices = get_indices_from_dir(str(dir_removed), FILE_EXT)
    occluded_indices = get_indices_from_dir(str(dir_occluded), FILE_EXT)
    
    all_indices_set = main_indices.union(removed_indices).union(occluded_indices)
    # Sort the indices as integers to determine the correct line number in boxes.txt
    all_indices_sorted = sorted(list(all_indices_set), key=int)
    # Indices corresponding to frames that should be removed from boxes.txt
    indices_to_remove = all_indices_set.difference(main_indices)
    
    # Calculate 1-based line numbers to skip
    positions_to_skip = set()
    for position_0based, index in enumerate(all_indices_sorted):
        if index in indices_to_remove:
            positions_to_skip.add(position_0based + 1)
            
    # Print key details
    print(f"   Total indices found: {len(all_indices_set)}")
    print(f"   Lines to skip (corresponding to frames not in 'main'): {len(positions_to_skip)}")

    # 2. Read the input file and Write the filtered file
    lines_kept_count = 0
    lines_skipped_count = 0
    
    try:
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line_number, line in enumerate(infile, 1):
                if line_number not in positions_to_skip:
                    outfile.write(line)
                    lines_kept_count += 1
                else:
                    lines_skipped_count += 1
                    
        print(f"   ✅ Filtering completed. Lines kept: {lines_kept_count}, Removed: {lines_skipped_count}.")
        return True

    except Exception as e:
        print(f"--- ❌ Error during boxes.txt filtering: {e} ---")
        return False


# --- Support Functions for Execution (Unchanged) ---

def run_script(script_path, jump_number):
    """
    Executes an external Python script, regardless of its location,
    passing the jump number as input.
    """
    
    print(f"\n--- Executing: {script_path} for Jump {jump_number} ---")
    
    if not os.path.exists(script_path):
        print(f"--- ❌ Error: Script not found at path: {script_path} ---")
        return False
        
    try:
        # Pass jump_number via stdin for scripts that expect it
        result = subprocess.run(
            [sys.executable, script_path],
            input=str(jump_number).encode('utf-8'),
            capture_output=False, 
            check=True,          
            text=False           
        )
        print(f"--- ✅ {script_path} completed successfully. ---")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"--- ❌ Error in {script_path} ---")
        print(f"The process returned an error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"--- ❌ Unknown error during execution of {script_path}: {e} ---")
        return False


def get_jump_number():
    """Asks the user for the jump number to analyze."""
    while True:
        try:
            jump_number = int(input("\nWhich JUMP do you want to analyze (e.g., 6)? "))
            if jump_number > 0:
                return jump_number
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_yes_no_input(prompt):
    """Asks the user for a Y/N response."""
    while True:
        response = input(f"{prompt} (y/n): ").lower().strip()
        if response in ['y', 'yes', 'n', 'no']:
            return response.startswith('y')
        else:
            print("Invalid response. Please enter 'y' or 'n'.")

# --- UPDATED Main Workflow ---

def main_workflow():
    """The main program workflow."""
    
    print("\n" + "="*50)
    print("🎬 SKI JUMP ANALYSIS WORKFLOW")
    print("="*50)

    while True:
        # 1. Get the Jump Number
        jump_number = get_jump_number()
        
        # 2. Execute Extract Annotations (Root)
        print("\nStep 1/5: Extracting Annotations...")
        if not run_script(EXTRACT_SCRIPT, jump_number):
            if get_yes_no_input("Extraction failed. Do you want to try again/analyze another jump?"):
                continue
            else:
                break
        
        # 3. Execute Filter Boxes (New Local Step)
        print("\nStep 2/5: Bounding Boxes Filtering (boxes.txt -> boxes_filtered.txt)...")
        if not filtra_boxes_txt_local(jump_number):
            print("--- ❌ Boxes Filtering failed. Stopping the workflow. ---")
            break
        
        # 4. Ask for Interpolation (Root)
        do_interpolate = get_yes_no_input("Do you want to INTERPOLATE missing frames?")
        if do_interpolate:
            print("\nStep 3/5: Keypoints Interpolation...")
            if not run_script(INTERPOLATE_SCRIPT, jump_number):
                print("⚠️ Warning: Interpolation failed. Proceeding with visualization/video only for extracted frames.")
            else:
                 print("Interpolation complete. The interpolated COCO file is now available.")
        else:
            print("Interpolation skipped.")

        # 5. Ask for Visualization (Subfolder)
        do_visualize = get_yes_no_input("Do you want to VISUALIZE and create the Video?")
        
        if do_visualize:
            # 5a. Execute Visualization (Subfolder)
            print("\nStep 4/5: Creating Visualizations...")
            if not run_script(VISUALIZE_SCRIPT, jump_number):
                print("❌ Visualization failed. Cannot proceed to video creation.")
            else:
                # 5b. Execute Video Creation (Subfolder)
                print("\nStep 5/5: Creating Video...")
                run_script(CREATE_VIDEO_SCRIPT, jump_number)
        else:
            print("Visualization and video creation skipped.")

        # 6. Ask whether to continue
        if not get_yes_no_input("\nAnalysis completed. Do you want to analyze another jump?"):
            break

    print("\n" + "="*50)
    print("Program terminated. Goodbye!")
    print("="*50)


if __name__ == "__main__":
    main_workflow()