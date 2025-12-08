import subprocess
import sys
import os
import glob
from pathlib import Path
import json
import re

# --- Path Configuration ---

# Scripts in the project root
EXTRACT_SCRIPT = "extract_annotations.py"
INTERPOLATE_SCRIPT = "interpolation.py"

# Scripts in the subfolder
VISUALIZE_DIR = "visualize_annotations"
VISUALIZE_SCRIPT = os.path.join(VISUALIZE_DIR, "visualize_interpolation.py")
CREATE_VIDEO_SCRIPT = os.path.join(VISUALIZE_DIR, "create_video.py")

# Fixed base path for the 'frames' directory
ROOT_FRAMES_PATH = Path(r"dataset\frames")
FILE_EXT = "*.jpg"

# --- Support Functions for BOXES FILTERING ---

def get_indices_from_dir(directory, file_extension="*.jpg"):
    """ Retrieves frame numbers from a directory using glob and name extraction. """
    indices = set()
    try:
        pattern = os.path.join(directory, file_extension)
        file_paths = glob.glob(pattern)
        for file_path in file_paths:
            full_file_name = os.path.basename(file_path)
            name_without_extension, _ = os.path.splitext(full_file_name)
            indices.add(name_without_extension)
        return indices
    except Exception as e:
        print(f"⚠️ Warning while retrieving indices from {directory}: {e}")
        return indices

def filtra_boxes_txt_local(jump_number):
    """
    Executes the boxes.txt filtering logic for a specific jump.
    """
    JUMP_ID = f"JP{jump_number:04d}"
    BASE_PATH = ROOT_FRAMES_PATH / JUMP_ID
    
    if not BASE_PATH.exists():
        print(f"--- ❌ Error: The frames directory for jump {JUMP_ID} does not exist: {BASE_PATH} ---")
        return False

    # --- Path Definition ---
    main_dir = BASE_PATH
    dir_removed = BASE_PATH / 'removed'
    dir_occluded = BASE_PATH / 'occluded'
    
    input_file_path = main_dir / 'boxes.txt'
    output_file_path = main_dir / 'boxes_filtered.txt'
    
    if not input_file_path.exists():
        print(f"--- ❌ Error: boxes.txt file not found at path: {input_file_path} ---")
        return False

    # 1. Calculate Indices
    main_indices = get_indices_from_dir(str(main_dir), FILE_EXT)
    removed_indices = get_indices_from_dir(str(dir_removed), FILE_EXT)
    occluded_indices = get_indices_from_dir(str(dir_occluded), FILE_EXT)
    
    all_indices_set = main_indices.union(removed_indices).union(occluded_indices)
    all_indices_sorted = sorted(list(all_indices_set), key=int)
    indices_to_remove = all_indices_set.difference(main_indices)
    
    # Calculate skip positions
    positions_to_skip = set()
    for position_0based, index in enumerate(all_indices_sorted):
        if index in indices_to_remove:
            positions_to_skip.add(position_0based + 1)
            
    print(f"   Total indices found: {len(all_indices_set)}")
    print(f"   Lines to skip: {len(positions_to_skip)}")

    # 2. Write filtered file
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


# --- Support Functions for Execution ---

def run_script(script_path, jump_number):
    """
    Executes an external Python script passing the jump number as input via stdin.
    """
    print(f"\n   -> Executing: {os.path.basename(script_path)} ...")
    
    if not os.path.exists(script_path):
        print(f"   ❌ Error: Script not found at path: {script_path}")
        return False
        
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            input=str(jump_number).encode('utf-8'),
            capture_output=False, 
            check=True,          
            text=False           
        )
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error in {script_path} (Code: {e.returncode})")
        return False
    except Exception as e:
        print(f"   ❌ Unknown error: {e}")
        return False


def get_jumps_input():
    """
    Parses user input for a range (1-5) or single number (6).
    Returns a list of integers.
    """
    while True:
        user_input = input("\nEnter Jump range (e.g., '1-5' for range, or '6' for single): ").strip()
        try:
            if '-' in user_input:
                parts = user_input.split('-')
                if len(parts) == 2:
                    start, end = map(int, parts)
                    if start > end:
                        print("Start number must be less than end number.")
                        continue
                    return list(range(start, end + 1))
            else:
                return [int(user_input)]
        except ValueError:
            print("Invalid format. Please use 'Start-End' (e.g., 1-5) or a single number.")

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
    """The main program workflow handling ranges."""
    
    print("\n" + "="*50)
    print("🎬 SKI JUMP ANALYSIS WORKFLOW (BATCH MODE)")
    print("="*50)

    # 1. Get the list of Jumps to process
    jumps_to_process = get_jumps_input()
    print(f"\nSelected Jumps: {jumps_to_process}")

    # 2. Ask Configuration Questions ONCE for the whole batch
    do_interpolate = get_yes_no_input("Do you want to INTERPOLATE missing frames for this batch?")
    
    do_visualize = False
    if get_yes_no_input("Do you want to VISUALIZE and create Videos for this batch?"):
        do_visualize = True

    print("\n" + "="*50)
    print("🚀 STARTING BATCH PROCESSING")
    print("="*50)

    # 3. Loop through each jump
    for i, jump_number in enumerate(jumps_to_process):
        jump_label = f"JP{jump_number:04d}"
        print(f"\n[{i+1}/{len(jumps_to_process)}] Processing {jump_label}...")
        print("-" * 30)

        # Step 1: Extract Annotations
        print(f"1. Extracting Annotations for {jump_label}...")
        if not run_script(EXTRACT_SCRIPT, jump_number):
            print(f"⛔ Skipping remaining steps for {jump_label} due to extraction failure.")
            continue # Move to next jump

        # Step 2: Filter Boxes
        print(f"2. Filtering Boxes for {jump_label}...")
        if not filtra_boxes_txt_local(jump_number):
            print(f"⛔ Skipping remaining steps for {jump_label} due to filtering failure.")
            continue # Move to next jump
        
        # Step 3: Interpolation (Optional)
        if do_interpolate:
            print(f"3. Interpolating Keypoints for {jump_label}...")
            if not run_script(INTERPOLATE_SCRIPT, jump_number):
                print(f"⚠️ Interpolation failed for {jump_label}. Visualization might use non-interpolated data.")
            else:
                print("   Interpolation successful.")
        else:
            print("3. Interpolation skipped (User config).")

        # Step 4 & 5: Visualization & Video (Optional)
        if do_visualize:
            print(f"4. Visualizing Frames for {jump_label}...")
            if run_script(VISUALIZE_SCRIPT, jump_number):
                print(f"5. Creating Video for {jump_label}...")
                run_script(CREATE_VIDEO_SCRIPT, jump_number)
            else:
                print(f"❌ Visualization failed for {jump_label}. Video creation skipped.")
        else:
            print("4. Visualization/Video skipped (User config).")
        
        print(f"✅ Finished processing {jump_label}.")

    print("\n" + "="*50)
    print("Batch processing completed.")
    print("="*50)


if __name__ == "__main__":
    main_workflow()