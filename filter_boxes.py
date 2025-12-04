import os
import glob
from pathlib import Path # Added for better path management

def filtra_boxes_txt(base_dir, file_extension="*.jpg"):
    """
    Removes lines from the boxes.txt file that correspond to frames
    missing in the main directory compared to the total.

    Args:
        base_dir (str): The path to the base directory (e.g., '.../JP0006').
        file_extension (str): The image file extension (e.g., "*.jpg").
    """
    
    # --- Path Definition ---
    dir_principale = os.path.join(base_dir)
    dir_removed = os.path.join(base_dir, 'removed')
    dir_occluded = os.path.join(base_dir, 'occluded')
    
    # The original boxes.txt file is in the main directory
    input_file_path = os.path.join(dir_principale, 'boxes.txt')
    output_file_path = os.path.join(dir_principale, 'boxes_filtered.txt')
    
    if not os.path.exists(input_file_path):
        print(f"ERROR: boxes.txt file not found at path: {input_file_path}")
        return

    # --- Helper Function (remains unchanged) ---
    def get_indices_from_dir(directory):
        indices = set()
        pattern = os.path.join(directory, file_extension)
        # We replace the previous glob with a more robust glob for .jpg only
        file_paths = glob.glob(pattern)
        for file_path in file_paths:
            full_file_name = os.path.basename(file_path)
            name_without_extension, _ = os.path.splitext(full_file_name)
            indices.add(name_without_extension)
        return indices

    # 1. Calculate Indices and Skip Positions
    
    main_indices = get_indices_from_dir(dir_principale)
    removed_indices = get_indices_from_dir(dir_removed)
    occluded_indices = get_indices_from_dir(dir_occluded)
    
    # 1a. Calculate all unique indices
    tutti_gli_indici_set = main_indices.union(removed_indices).union(occluded_indices)
    # Sort the indices as integers to determine the correct line number in boxes.txt
    tutti_gli_indici_ordinati = sorted(list(tutti_gli_indici_set), key=int)
    
    # 1b. Calculate indices to remove (those not in 'principale')
    indici_da_rimuovere = tutti_gli_indici_set.difference(main_indices)
    
    # 1c. Map the missing index to the 1-based POSITION (line number)
    posizioni_da_saltare = set()
    for posizione_0based, indice in enumerate(tutti_gli_indici_ordinati):
        if indice in indici_da_rimuovere:
            posizioni_da_saltare.add(posizione_0based + 1)

    print(f"Indices to remove: {sorted(list(indici_da_rimuovere), key=int)}")
    print(f"Total lines to skip: {len(posizioni_da_saltare)}")
    print(f"Positions (line number) to skip in boxes.txt: {sorted(list(posizioni_da_saltare))}")
    print("-" * 70)
    
    # 2. Read the input file and Write the filtered file
    
    righe_mantenute_count = 0
    righe_saltate_count = 0
    
    try:
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line_number, line in enumerate(infile, 1):
                # line_number is the 1-based line number (1, 2, 3, ...)
                
                if line_number not in posizioni_da_saltare:
                    outfile.write(line)
                    righe_mantenute_count += 1
                else:
                    righe_saltate_count += 1
                    
        print(f"✅ Filtering completed successfully!")
        print(f"Original file: {input_file_path}")
        print(f"New file created: {output_file_path}")
        print(f"   - Lines kept: {righe_mantenute_count}")
        print(f"   - Lines skipped (removed): {righe_saltate_count}")
        print(f"   - Total original lines (estimated): {righe_mantenute_count + righe_saltate_count}")

    except Exception as e:
        print(f"ERROR during file opening/writing: {e}")


# --- Main Part: UPDATED Execution ---

# Fixed base path up to the 'frames' directory
ROOT_PATH = Path(r"dataset\frames")
FILE_EXT = "*.jpg"

while True:
    try:
        # 1. Ask for the jump number
        jump_number = int(input("Enter the Jump number (e.g., 6 for JP0006): "))
        if jump_number > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# 2. Construct the formatted JUMP ID (e.g., 6 -> JP0006)
JUMP_ID = f"JP{jump_number:04d}"

# 3. Construct the full base directory path
BASE_PATH = ROOT_PATH / JUMP_ID

if not BASE_PATH.exists():
    print(f"ERROR: The base directory for jump {JUMP_ID} does not exist: {BASE_PATH}")
    # sys.exit() is imported in the main_annotation.py but not here, so a print is sufficient.
    # If this were a standalone script, sys.exit() would be needed.
    # Assuming this script is run as part of a larger system or that manual exit is expected.
    # NOTE: sys is not imported in this specific file, so a functional exit is not possible.
    pass

# 4. Call the filtering function
print("=" * 70)
print(f"STARTING BOXES.TXT FILE FILTERING for {JUMP_ID}")
print("=" * 70)
filtra_boxes_txt(BASE_PATH, FILE_EXT)