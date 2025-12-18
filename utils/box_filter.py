import os
import glob
from pathlib import Path

def get_indices(directory, ext="*.jpg"):
    indices = set()
    for file_path in glob.glob(os.path.join(directory, ext)):
        name = os.path.splitext(os.path.basename(file_path))[0]
        indices.add(name)
    return indices

def filter_boxes(jump_number, frames_root="dataset/frames"):
    jump_id = f"JP{jump_number:04d}"
    base_path = Path(frames_root) / jump_id
    
    if not base_path.exists():
        print(f"❌ Path not found: {base_path}")
        return False

    input_file = base_path / 'boxes.txt'
    output_file = base_path / 'boxes_filtered.txt'
    
    if not input_file.exists():
        print(f"❌ boxes.txt not found in {base_path}")
        return False

    # Get indices from subfolders
    main_idx = get_indices(str(base_path))
    removed_idx = get_indices(str(base_path / 'removed'))
    occluded_idx = get_indices(str(base_path / 'occluded'))
    
    # Calculate skip logic
    all_indices = main_idx | removed_idx | occluded_idx
    all_sorted = sorted(list(all_indices), key=int)
    to_remove = all_indices - main_idx
    
    skip_lines = {i + 1 for i, idx in enumerate(all_sorted) if idx in to_remove}

    try:
        kept = 0
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line_num, line in enumerate(infile, 1):
                if line_num not in skip_lines:
                    outfile.write(line)
                    kept += 1
        
        print(f"✅ Boxes filtered. Lines kept: {kept}")
        return True
    except Exception as e:
        print(f"❌ Error filtering boxes: {e}")
        return False