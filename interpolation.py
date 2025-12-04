import json
import re
import copy
from pathlib import Path
import os
import sys 
import glob

# === DYNAMIC INPUT FOR JUMP NUMBER ===
while True:
    try:
        # Prompt user for the jump number
        jump_number = int(input("Enter the Jump number to process (e.g., 6): "))
        if jump_number > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

JUMP_ID = f"JP{jump_number:04d}" # Format as JP0006, JP0012, etc.

# === PATH DEFINITIONS ===
# Defines the subdirectory using the dynamic JUMP_ID
data_dir = Path("dataset") / "annotations" / JUMP_ID / "train" 

input_json_path = data_dir / f"annotations_jump{jump_number}.json" 
output_json_path = data_dir / f"annotations_interpolated_jump{jump_number}.coco.json" # Updated name

# Path to the file with manual bounding boxes (assuming your path structure)
#BOXES_FILE = Path(r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiTB dataset\SkiTB\JP") / JUMP_ID / r"MC\boxes.txt"
BOXES_FILE = Path(r"dataset\frames")/ JUMP_ID / r"boxes_filtered.txt"

# === SUPPORT FUNCTIONS (MODIFICATIONS HERE) ===

def extract_frame_number(name: str) -> int:
    """
    Extracts the first group of digits from a string (e.g., '00063.jpg' -> 63).
    """
    m = re.search(r"(\d+)", name)
    if not m:
        raise ValueError(f"No number found in '{name}'")
    return int(m.group(1))

def build_name_template(example_name: str):
    """
    Given an example name like '00063.jpg' or 'frame_00063.png',
    builds a template function that returns a consistent name for a given frame number, 
    e.g., 64 -> '00064.jpg'.
    """
    m = re.search(r"(\d+)", example_name)
    if not m:
        raise ValueError(f"No number found in '{example_name}'")
    prefix = example_name[:m.start()]
    digits = m.group(1)
    suffix = example_name[m.end():]
    pad = len(digits)

    def make_name(frame_number: int) -> str:
        return f"{prefix}{frame_number:0{pad}d}{suffix}"

    return make_name

def interpolate_list(list_a, list_b, t: float):
    """
    Linear interpolation element-wise between two lists of numbers.
    Returns a list of floats.
    """
    if len(list_a) != len(list_b):
        raise ValueError("Lists of different lengths, cannot interpolate")
    return [a + t * (b - a) for a, b in zip(list_a, list_b)]

def load_manual_bbox_data(file_path):
    """
    Loads bounding box coordinates from the boxes.txt file.
    Returns a sequential list of bboxes [[x, y, w, h], ...].
    """
    bbox_list = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                clean_line = line.strip()
                
                if not clean_line or clean_line.startswith('['):
                    continue
                
                try:
                    coords = [float(c.strip()) for c in clean_line.split(',')]
                    if len(coords) == 4:
                        # Convert coordinates to integers (as expected for COCO BBoxes)
                        bbox_list.append([int(x) for x in coords])
                except ValueError:
                    continue 
        return bbox_list
    except FileNotFoundError:
        print(f"❌ Error: Manual bounding box file not found at {file_path}")
        return None

def normalize_keypoints(kp_data, bbox):
    """
    Normalizes keypoint coordinates (x, y) with respect to the bounding box [x_bbox, y_bbox, w, h].
    Returns a list of keypoints in relative coordinates (0 to 1).
    """
    x_bbox, y_bbox, w, h = bbox
    kp_norm = []
    for i in range(0, len(kp_data), 3):
        x, y, v = kp_data[i:i+3]
        
        # Normalization: (coordinate - bbox_origin_point) / bbox_dimension
        # Avoid division by zero in case of degenerate bboxes, though rare.
        x_norm = (x - x_bbox) / w if w > 0 else 0
        y_norm = (y - y_bbox) / h if h > 0 else 0
        
        # Visibility (v) is not normalized
        kp_norm.extend([x_norm, y_norm, v])
    return kp_norm

def denormalize_keypoints(kp_norm_data, bbox_new):
    """
    De-normalizes relative keypoint coordinates (0 to 1) to the new target bounding box.
    Returns a list of keypoints in absolute coordinates (pixels).
    """
    x_bbox_new, y_bbox_new, w_new, h_new = bbox_new
    kp_new = []
    for i in range(0, len(kp_norm_data), 3):
        x_norm, y_norm, v = kp_norm_data[i:i+3]
        
        # De-normalization: normalized_coordinate * new_bbox_dimension + new_bbox_origin_point
        x = x_norm * w_new + x_bbox_new
        y = y_norm * h_new + y_bbox_new

        # Rounding coordinates and handling visibility
        x = round(x, 3)
        y = round(y, 3)
        v = int(round(v)) # Ensures visibility is an integer
        
        kp_new.extend([x, y, v])
    return kp_new

def interpolate_normalized_keypoints(kp_norm_a, kp_norm_b, t: float):
    """
    Performs linear interpolation on already normalized keypoints (x, y in 0-1, v in 0-2).
    """
    if len(kp_norm_a) != len(kp_norm_b):
        raise ValueError("Normalized keypoints lists of different lengths, cannot interpolate")

    new_kp_norm = []
    for i in range(0, len(kp_norm_a), 3):
        xa, ya, va = kp_norm_a[i:i+3]
        xb, yb, vb = kp_norm_b[i:i+3]

        x_norm = xa + t * (xb - xa)
        y_norm = ya + t * (yb - ya)
        
        # Interpolate visibility (v)
        if va == vb:
            v_interp = va
        else:
            v_interp = va + t * (vb - va)

        new_kp_norm.extend([x_norm, y_norm, v_interp])

    return new_kp_norm

def get_indices_from_dir(directory):
    """ Retrieves frame numbers from a directory using glob and extract_frame_number. """
    indices = set()
 
    if not os.path.exists(directory):
        print(f"⚠️ Directory not found: {directory}")
        return indices

    try:
        # Use the most probable pattern or search for both
        #pattern_png = os.path.join(directory, "????*.png")
        pattern_jpg = os.path.join(directory, "*.jpg")
        
        file_paths = glob.glob(pattern_jpg)
            
        for file_path in file_paths:
            full_file_name = os.path.basename(file_path)
            try:
                frame_number = extract_frame_number(full_file_name)
                indices.add(frame_number)
            except ValueError:
                # Ignore non-conforming files
                continue
        return indices
    except Exception as e:
        print(f"⚠️ Warning while retrieving indices from {directory}: {e}")
        return indices
    
# === LOAD ORIGINAL COCO FILE & MANUAL BBOX DATA ===

try:
    with open(input_json_path, "r") as f:
        coco_data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: Input file not found at {input_json_path}. Please check the Jump ID and path.")
    sys.exit() 

manual_bbox_list = load_manual_bbox_data(BOXES_FILE)
if manual_bbox_list is None:
    sys.exit() 

images = coco_data["images"]
annotations = coco_data["annotations"]

# Map image_id -> annotation (assumes ONE annotation per image)
ann_by_image_id = {}
for ann in annotations:
    image_id = ann["image_id"]
    if image_id in ann_by_image_id:
        raise ValueError(f"More than one annotation for image_id={image_id}, "
                         "the code assumes a single annotation per image.")
    ann_by_image_id[image_id] = ann

annotated_image_ids = set(ann_by_image_id.keys())

# === BUILD FRAME INDEX & FIND STARTING OFFSET (FOR BBOX MAPPING) ===

frames_index = []
all_frame_nums = []

for img in images:
    if "extra" in img and isinstance(img["extra"], dict) and "name" in img["extra"]:
        name_string = img["extra"]["name"]
    else:
        name_string = img["file_name"]
    
    frame_number = extract_frame_number(name_string)
    all_frame_nums.append(frame_number)
    
    if img["id"] in annotated_image_ids:
        frames_index.append((frame_number, img))

min_frame_num = min(all_frame_nums) if all_frame_nums else 1

# Rebuild the base path for directories (necessary for glob)
# Assuming BOXES_FILE is correctly defined as Path(r"dataset\frames")/ JUMP_ID / ...
BASE_PATH = BOXES_FILE.parent
dir_principale = BASE_PATH
dir_removed = BASE_PATH / 'removed'
dir_occluded = BASE_PATH / 'occluded'

# Debug: print paths to verify
print(f"🔍 Debug - Paths searched:")
print(f"   Base: {dir_principale}")
print(f"   Removed: {dir_removed}")
print(f"   Occluded: {dir_occluded}")
print(f"   Does dir_principale exist? {dir_principale.exists()}")

# 1. Calculate all index sets
indici_principale = get_indices_from_dir(dir_principale)
indici_removed = get_indices_from_dir(dir_removed)
indici_occluded = get_indices_from_dir(dir_occluded)

tutti_gli_indici_set = indici_principale.union(indici_removed).union(indici_occluded)

# 2. Calculate the set of KEPT frames (those that are in 'principale' OR that are in total BUT have NOT been removed)
# Since boxes_filtered.txt contains lines only for frames that have NOT been removed.
# The kept frames are simply all the total indices that were NOT discarded.
indici_removed = tutti_gli_indici_set.difference(indici_principale)
indici_mantenuti_set = tutti_gli_indici_set.difference(indici_removed) # This is the correct set

# 3. Create the sorted list of kept frame numbers
KEPT_FRAME_NUMBERS_IN_ORDER = sorted(list(indici_mantenuti_set), key=int)

# --- SAFETY CHECK ---
if len(KEPT_FRAME_NUMBERS_IN_ORDER) != len(manual_bbox_list):
    print("❌ CRITICAL SYNC ERROR! Length of kept frame list does not match length of filtered bbox list.")
    print(f"   - Kept Frames (from file system logic): {len(KEPT_FRAME_NUMBERS_IN_ORDER)}")
    print(f"   - Filtered BBoxes (from boxes_filtered.txt): {len(manual_bbox_list)}")
    print("   -> Check the image file extension or the filtering process.")
    sys.exit()

# --- CORRECT MAPPING BLOCK ---
# Map the sequential bbox list to the correct, non-contiguous list of kept frame numbers
manual_bbox_by_frame_num = {}
for i, bbox in enumerate(manual_bbox_list):
    # i is the sequential index (0, 1, 2, ...) which corresponds to the line in boxes_filtered.txt
    # KEPT_FRAME_NUMBERS_IN_ORDER[i] is the actual frame number (e.g., 52)
    frame_num = KEPT_FRAME_NUMBERS_IN_ORDER[i]
    manual_bbox_by_frame_num[frame_num] = bbox
    
# Find the actual minimum frame number to use as a template reference (min_frame_num is no longer used for mapping, but might be needed elsewhere)
min_frame_num = min(KEPT_FRAME_NUMBERS_IN_ORDER) if KEPT_FRAME_NUMBERS_IN_ORDER else 1

# --- FRAME INDEX CONSTRUCTION (The rest of the code remains almost unchanged) ---

images = coco_data["images"]
annotations = coco_data["annotations"]

# Map image_id -> annotation (assumes ONE annotation per image)
ann_by_image_id = {}
for ann in annotations:
    image_id = ann["image_id"]
    if image_id in ann_by_image_id:
        raise ValueError(f"More than one annotation for image_id={image_id}, "
                         "the code assumes a single annotation per image.")
    ann_by_image_id[image_id] = ann

annotated_image_ids = set(ann_by_image_id.keys())

# === BUILD FRAME INDEX & NAME TEMPLATE ===
frames_index = []

for img in images:
    if "extra" in img and isinstance(img["extra"], dict) and "name" in img["extra"]:
        name_string = img["extra"]["name"]
    else:
        name_string = img["file_name"]
    
    frame_number = extract_frame_number(name_string)
    
    if img["id"] in annotated_image_ids:
        frames_index.append((frame_number, img))

# min_frame_num was calculated earlier, no need to recalculate here.

frames_index.sort(key=lambda x: x[0])

# Name template generation
example_extra_name = None
example_file_name = None

# Name template generation
example_extra_name = None
example_file_name = None

for _, img in frames_index:
    if "extra" in img and isinstance(img["extra"], dict) and "name" in img["extra"]:
        example_extra_name = img["extra"]["name"]
        break

for _, img in frames_index:
    if img.get("file_name"):
        example_file_name = img["file_name"]
        break

make_extra_name = build_name_template(example_extra_name) if example_extra_name else None
make_file_name = build_name_template(example_file_name) if example_file_name else None

# === PREPARE NEW IDs ===

max_image_id = max(img["id"] for img in images) if images else 0
max_ann_id = max(ann["id"] for ann in annotations) if annotations else 0

next_image_id = max_image_id + 1
next_ann_id = max_ann_id + 1

new_images = []
new_annotations = []

# === LOOP OVER EXISTING FRAME PAIRS TO INTERPOLATE MISSING ONES (MANDATORY BBOX LOGIC) ===

for (frame_a_num, img_a), (frame_b_num, img_b) in zip(frames_index[:-1], frames_index[1:]):
    if frame_b_num <= frame_a_num + 1:
        continue

    ann_a = ann_by_image_id[img_a["id"]]
    ann_b = ann_by_image_id[img_b["id"]]
    
    # Source/destination Bbox for normalization
    bbox_a = ann_a["bbox"]
    bbox_b = ann_b["bbox"]
    
    # Normalize original keypoints only once per frame pair
    kp_norm_a = interpolate_normalized_keypoints(normalize_keypoints(ann_a["keypoints"], bbox_a), normalize_keypoints(ann_a["keypoints"], bbox_a), 0.0) # t=0, conversion only
    kp_norm_b = interpolate_normalized_keypoints(normalize_keypoints(ann_b["keypoints"], bbox_b), normalize_keypoints(ann_b["keypoints"], bbox_b), 0.0) # t=0, conversion only

    # For all intermediate frames
    for current_frame_num in range(frame_a_num + 1, frame_b_num):
        
        t = (current_frame_num - frame_a_num) / (frame_b_num - frame_a_num)

        # --- New Image creation (Unchanged) ---
        new_img = copy.deepcopy(img_a)
        new_img["id"] = next_image_id
        next_image_id += 1
        if make_file_name is not None: new_img["file_name"] = make_file_name(current_frame_num)
        if make_extra_name is not None:
            if "extra" not in new_img or not isinstance(new_img["extra"], dict): new_img["extra"] = {}
            new_img["extra"]["name"] = make_extra_name(current_frame_num)
        new_images.append(new_img)

        # --- New Annotation creation (Unchanged) ---
        new_ann = copy.deepcopy(ann_a)
        new_ann["id"] = next_ann_id
        next_ann_id += 1
        new_ann["image_id"] = new_img["id"]

        # 2. BBOX: USE MANUAL DATA (TARGET BBOX)
        if current_frame_num in manual_bbox_by_frame_num:
            bbox_manual_target = manual_bbox_by_frame_num[current_frame_num]
            new_ann["bbox"] = bbox_manual_target
        else:
            # Fallback: STANDARD BBOX INTERPOLATION (if manual data is missing)
            bbox_manual_target = interpolate_list(bbox_a, bbox_b, t)
            new_ann["bbox"] = bbox_manual_target
            print(f"⚠️ Warning: Manual Bbox not found for frame {current_frame_num}. Standard Bbox interpolation used.")


        # 3. KEYPOINTS: INTERPOLATE IN NORMALIZED SPACE AND DENORMALIZE WITH TARGET BBOX
        if "keypoints" in ann_a and "keypoints" in ann_b:
            
            # Keypoint interpolation in 0-1 space
            kp_interp_norm = interpolate_normalized_keypoints(kp_norm_a, kp_norm_b, t)
            
            # Denormalization to pixel space using the MANUAL BBOX
            new_kp = denormalize_keypoints(kp_interp_norm, bbox_manual_target)
            new_ann["keypoints"] = new_kp

        new_annotations.append(new_ann)

# === ADD NEW ENTRIES AND SAVE THE NEW JSON ===

coco_data["images"].extend(new_images)
coco_data["annotations"].extend(new_annotations)

if not os.path.exists(output_json_path.parent):
    os.makedirs(output_json_path.parent)

with open(output_json_path, "w") as f:
    json.dump(coco_data, f, indent=2)

print(f"File saved to: {output_json_path}")
print(f"New images created: {len(new_images)}")
print(f"New annotations created: {len(new_annotations)}")