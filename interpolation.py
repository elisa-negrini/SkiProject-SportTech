import json
import re
import copy
from pathlib import Path
import os # Imported for the dynamic input loop

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

input_json_path = data_dir / "annotations.json" # Assumed name for filtered input
output_json_path = data_dir / "annotations_interpolated.coco.json"

# === SUPPORT FUNCTIONS ===

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

def interpolate_keypoints(kp_a, kp_b, t: float):
    """
    Linear interpolation for COCO keypoints (x, y, v) triplets.
    x and y -> linear interpolation, rounded to 3 decimal places.
    v (visibility) -> keeps the visibility from frame A if equal to B,
                      otherwise interpolates and rounds to the nearest integer.
    """
    if len(kp_a) != len(kp_b):
        raise ValueError("Keypoints of different lengths, cannot interpolate")

    new_kp = []
    for i in range(0, len(kp_a), 3):
        xa, ya, va = kp_a[i:i+3]
        xb, yb, vb = kp_b[i:i+3]

        x = xa + t * (xb - xa)
        y = ya + t * (yb - ya)

        # Round to 3 decimal places (can be changed to integers if desired)
        x = round(x, 3)
        y = round(y, 3)

        if va == vb:
            v = va
        else:
            v = int(round(va + t * (vb - va)))

        new_kp.extend([x, y, v])

    return new_kp


# === LOAD ORIGINAL COCO FILE ===

try:
    with open(input_json_path, "r") as f:
        coco_data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: Input file not found at {input_json_path}. Please check the Jump ID and path.")
    exit()

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

# === BUILD FRAME INDEX FOR EACH IMAGE ===

frames_index = []  # list of (frame_number, image_dict)

for img in images:
    # Prefer original name if it exists ('extra'->'name' field),
    # otherwise use 'file_name'
    if "extra" in img and isinstance(img["extra"], dict) and "name" in img["extra"]:
        name_string = img["extra"]["name"]
    else:
        name_string = img["file_name"]

    frame_number = extract_frame_number(name_string)
    frames_index.append((frame_number, img))

# Sort by frame index
frames_index.sort(key=lambda x: x[0])

# Name template generation
# Use the first example found
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

# === PREPARE NEW IDs FOR IMAGES AND ANNOTATIONS ===

max_image_id = max(img["id"] for img in images) if images else 0
max_ann_id = max(ann["id"] for ann in annotations) if annotations else 0

next_image_id = max_image_id + 1
next_ann_id = max_ann_id + 1

new_images = []
new_annotations = []

# === LOOP OVER EXISTING FRAME PAIRS TO INTERPOLATE MISSING ONES ===

# Iterate over pairs of existing frames (A, B)
for (frame_a_num, img_a), (frame_b_num, img_b) in zip(frames_index[:-1], frames_index[1:]):
    # If there are no "gaps" between frame_a and frame_b, skip
    if frame_b_num <= frame_a_num + 1:
        continue

    ann_a = ann_by_image_id[img_a["id"]]
    ann_b = ann_by_image_id[img_b["id"]]

    # For all intermediate frames
    for current_frame_num in range(frame_a_num + 1, frame_b_num):
        # Calculate interpolation ratio (t)
        t = (current_frame_num - frame_a_num) / (frame_b_num - frame_a_num)  # 0 < t < 1

        # --- New Image creation ---
        new_img = copy.deepcopy(img_a)
        new_img["id"] = next_image_id
        next_image_id += 1

        if make_file_name is not None:
            new_img["file_name"] = make_file_name(current_frame_num)

        if make_extra_name is not None:
            if "extra" not in new_img or not isinstance(new_img["extra"], dict):
                new_img["extra"] = {}
            new_img["extra"]["name"] = make_extra_name(current_frame_num)

        new_images.append(new_img)

        # --- New Annotation creation ---
        new_ann = copy.deepcopy(ann_a)
        new_ann["id"] = next_ann_id
        next_ann_id += 1
        new_ann["image_id"] = new_img["id"]

        # bbox: [x, y, w, h]
        bbox_a = ann_a["bbox"]
        bbox_b = ann_b["bbox"]
        new_bbox = interpolate_list(bbox_a, bbox_b, t)
        new_ann["bbox"] = new_bbox

        # keypoints
        if "keypoints" in ann_a and "keypoints" in ann_b:
            kp_a = ann_a["keypoints"]
            kp_b = ann_b["keypoints"]
            new_kp = interpolate_keypoints(kp_a, kp_b, t)
            new_ann["keypoints"] = new_kp

        new_annotations.append(new_ann)

# === ADD NEW ENTRIES AND SAVE THE NEW JSON ===

coco_data["images"].extend(new_images)
coco_data["annotations"].extend(new_annotations)

# Ensure output directory exists before writing
if not os.path.exists(output_json_path.parent):
    os.makedirs(output_json_path.parent)

with open(output_json_path, "w") as f:
    json.dump(coco_data, f, indent=2)

print(f"File saved to: {output_json_path}")
print(f"New images created: {len(new_images)}")
print(f"New annotations created: {len(new_annotations)}")