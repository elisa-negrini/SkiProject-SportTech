import json
import re
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ====== CONFIGURATION ======

# --- DYNAMIC INPUT FOR JUMP NUMBER ---
while True:
    try:
        jump_number = int(input("Enter the Jump number you want to visualize (e.g., 6): "))
        if jump_number > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

JUMP_ID = f"JP{jump_number:04d}" # Format as JP0006, JP0012, etc.

# --- FILE PATHS (Dynamic) ---
coco_path = f"dataset/annotations/{JUMP_ID}/train/annotations_interpolated_jump{jump_number}.coco.json"  # Interpolated COCO file
images_folder = f"dataset/frames/{JUMP_ID}"  # Folder with all frames
output_folder = f"dataset/annotations/{JUMP_ID}/visualizations"  # Where to save visualizations

# Keypoint conversion map
# Array Index (0-based) -> Display Number
KEYPOINT_NUMBER_MAP = {
    1: 1,    # remains 1
    2: 6,    # 2 becomes 6
    3: 3,    # remains 3
    4: 4,    # remains 4
    5: 5,    # remains 5
    6: 7,    # 6 becomes 7
    7: 8,    # 7 becomes 8
    8: 9,    # 8 becomes 9
    9: 10,   # 9 becomes 10
    10: 11,  # 10 becomes 11
    11: 12,  # 11 becomes 12
    12: 17,  # 12 becomes 17
    13: 18,  # 13 becomes 18
    14: 19,  # 14 becomes 19
    15: 20,  # 15 becomes 20
    16: 21,  # 16 becomes 21
    17: 13,  # 17 becomes 13
    18: 14,  # 18 becomes 14
    19: 2,   # 19 becomes 2
    20: 16,  # 20 becomes 16
    21: 15,  # 21 becomes 15
    22: 22,  # remains 22
    23: 23,  # remains 23
}

# Fluorescent yellow/green color for numbers (BGR)
NUMBER_COLOR = (0, 255, 200) 
KEYPOINT_COLORS = {
    1: (255, 210, 0),       # Cyan - head/neck
    2: (0, 0, 0),           # Black - center torso
    3: (255, 128, 0),       # Light Blue - left shoulder
    4: (255, 128, 0),       # Light Blue - left elbow
    5: (255, 128, 0),       # Light Blue - left wrist
    6: (0, 128, 255),       # Orange - right shoulder
    7: (0, 128, 255),       # Orange - right elbow
    8: (0, 128, 255),       # Orange - right wrist
    9: (0, 0, 0),           # Black - center pelvis
    10: (0, 255, 255),      # Yellow - left hip
    11: (0, 255, 255),      # Yellow - left knee
    12: (0, 255, 255),      # Yellow - left ankle
    13: (128, 0, 255),      # Magenta - right hip
    14: (128, 0, 255),      # Magenta - right knee
    15: (128, 0, 255),      # Magenta - right ankle
    16: (128, 0, 255),      # Dark Violet - right foot
    17: (255, 0, 128),      # Violet - left shoulder/side
    18: (255, 0, 128),      # Violet - left torso
    19: (255, 0, 128),      # Violet - left pelvis
    20: (128, 0, 255),      # Pink - posterior left hip
    21: (128, 0, 255),      # Pink - posterior left knee
    22: (128, 0, 255),      # Pink - posterior center
    23: (128, 0, 255),      # Pink - left foot
}

CONNECTION_COLORS = {
    # Head and Neck
    (1, 2): (255, 210, 0),          # Cyan
    
    # Right Arm (Orange)
    (2, 6): (0, 128, 255),      
    (6, 7): (0, 128, 255),      
    (7, 8): (0, 128, 255),      
    
    # Left Arm (Light Blue)
    (2, 3): (255, 128, 0),      
    (3, 4): (255, 128, 0),      
    (4, 5): (255, 128, 0),      
    
    # Center Torso
    (2, 9): (0, 0, 0),          # Black
    
    # Left Leg (Yellow)
    (9, 10): (0, 255, 255),      
    (10, 11): (0, 255, 255),     
    (11, 12): (0, 255, 255),     
    (12, 13): (0, 255, 255),     
    (12, 14): (0, 255, 255),     # Note: Connection from left ankle to left knee (via conversion 12->17, 13->18, 14->19, 12->17, 13->18, 14->19). This is the original logic.

    # Right Leg (Magenta)
    (9, 17): (255, 0, 128),      # Violet
    (17, 18): (255, 0, 128),     # Violet
    (18, 19): (255, 0, 128),     # Violet
    (19, 20): (255, 0, 128),     # Violet
    (19, 21): (255, 0, 128),     # Violet
    
    # Ski/Foot Connections (Pink/Violet)
    (20, 23): (128, 0, 255),     
    (21, 22): (128, 0, 255),     
    (13, 16): (128, 0, 255),     
    (14, 15): (128, 0, 255),     
}

BBOX_COLOR = (0, 255, 0)  # Green for bounding box
INTERPOLATED_ALPHA = 0.6  # Transparency for interpolated frames

# ====== SUPPORT FUNCTIONS ======

def extract_frame_number(name: str) -> int:
    """Extracts the frame number from the file name"""
    m = re.search(r"(\d+)", name)
    if not m:
        raise ValueError(f"No number found in '{name}'")
    return int(m.group(1))

def get_image_name(img):
    """Returns the logical image name"""
    if "extra" in img and isinstance(img["extra"], dict) and "name" in img["extra"]:
        return img["extra"]["name"]
    return img["file_name"]

def draw_skeleton(image, keypoints, skeleton, is_interpolated=False):
    """Draws the skeleton on the image with consistent colors"""
    kp = np.array(keypoints).reshape(-1, 3)
    
    # Create overlay for transparency if interpolated
    overlay = image.copy() if is_interpolated else image
    
    # Draw skeleton connections
    for connection in skeleton:
        pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
        if pt1_idx < len(kp) and pt2_idx < len(kp):
            x1, y1, v1 = kp[pt1_idx]
            x2, y2, v2 = kp[pt2_idx]
            if v1 > 0 and v2 > 0:
                # Convert indices to display numbers
                display_num1 = KEYPOINT_NUMBER_MAP.get(pt1_idx + 1, pt1_idx + 1)
                display_num2 = KEYPOINT_NUMBER_MAP.get(pt2_idx + 1, pt2_idx + 1)
                
                # Look up color in the connections map (try both directions)
                color = CONNECTION_COLORS.get((display_num1, display_num2))
                if color is None:
                    color = CONNECTION_COLORS.get((display_num2, display_num1))
                
                # If no specific connection color is found, use the color of the first point
                if color is None:
                    color = KEYPOINT_COLORS.get(display_num1, (255, 255, 255))
                
                cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    
    # Draw keypoints
    for i, (x, y, v) in enumerate(kp):
        if v > 0:
            original_num = i + 1
            # Use map to get the number to display
            display_num = KEYPOINT_NUMBER_MAP.get(original_num, original_num)
            # Use color based on the DISPLAYED number
            color = KEYPOINT_COLORS.get(display_num, (255, 255, 255))
            
            # Filled circle without border
            cv2.circle(overlay, (int(x), int(y)), 7, color, -1)
            
            # Keypoint number in fluorescent yellow/green - slightly offset (offset +10 up-right)
            text_x = int(x) + 10
            text_y = int(y) - 10
            cv2.putText(overlay, str(display_num), (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, NUMBER_COLOR, 1)
    
    # Apply transparency if interpolated
    if is_interpolated:
        cv2.addWeighted(overlay, INTERPOLATED_ALPHA, image, 1 - INTERPOLATED_ALPHA, 0, image)
    else:
        image[:] = overlay
    
    return image

def draw_bbox(image, bbox, is_interpolated=False):
    """Draws the bounding box"""
    x, y, w, h = [int(v) for v in bbox]
    
    if is_interpolated:
        # Dashed line for interpolated frames
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), BBOX_COLOR, 2)
        cv2.addWeighted(overlay, INTERPOLATED_ALPHA, image, 1 - INTERPOLATED_ALPHA, 0, image)
    else:
        cv2.rectangle(image, (x, y), (x + w, y + h), BBOX_COLOR, 2)
    
    return image

# ====== DATA LOADING ======

print("Loading annotations...")
try:
    with open(coco_path, "r") as f:
        coco = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: COCO file not found at {coco_path}")
    exit()

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# Find the "skier" category and its skeleton
skier_category = next((cat for cat in categories if cat["name"] == "skier"), None)
skeleton = skier_category["skeleton"] if skier_category else []

# Map image_id -> annotation
ann_by_image = {ann["image_id"]: ann for ann in annotations}

# Identify original frames (those with lower IDs are generally originals, adjust if needed)
# Alternatively, you might load these from the original _annotations.coco.json file
original_images_ids = set() 

# Create list of sorted frames
frames = []
for img in images:
    try:
        frame_num = extract_frame_number(get_image_name(img))
        ann = ann_by_image.get(img["id"])
        if ann:
            frames.append((frame_num, img, ann))
    except ValueError as e:
        print(f"Skipping image {get_image_name(img)}: {e}")

frames.sort(key=lambda x: x[0])

# ====== VISUALIZATION ======

output_path_obj = Path(output_folder)
output_path_obj.mkdir(parents=True, exist_ok=True)
images_path = Path(images_folder)

print(f"Processing {len(frames)} frames...")

for frame_num, img, ann in frames:
    # Find the image file
    img_name = get_image_name(img)
    img_path = images_path / img_name
    
    if not img_path.exists():
        print(f"⚠️ Image not found: {img_path}")
        continue
    
    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"⚠️ Could not load: {img_path}")
        continue
    
    # Determine if interpolated (refine this logic if necessary)
    is_interpolated = img["id"] not in original_images_ids if original_images_ids else False
    
    # Draw bbox
    if "bbox" in ann:
        image = draw_bbox(image, ann["bbox"], is_interpolated)
    
    # Draw skeleton
    if "keypoints" in ann and skeleton:
        image = draw_skeleton(image, ann["keypoints"], skeleton, is_interpolated)
    
    # Add label with background
    label = f"Frame {frame_num}" + (" [INTERPOLATED]" if is_interpolated else "")
    font_scale = 1.0
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Background for text
    cv2.rectangle(image, (5, 5), (15 + text_width, 35 + text_height), (0, 0, 0), -1)
    cv2.rectangle(image, (5, 5), (15 + text_width, 35 + text_height), (255, 255, 255), 2)
    
    # Text
    color = (255, 165, 0) if is_interpolated else (0, 255, 0) # Orange or Green
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    # Save
    output_file_path = output_path_obj / f"frame_{frame_num:05d}.jpg"
    cv2.imwrite(str(output_file_path), image)
    
    if frame_num % 10 == 0:
        print(f"✓ Processed frame {frame_num}")

print(f"\n✅ Completed! {len(frames)} images saved in '{output_folder}/'")
print("\nTo create a video (example using ffmpeg, adjust framerate/path as needed):")
print(f"ffmpeg -framerate 10 -pattern_type glob -i '{output_folder}/*.jpg' -c:v libx264 output_video_{JUMP_ID}.mp4")