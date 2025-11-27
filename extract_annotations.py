import json
import os

# --- DYNAMIC INPUT FOR JUMP NUMBER ---
while True:
    try:
        # 1) Add dynamic jump number input
        jump_number = int(input("Enter the Jump number you want to extract (e.g., 6): "))
        if jump_number > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

JUMP_ID = f"JP{jump_number:04d}" # Format as JP0006, JP0012, etc.

# --- CONFIGURATION (Dynamic) ---
# The root path of the folder containing 'train' (where the JSON is located)
DATASET_ROOT = './raw_annotations/' 
TAG_TO_SEARCH = f"jump{jump_number}" # Dynamic tag based on user input (e.g., jump6)
NEW_ANN_FILE_NAME = f"annotations_jump{jump_number}.json" # Renamed output file slightly for clarity

# Path to the full COCO JSON file
SOURCE_ANN_PATH = os.path.join(DATASET_ROOT, 'train', '_annotations.coco.json')

# 2) Output path updated to the visualization folder
TARGET_DIR = os.path.join('dataset', 'annotations', JUMP_ID, 'train')
TARGET_ANN_PATH = os.path.join(TARGET_DIR, NEW_ANN_FILE_NAME)
# ----------------------

# Ensure the target directory exists before saving
os.makedirs(TARGET_DIR, exist_ok=True)

# 1. Load the full COCO JSON file
try:
    with open(SOURCE_ANN_PATH, 'r') as f:
        coco_data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: File not found at path: {SOURCE_ANN_PATH}")
    exit()

# 2. Identify the image IDs to keep (those with the dynamic tag)
images_to_keep = []
image_ids_to_keep = set()

for image in coco_data['images']:
    # Check for the tag presence in 'extra' -> 'user_tags'
    if 'extra' in image and 'user_tags' in image['extra'] and TAG_TO_SEARCH in image['extra']['user_tags']:
        images_to_keep.append(image)
        image_ids_to_keep.add(image['id'])

print(f"✅ Found {len(images_to_keep)} images with tag '{TAG_TO_SEARCH}'.")
if not images_to_keep:
    print("❌ No images found with this tag. Please check the tag name.")
    exit()

# 3. Filter the annotations (only those related to the found image IDs)
annotations_to_keep = []
for annotation in coco_data['annotations']:
    if annotation['image_id'] in image_ids_to_keep:
        annotations_to_keep.append(annotation)

print(f"✅ Found {len(annotations_to_keep)} corresponding annotations.")

# 4. Create the new filtered COCO JSON structure
# Note: The 'images' section is filtered to include only those with the specified tag.
new_coco_data = {
    'info': coco_data['info'],
    'licenses': coco_data['licenses'],
    'categories': coco_data['categories'],
    'images': images_to_keep,          
    'annotations': annotations_to_keep  
}

# 5. Save the new filtered JSON file
with open(TARGET_ANN_PATH, 'w') as f:
    json.dump(new_coco_data, f, indent=4)

print(f"\n🎉 Process completed!")
print(f"Filtered annotations file saved to: {TARGET_ANN_PATH}")