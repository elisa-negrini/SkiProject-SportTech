import json

# --- CONFIG ---
INPUT_FILE = r'dataset\annotations\JP0002\train.json'
OUTPUT_FILE = 'train_converted_coco.json'

# Set image resolution for denormalization (use 1x1 to keep normalized coords)
IMG_WIDTH = 1
IMG_HEIGHT = 1

def convert_to_coco(input_data):
    # Struttura base COCO
    coco_output = {
        "info": {
            "year": "2025",
            "version": "1.0",
            "description": "Converted from train.json",
            "date_created": "2025-12-08"
        },
        "licenses": [{"id": 1, "name": "Unknown"}],
        "categories": [
            {
                "id": 1,
                "name": "skier",
                "supercategory": "objects",
                # Keypoints are numbered 1-23
                "keypoints": [str(i) for i in range(1, 24)], 
                "skeleton": [
                    [18, 21], [17, 20], [1, 19], [19, 2], [14, 15], [8, 12], 
                    [19, 8], [12, 13], [13, 14], [14, 16], [17, 11], [19, 3], 
                    [3, 4], [4, 5], [2, 6], [6, 7], [18, 11], [8, 9], 
                    [9, 10], [10, 11], [16, 22], [15, 23]
                ]
            }
        ],
        "images": [],
        "annotations": []
    }

    annotation_id = 1

    for idx, item in enumerate(input_data):
        # Il file train.json è strutturato come [class_id, [[x,y], [x,y]...]]
        class_id = item[0]
        points = item[1]
        
        image_id = idx + 1
        file_name = f"frame_{idx:05d}.jpg" # Generate placeholder filename

        # 1. Create image entry
        image_info = {
            "id": image_id,
            "license": 1,
            "file_name": file_name,
            "height": IMG_HEIGHT,
            "width": IMG_WIDTH,
            "date_captured": "2025-12-08"
        }
        coco_output["images"].append(image_info)

        # 2. Process keypoints
        coco_keypoints = []
        x_coords = []
        y_coords = []

        for pt in points:
            # Denormalize (or keep if width/height = 1)
            x = pt[0] * IMG_WIDTH
            y = pt[1] * IMG_HEIGHT
            
            x_coords.append(x)
            y_coords.append(y)
            
            # Append x, y and visibility (2 = labeled and visible)
            coco_keypoints.extend([x, y, 2])

        # 3. Compute bounding box (x_min, y_min, width, height)
        if x_coords and y_coords:
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            width = max_x - min_x
            height = max_y - min_y
            bbox = [min_x, min_y, width, height]
            area = width * height
        else:
            bbox = [0, 0, 0, 0]
            area = 0

        # 4. Create annotation entry
        ann_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1, # Forziamo 1 per "skier" come nell'esempio
            "bbox": bbox,
            "area": area,
            "segmentation": [], # Vuoto per i keypoints
            "iscrowd": 0,
            "keypoints": coco_keypoints,
            "num_keypoints": 23
        }
        coco_output["annotations"].append(ann_info)
        annotation_id += 1

    return coco_output

# --- ESECUZIONE ---
try:
    # Carica il contenuto fornito (simulazione lettura file)
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    # Converti
    result = convert_to_coco(data)

    # Salva
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Conversion complete! File saved as: {OUTPUT_FILE}")
    print(f"Processed {len(result['images'])} images and annotations.")

except FileNotFoundError:
    print(f"Error: Input file {INPUT_FILE} not found in folder.")