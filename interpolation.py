import json
import re
import copy
from pathlib import Path
import os
import sys 

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
output_json_path = data_dir / f"annotations_interpolated_jump{jump_number}.coco.json" # Nome aggiornato

# Percorso del file con le bounding box manuali (si assume la struttura del tuo path)
BOXES_FILE = Path(r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiTB dataset\SkiTB\JP") / JUMP_ID / r"MC\boxes.txt"


# === SUPPORT FUNCTIONS (MODIFICHE QUI) ===

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
    Carica le coordinate delle bounding box dal file boxes.txt.
    Ritorna una lista sequenziale di bbox [[x, y, w, h], ...].
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
                        bbox_list.append([int(x) for x in coords])
                except ValueError:
                    continue 
        return bbox_list
    except FileNotFoundError:
        print(f"❌ Errore: File delle bounding box manuali non trovato in {file_path}")
        return None

def normalize_keypoints(kp_data, bbox):
    """
    Normalizza le coordinate dei keypoint (x, y) rispetto alla bounding box [x_bbox, y_bbox, w, h].
    Ritorna una lista di keypoint in coordinate relative (0 a 1).
    """
    x_bbox, y_bbox, w, h = bbox
    kp_norm = []
    for i in range(0, len(kp_data), 3):
        x, y, v = kp_data[i:i+3]
        
        # Normalizzazione: (coordinata - punto_origine_bbox) / dimensione_bbox
        # Evita divisione per zero nel caso di bbox degeneri, anche se raro.
        x_norm = (x - x_bbox) / w if w > 0 else 0
        y_norm = (y - y_bbox) / h if h > 0 else 0
        
        # Visibilità (v) non viene normalizzata
        kp_norm.extend([x_norm, y_norm, v])
    return kp_norm

def denormalize_keypoints(kp_norm_data, bbox_new):
    """
    De-normalizza le coordinate dei keypoint relative (0 a 1) alla nuova bounding box target.
    Ritorna una lista di keypoint in coordinate assolute (pixel).
    """
    x_bbox_new, y_bbox_new, w_new, h_new = bbox_new
    kp_new = []
    for i in range(0, len(kp_norm_data), 3):
        x_norm, y_norm, v = kp_norm_data[i:i+3]
        
        # De-normalizzazione: coordinata_norm * dimensione_bbox_nuova + punto_origine_bbox_nuova
        x = x_norm * w_new + x_bbox_new
        y = y_norm * h_new + y_bbox_new

        # Arrotondamento delle coordinate e gestione della visibilità
        x = round(x, 3)
        y = round(y, 3)
        v = int(round(v)) # Assicura che la visibilità sia un intero
        
        kp_new.extend([x, y, v])
    return kp_new

def interpolate_normalized_keypoints(kp_norm_a, kp_norm_b, t: float):
    """
    Esegue l'interpolazione lineare su keypoint già normalizzati (x, y in 0-1, v in 0-2).
    """
    if len(kp_norm_a) != len(kp_norm_b):
        raise ValueError("Normalized keypoints lists of different lengths, cannot interpolate")

    new_kp_norm = []
    for i in range(0, len(kp_norm_a), 3):
        xa, ya, va = kp_norm_a[i:i+3]
        xb, yb, vb = kp_norm_b[i:i+3]

        x_norm = xa + t * (xb - xa)
        y_norm = ya + t * (yb - ya)
        
        # Interpolazione della visibilità (v)
        if va == vb:
            v_interp = va
        else:
            v_interp = va + t * (vb - va)

        new_kp_norm.extend([x_norm, y_norm, v_interp])

    return new_kp_norm

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

# === BUILD FRAME INDEX & FIND STARTING OFFSET (PER LA MAPPATURA BBOX) ===

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

# Mappa la lista sequenziale di bbox al numero di frame reale (OFFSET CORRETTO)
manual_bbox_by_frame_num = {}
for i, bbox in enumerate(manual_bbox_list):
    frame_num = min_frame_num + i 
    manual_bbox_by_frame_num[frame_num] = bbox

frames_index.sort(key=lambda x: x[0])

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

# === LOOP OVER EXISTING FRAME PAIRS TO INTERPOLATE MISSING ONES (LOGICA BBOX OBBLIGATA) ===

for (frame_a_num, img_a), (frame_b_num, img_b) in zip(frames_index[:-1], frames_index[1:]):
    if frame_b_num <= frame_a_num + 1:
        continue

    ann_a = ann_by_image_id[img_a["id"]]
    ann_b = ann_by_image_id[img_b["id"]]
    
    # Bbox di origine/destinazione per normalizzazione
    bbox_a = ann_a["bbox"]
    bbox_b = ann_b["bbox"]
    
    # Normalizza i keypoint originali una sola volta per coppia di frame
    kp_norm_a = interpolate_normalized_keypoints(normalize_keypoints(ann_a["keypoints"], bbox_a), normalize_keypoints(ann_a["keypoints"], bbox_a), 0.0) # t=0, solo conversione
    kp_norm_b = interpolate_normalized_keypoints(normalize_keypoints(ann_b["keypoints"], bbox_b), normalize_keypoints(ann_b["keypoints"], bbox_b), 0.0) # t=0, solo conversione

    # For all intermediate frames
    for current_frame_num in range(frame_a_num + 1, frame_b_num):
        
        t = (current_frame_num - frame_a_num) / (frame_b_num - frame_a_num)

        # --- New Image creation (Invariato) ---
        new_img = copy.deepcopy(img_a)
        new_img["id"] = next_image_id
        next_image_id += 1
        if make_file_name is not None: new_img["file_name"] = make_file_name(current_frame_num)
        if make_extra_name is not None:
            if "extra" not in new_img or not isinstance(new_img["extra"], dict): new_img["extra"] = {}
            new_img["extra"]["name"] = make_extra_name(current_frame_num)
        new_images.append(new_img)

        # --- New Annotation creation (Invariato) ---
        new_ann = copy.deepcopy(ann_a)
        new_ann["id"] = next_ann_id
        next_ann_id += 1
        new_ann["image_id"] = new_img["id"]

        # 2. BBOX: USA IL DATO MANUALE (TARGET BBOX)
        if current_frame_num in manual_bbox_by_frame_num:
            bbox_manual_target = manual_bbox_by_frame_num[current_frame_num]
            new_ann["bbox"] = bbox_manual_target
        else:
            # Fallback: INTERPOLAZIONE BBOX STANDARD (se il dato manuale manca)
            bbox_manual_target = interpolate_list(bbox_a, bbox_b, t)
            new_ann["bbox"] = bbox_manual_target
            print(f"⚠️ Avviso: Bbox manuale non trovata per frame {current_frame_num}. Usata interpolazione Bbox standard.")


        # 3. KEYPOINTS: INTERPOLA IN SPAZIO NORMALIZZATO E DENORMALIZZA CON TARGET BBOX
        if "keypoints" in ann_a and "keypoints" in ann_b:
            
            # Interpolazione dei keypoint nello spazio 0-1
            kp_interp_norm = interpolate_normalized_keypoints(kp_norm_a, kp_norm_b, t)
            
            # Denormalizzazione nello spazio pixel usando la BBOX MANUALE
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