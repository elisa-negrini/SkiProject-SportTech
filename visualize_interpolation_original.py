import json
import re
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ====== CONFIGURAZIONE ======
coco_path = "dataset/jump-3/train/annotations_interpolated.coco.json"  # Il tuo file interpolato
images_folder = "C:/Users/utente/Desktop/UNITN secondo anno/Sport Tech/ski project/SkiTB dataset/SkiTB/JP/JP0003/frames"  # Cartella con tutte le immagini
output_folder = "dataset/jump-3/visualizations"  # Dove salvare le visualizzazioni

# Colori per visualizzazione
KEYPOINT_COLOR = (0, 255, 0)  # Verde per i keypoints
SKELETON_COLOR = (255, 0, 0)  # Rosso per le connessioni
BBOX_COLOR = (0, 0, 255)  # Blu per bounding box
INTERPOLATED_COLOR = (255, 165, 0)  # Arancione per frame interpolati

# ====== FUNZIONI DI SUPPORTO ======

def extract_frame_number(name: str) -> int:
    """Estrae il numero del frame dal nome file"""
    m = re.search(r"(\d+)", name)
    if not m:
        raise ValueError(f"Nessun numero trovato in '{name}'")
    return int(m.group(1))

def get_image_name(img):
    """Ritorna il nome logico dell'immagine"""
    if "extra" in img and isinstance(img["extra"], dict) and "name" in img["extra"]:
        return img["extra"]["name"]
    return img["file_name"]

def draw_skeleton(image, keypoints, skeleton, is_interpolated=False):
    """Disegna lo scheletro sull'immagine"""
    kp = np.array(keypoints).reshape(-1, 3)
    
    # Colore diverso per frame interpolati
    kp_color = INTERPOLATED_COLOR if is_interpolated else KEYPOINT_COLOR
    sk_color = INTERPOLATED_COLOR if is_interpolated else SKELETON_COLOR
    
    # Disegna le connessioni dello scheletro
    for connection in skeleton:
        pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1  # Gli indici partono da 1
        if pt1_idx < len(kp) and pt2_idx < len(kp):
            x1, y1, v1 = kp[pt1_idx]
            x2, y2, v2 = kp[pt2_idx]
            if v1 > 0 and v2 > 0:  # Disegna solo se entrambi i punti sono visibili
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), sk_color, 2)
    
    # Disegna i keypoints
    for i, (x, y, v) in enumerate(kp):
        if v > 0:  # Solo se visibile
            cv2.circle(image, (int(x), int(y)), 5, kp_color, -1)
            # Aggiungi il numero del keypoint
            cv2.putText(image, str(i+1), (int(x)+7, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, kp_color, 1)
    
    return image

def draw_bbox(image, bbox, is_interpolated=False):
    """Disegna il bounding box"""
    x, y, w, h = [int(v) for v in bbox]
    color = INTERPOLATED_COLOR if is_interpolated else BBOX_COLOR
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    return image

# ====== CARICAMENTO DATI ======

print("Caricamento annotazioni...")
with open(coco_path, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# Trova la categoria "skier" e il suo skeleton
skier_category = next((cat for cat in categories if cat["name"] == "skier"), None)
skeleton = skier_category["skeleton"] if skier_category else []

# Mappa image_id -> annotation
ann_by_image = {ann["image_id"]: ann for ann in annotations}

# Identifica frame originali (quelli con ID più bassi sono generalmente gli originali)
# oppure puoi confrontare con il file originale
original_images_ids = set()  # Potresti voler caricare questi da _annotations.coco.json

# Crea lista di frame ordinati
frames = []
for img in images:
    frame_num = extract_frame_number(get_image_name(img))
    ann = ann_by_image.get(img["id"])
    if ann:
        frames.append((frame_num, img, ann))

frames.sort(key=lambda x: x[0])

# ====== VISUALIZZAZIONE ======

Path(output_folder).mkdir(exist_ok=True)
images_path = Path(images_folder)

print(f"Elaborazione di {len(frames)} frame...")

for frame_num, img, ann in frames:
    # Cerca l'immagine
    img_name = get_image_name(img)
    img_path = images_path / img_name
    
    if not img_path.exists():
        print(f"⚠️  Immagine non trovata: {img_path}")
        continue
    
    # Carica immagine
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"⚠️  Impossibile caricare: {img_path}")
        continue
    
    # Determina se è interpolato (puoi raffinare questa logica)
    is_interpolated = img["id"] not in original_images_ids if original_images_ids else False
    
    # Disegna bbox
    if "bbox" in ann:
        image = draw_bbox(image, ann["bbox"], is_interpolated)
    
    # Disegna skeleton
    if "keypoints" in ann and skeleton:
        image = draw_skeleton(image, ann["keypoints"], skeleton, is_interpolated)
    
    # Aggiungi label
    label = f"Frame {frame_num}" + (" [INTERPOLATED]" if is_interpolated else " [ORIGINAL]")
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
               INTERPOLATED_COLOR if is_interpolated else (255, 255, 255), 2)
    
    # Salva
    output_path = Path(output_folder) / f"frame_{frame_num:05d}.jpg"
    cv2.imwrite(str(output_path), image)
    
    if frame_num % 10 == 0:
        print(f"✓ Processato frame {frame_num}")

print(f"\n✅ Completato! {len(frames)} immagini salvate in '{output_folder}/'")
print("\nPer creare un video:")
print(f"ffmpeg -framerate 10 -pattern_type glob -i '{output_folder}/*.jpg' -c:v libx264 output_video.mp4")