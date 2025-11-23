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

# Mappa di conversione dei numeri dei keypoints
# Indice array (0-based) -> Numero da visualizzare
KEYPOINT_NUMBER_MAP = {
    1: 1,   # resta 1
    2: 6,   # 2 diventa 6
    3: 3,   # resta 3
    4: 4,   # resta 4
    5: 5,   # resta 5
    6: 7,   # 6 diventa 7
    7: 8,   # 7 diventa 8
    8: 9,   # 8 diventa 9
    9: 10,  # 9 diventa 10
    10: 11, # 10 diventa 11
    11: 12, # 11 diventa 12
    12: 17, # 12 diventa 17
    13: 18, # 13 diventa 18
    14: 19, # 14 diventa 19
    15: 20, # 15 diventa 20
    16: 21, # 16 diventa 21
    17: 13, # 17 diventa 13
    18: 14, # 18 diventa 14
    19: 2,  # 19 diventa 2
    20: 16, # 20 diventa 16
    21: 15, # 21 diventa 15
    22: 22, # resta 22
    23: 23, # resta 23
}

# Colore giallo/verde fluo per i numeri (BGR)
NUMBER_COLOR = (0, 255, 200)  # Verde-giallo fluo
KEYPOINT_COLORS = {
    1: (255, 210, 0),      # Ciano - testa/collo
    2: (0, 0, 0),          # Nero - centro torso
    3: (255, 128, 0),      # Azzurro - spalla sx
    4: (255, 128, 0),      # Azzurro - gomito sx
    5: (255, 128, 0),      # Azzurro - polso sx
    6: (0, 128, 255),      # Arancione - spalla dx
    7: (0, 128, 255),      # Arancione - gomito dx
    8: (0, 128, 255),      # Arancione - polso dx
    9: (0, 0, 0),          # Nero - centro bacino
    10: (0, 255, 255),     # Giallo - anca sx
    11: (0, 255, 255),     # Giallo - ginocchio sx
    12: (0, 255, 255),     # Giallo - caviglia sx
    13: (128, 0, 255),     # Magenta - anca dx
    14: (128, 0, 255),     # Magenta - ginocchio dx
    15: (128, 0, 255),     # Magenta - caviglia dx
    16: (128, 0, 255),     # Viola scuro - piede dx
    17: (255, 0, 128),     # Viola - spalla/lato sx
    18: (255, 0, 128),     # Viola - torso sx
    19: (255, 0, 128),     # Viola - bacino sx
    20: (128, 0, 255),     # Rosa - anca posteriore sx
    21: (128, 0, 255),     # Rosa - ginocchio posteriore sx
    22: (128, 0, 255),     # Rosa - centro posteriore
    23: (128, 0, 255),     # Rosa - piede sx
}

CONNECTION_COLORS = {
    # Testa e collo
    (1, 2): (255, 210, 0),         # Nero
    
    # Braccio destro (Arancione)
    (2, 6): (0, 128, 255),     # Arancione
    (6, 7): (0, 128, 255),     # Arancione
    (7, 8): (0, 128, 255),     # Arancione
    
    # Braccio sinistro (Azzurro)
    (2, 3): (255, 128, 0),     # Azzurro
    (3, 4): (255, 128, 0),     # Azzurro
    (4, 5): (255, 128, 0),     # Azzurro
    
    # Torso centrale
    (2, 9): (0, 0, 0),         # Nero
    
    # Gamba sinistra (Giallo)
    (9, 10): (0, 255, 255),    # Giallo
    (10, 11): (0, 255, 255),   # Giallo
    (11, 12): (0, 255, 255),   # Giallo
    (12, 13): (0, 255, 255),   # Giallo
    (12, 14): (0, 255, 255),   # Giallo

    # Gamba destra (Magenta)
    (9, 17): (255, 0, 128),    # Magenta
    (17, 18): (255, 0, 128),   # Magenta
    (18, 19): (255, 0, 128),   # Magenta
    (19, 20): (255, 0, 128),   # Magenta
    (19, 21): (255, 0, 128),
    
    # Sci
    (20, 23): (128, 0, 255),   # Rosa
    (21, 22): (128, 0, 255),   # Rosa
    (13, 16): (128, 0, 255),   # Rosa
    (14, 15): (128, 0, 255),   # Rosa
}

BBOX_COLOR = (0, 255, 0)  # Verde per bounding box
INTERPOLATED_ALPHA = 0.6  # Trasparenza per frame interpolati

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
    """Disegna lo scheletro sull'immagine con colori coerenti"""
    kp = np.array(keypoints).reshape(-1, 3)
    
    # Crea overlay per trasparenza se interpolato
    overlay = image.copy() if is_interpolated else image
    
    # Disegna le connessioni dello scheletro
    for connection in skeleton:
        pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
        if pt1_idx < len(kp) and pt2_idx < len(kp):
            x1, y1, v1 = kp[pt1_idx]
            x2, y2, v2 = kp[pt2_idx]
            if v1 > 0 and v2 > 0:
                # # Usa il colore basato sul numero VISUALIZZATO del primo punto
                # display_num1 = KEYPOINT_NUMBER_MAP.get(pt1_idx + 1, pt1_idx + 1)
                # color = KEYPOINT_COLORS.get(display_num1, (255, 255, 255))
                # cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                # Converti gli indici in numeri visualizzati
                display_num1 = KEYPOINT_NUMBER_MAP.get(pt1_idx + 1, pt1_idx + 1)
                display_num2 = KEYPOINT_NUMBER_MAP.get(pt2_idx + 1, pt2_idx + 1)
                
                # Cerca il colore nella mappa delle connessioni (prova entrambe le direzioni)
                color = CONNECTION_COLORS.get((display_num1, display_num2))
                if color is None:
                    color = CONNECTION_COLORS.get((display_num2, display_num1))
                
                # Se non trova una connessione specifica, usa il colore del primo punto
                if color is None:
                    color = KEYPOINT_COLORS.get(display_num1, (255, 255, 255))
                
                cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    
    # Disegna i keypoints
    for i, (x, y, v) in enumerate(kp):
        if v > 0:
            original_num = i + 1
            # Usa la mappa per ottenere il numero da visualizzare
            display_num = KEYPOINT_NUMBER_MAP.get(original_num, original_num)
            # Usa il colore basato sul numero VISUALIZZATO
            color = KEYPOINT_COLORS.get(display_num, (255, 255, 255))
            
            # Cerchio pieno senza bordo
            cv2.circle(overlay, (int(x), int(y)), 7, color, -1)
            
            # Numero del keypoint in giallo/verde fluo - leggermente fuori dal cerchio (offset +10 in alto a destra)
            text_x = int(x) + 10
            text_y = int(y) - 10
            cv2.putText(overlay, str(display_num), (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, NUMBER_COLOR, 1)
    
    # Applica trasparenza se interpolato
    if is_interpolated:
        cv2.addWeighted(overlay, INTERPOLATED_ALPHA, image, 1 - INTERPOLATED_ALPHA, 0, image)
    else:
        image[:] = overlay
    
    return image

def draw_bbox(image, bbox, is_interpolated=False):
    """Disegna il bounding box"""
    x, y, w, h = [int(v) for v in bbox]
    
    if is_interpolated:
        # Linea tratteggiata per interpolati
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), BBOX_COLOR, 2)
        cv2.addWeighted(overlay, INTERPOLATED_ALPHA, image, 1 - INTERPOLATED_ALPHA, 0, image)
    else:
        cv2.rectangle(image, (x, y), (x + w, y + h), BBOX_COLOR, 2)
    
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
    
    # Aggiungi label con sfondo
    label = f"Frame {frame_num}" + (" [INTERPOLATED]" if is_interpolated else "")
    font_scale = 1.0
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Sfondo per il testo
    cv2.rectangle(image, (5, 5), (15 + text_width, 35 + text_height), (0, 0, 0), -1)
    cv2.rectangle(image, (5, 5), (15 + text_width, 35 + text_height), (255, 255, 255), 2)
    
    # Testo
    color = (255, 165, 0) if is_interpolated else (0, 255, 0)
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    # Salva
    output_path = Path(output_folder) / f"frame_{frame_num:05d}.jpg"
    cv2.imwrite(str(output_path), image)
    
    if frame_num % 10 == 0:
        print(f"✓ Processato frame {frame_num}")

print(f"\n✅ Completato! {len(frames)} immagini salvate in '{output_folder}/'")
print("\nPer creare un video:")
print(f"ffmpeg -framerate 10 -pattern_type glob -i '{output_folder}/*.jpg' -c:v libx264 output_video.mp4")