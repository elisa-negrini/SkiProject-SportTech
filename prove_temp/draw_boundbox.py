import cv2
import numpy as np
import os
from pathlib import Path

# =================================================================
# === CONFIGURAZIONE UTENTE (MODIFICARE QUI) ======================
# =================================================================

# ⚠️ INSERISCI QUI IL PERCORSO ALLA TUA CARTELLA DI IMMAGINI (es. dataset/frames/JP0003/)
IMAGES_DIR = "dataset/frames/JP0007" 
# ⚠️ INSERISCI QUI IL PERCORSO ALLA CARTELLA DOVE SALVARE I RISULTATI
OUTPUT_DIR = "dataset/visualizations/bbox_manual_jp7"
# Nome del file contenente le coordinate delle bounding box (Usare r'' per i percorsi Windows)
BOXES_FILE = r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiTB dataset\SkiTB\JP\JP0007\MC\boxes.txt"

# Stile di disegno
BBOX_COLOR = (0, 255, 255) # Giallo (BGR)
BBOX_THICKNESS = 2

# =================================================================
# === FUNZIONI DI SUPPORTO ========================================
# =================================================================

def load_bbox_data(file_path):
    """
    Carica le coordinate delle bounding box dal file e le restituisce come lista di [x, y, w, h].
    Ogni riga corrisponde a un frame consecutivo.
    """
    bbox_list = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Pulisce la riga da spazi e caratteri di citazione
                clean_line = line.strip()
                
                # Ignora righe vuote o righe di intestazione (es. )
                if not clean_line or clean_line.startswith('['):
                    continue
                
                # Le coordinate sono separate da virgole
                try:
                    coords = [float(c.strip()) for c in clean_line.split(',')]
                    if len(coords) == 4:
                        # [x, y, w, h]
                        bbox_list.append([int(x) for x in coords])
                except ValueError:
                    print(f"⚠️ Riga bbox non valida e ignorata: {line.strip()}")
        return bbox_list
    except FileNotFoundError:
        print(f"❌ Errore: File delle bounding box non trovato in {file_path}")
        return None

# =================================================================
# === ESECUZIONE PRINCIPALE =======================================
# =================================================================

# 1. Carica i dati delle bounding box
bbox_data = load_bbox_data(BOXES_FILE)

if bbox_data is None:
    # Uscita se il file non è stato trovato
    exit()

# 2. Prepara le directory
images_path = Path(IMAGES_DIR)
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

# Trova tutti i file immagine e li ordina numericamente
image_files = sorted(
    [f for f in images_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']],
    # Estrai e ordina per il numero nel nome del file (es. 00001)
    key=lambda x: int(''.join(filter(str.isdigit, x.stem)))
)

if not image_files:
    print(f"❌ Errore: Nessuna immagine trovata in {images_path}")
    exit()

print(f"🖼️ Trovate {len(image_files)} immagini. Disegno {len(bbox_data)} bounding box.")

# 3. Processa e disegna
processed_count = 0
for i, img_file in enumerate(image_files):
    # La riga 'i' in bbox_data corrisponde al frame 'i'
    if i >= len(bbox_data):
        print(f"⚠️ Non ci sono più dati di bounding box per il frame {img_file.name}. Interruzione.")
        break
        
    # Carica l'immagine
    image = cv2.imread(str(img_file))
    if image is None:
        print(f"❌ Impossibile caricare l'immagine: {img_file.name}")
        continue
    
    # Estrai la bounding box [x, y, w, h]
    x, y, w, h = bbox_data[i]
    
    # Disegna il rettangolo
    # Coordinate: punto iniziale (x, y) e punto finale (x+w, y+h)
    cv2.rectangle(image, (x, y), (x + w, y + h), BBOX_COLOR, BBOX_THICKNESS)
    
    # Salva l'immagine processata
    output_file = output_path / img_file.name
    cv2.imwrite(str(output_file), image)
    
    processed_count += 1
    if (i + 1) % 50 == 0:
        print(f"✓ Processati {i + 1} frames.")

print(f"\n✅ Completato! {processed_count} immagini con bounding box manuale salvate in '{OUTPUT_DIR}'")