import json
import numpy as np
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt
import random

# --- CONFIGURAZIONE ---
DATASET_DIR = os.path.join('dataset', 'annotations')
OUTPUT_CSV = 'ski_dataset_normalized_adaptive.csv' # Nome aggiornato per distinguerlo
PLOT_OUTPUT_DIR = 'plots_verifica_adaptive'

# Punti Cardine (ID come stringhe nel JSON) per calcolare la scala
ANCHOR_IDS = {
    'neck': '2',        
    'r_shoulder': '3', 
    'l_shoulder': '6',  
    'r_hip': '17',      
    'l_hip': '10'       
}

# --- FUNZIONI CORE ---

def get_point(kpts, idx):
    """Estrae (x, y) se l'indice è valido e v > 0"""
    if idx < 0: return None
    start = idx * 3
    if start + 2 >= len(kpts): return None
    x, y, v = kpts[start], kpts[start+1], kpts[start+2]
    # Roboflow a volte usa v=2 per visibile. Consideriamo valido se v > 0.
    if v == 0: return None
    return np.array([x, y])

def get_category_with_keypoints(data):
    """Trova la categoria che contiene la definizione dei keypoints"""
    for cat in data['categories']:
        if 'keypoints' in cat and len(cat['keypoints']) > 0:
            return cat
    return None

def normalize_annotation_adaptive(ann, keypoint_names):
    if 'keypoints' not in ann: return None
    kpts = ann['keypoints']
    
    # 1. Trova gli indici dei punti cardine
    idx_map = {}
    for key, target_id in ANCHOR_IDS.items():
        try:
            idx_map[key] = keypoint_names.index(target_id)
        except ValueError:
            idx_map[key] = -1 

    # 2. Recupera coordinate cardine
    neck = get_point(kpts, idx_map['neck'])
    r_shoulder = get_point(kpts, idx_map['r_shoulder'])
    l_shoulder = get_point(kpts, idx_map['l_shoulder'])
    r_hip = get_point(kpts, idx_map['r_hip'])
    l_hip = get_point(kpts, idx_map['l_hip'])

    # 3. Calcola Riferimenti (Centro e Scala)
    
    # --- A. CENTRO (ROOT) ---
    root_ref = None
    if r_hip is not None and l_hip is not None:
        root_ref = (r_hip + l_hip) / 2.0
    elif r_hip is not None: root_ref = r_hip
    elif l_hip is not None: root_ref = l_hip
    
    if root_ref is None:
        return None # Impossibile trovare il centro

    # --- B. SCALA ADATTIVA (MAX tra Busto e Spalle) ---
    
    # Stima Collo (Top)
    top_ref = neck
    if top_ref is None and r_shoulder is not None and l_shoulder is not None:
        top_ref = (r_shoulder + l_shoulder) / 2.0
        
    # Calcolo Lunghezza Busto (Torso Length)
    torso_len = 0.0
    if top_ref is not None:
        torso_len = np.linalg.norm(top_ref - root_ref)
        
    # Calcolo Larghezza Spalle (Shoulder Width)
    shoulder_len = 0.0
    if r_shoulder is not None and l_shoulder is not None:
        shoulder_len = np.linalg.norm(r_shoulder - l_shoulder)
        
    # LOGICA ADATTIVA:
    # Se il busto è schiacciato (vista frontale), shoulder_len sarà maggiore -> usiamo quello.
    # Se siamo di lato, torso_len sarà maggiore -> usiamo quello.
    # Se uno dei due manca, usiamo l'altro.
    scale_factor = max(torso_len, shoulder_len)
    
    # Protezione divisione per zero
    if scale_factor < 1e-6: 
        return None

    # 4. Normalizzazione "Body Units" (Pura)
    # Origine (0,0) = Bacino.
    # Unità 1 = La dimensione massima del corpo visibile.
    
    normalized_kpts = {}
    for i, name in enumerate(keypoint_names):
        pt = get_point(kpts, i)
        
        col_x = f"kpt_{name}_x"
        col_y = f"kpt_{name}_y"
        
        if pt is not None:
            # Formula Pura: Distanza dal centro / Fattore di Scala
            # Nessuno shift (+0.5) e nessun moltiplicatore arbitrario (*7.0)
            vec = (pt - root_ref) / scale_factor
            
            normalized_kpts[col_x] = vec[0]
            normalized_kpts[col_y] = vec[1]
        else:
            normalized_kpts[col_x] = np.nan
            normalized_kpts[col_y] = np.nan
            
    return normalized_kpts

# --- PLOTTING ---
def plot_skeleton_check(row, keypoint_names, skeleton_pairs, output_path):
    plt.figure(figsize=(6, 6))
    
    pts = {}
    for i, name in enumerate(keypoint_names):
        col_x = f"kpt_{name}_x"
        col_y = f"kpt_{name}_y"
        
        if col_x in row and not pd.isna(row[col_x]):
            x, y = row[col_x], row[col_y]
            pts[i] = (x, y) 
            
            # Colore Blu per gli sci, Rosso per il resto
            is_ski = name in ['15', '16', '22', '23', '13', '14', '21', '20'] 
            c = 'blue' if is_ski else 'red'
            plt.scatter(x, y, c=c, s=20)

    if skeleton_pairs:
        for p1_idx, p2_idx in skeleton_pairs:
            i1 = p1_idx - 1
            i2 = p2_idx - 1
            if i1 in pts and i2 in pts:
                p1 = pts[i1]
                p2 = pts[i2]
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.7)

    # Setup Assi "Fisici" centrati sullo 0
    # Adattiamo la vista per vedere tutto (es. da -5 a +5 unità corpo)
    plt.xlim(-6, 6)
    plt.ylim(5, -3) # Y invertita
    
    # Linee di riferimento (Croce sul bacino)
    plt.axhline(0, color='black', linewidth=1, alpha=0.5)
    plt.axvline(0, color='black', linewidth=1, alpha=0.5)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f"Adaptive Check: {row['jump_id']} - {row['frame_name']}")
    plt.xlabel("Body Units (Width)")
    plt.ylabel("Body Units (Height)")
    plt.savefig(output_path)
    plt.close()
    print(f"   --> Plot salvato: {output_path}")

# --- MAIN LOOP ---
def main():
    if not os.path.exists(PLOT_OUTPUT_DIR): os.makedirs(PLOT_OUTPUT_DIR)
    
    # Cerca ricorsivamente i file interpolati
    search_pattern = os.path.join(DATASET_DIR, "**", "*interpolated*.coco.json")
    files = glob.glob(search_pattern, recursive=True)
    
    print(f"Trovati {len(files)} file JSON.")
    all_data = []
    
    for json_file in files:
        folder = os.path.basename(os.path.dirname(json_file))
        fname = os.path.basename(json_file)
        
        # Estrazione ID Salto (File > Cartella)
        match = re.search(r'(JP\d+|jump\d+)', fname, re.IGNORECASE)
        if not match:
            match = re.search(r'(JP\d+|jump\d+)', folder, re.IGNORECASE)
        jump_id = match.group(1) if match else folder
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            cat = get_category_with_keypoints(data)
            if cat is None:
                print(f"SALTATO {fname}: Nessuna categoria 'skier' trovata.")
                continue
                
            kpt_names = cat['keypoints'] 
            skeleton = cat.get('skeleton', []) 
            
            img_map = {img['id']: img['file_name'] for img in data['images']}
            
            jump_frames = []
            for ann in data['annotations']:
                if ann['category_id'] != cat['id']:
                    continue

                # USA LA NUOVA FUNZIONE ADAPTIVE
                norm = normalize_annotation_adaptive(ann, kpt_names)
                
                if norm:
                    row = {
                        'jump_id': jump_id,
                        'frame_name': img_map.get(ann['image_id'], 'unknown')
                    }
                    row.update(norm)
                    jump_frames.append(row)
            
            all_data.extend(jump_frames)
            
            # Plot di verifica casuale
            if jump_frames:
                sample = random.choice(jump_frames)
                plot_name = f"check_{jump_id}.png"
                plot_skeleton_check(sample, kpt_names, skeleton, os.path.join(PLOT_OUTPUT_DIR, plot_name))
                
        except Exception as e:
            print(f"Errore su {fname}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ FINITO. CSV salvato: {OUTPUT_CSV}")
        print(f"✅ PLOT salvati in: {PLOT_OUTPUT_DIR}/")
        
        ids = df['jump_id'].unique()
        print(f"ID Salti rilevati: {ids[:5]} ... (Totale: {len(ids)})")
    else:
        print("Nessun dato valido.")

if __name__ == "__main__":
    main()