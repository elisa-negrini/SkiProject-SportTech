import json
import numpy as np
import pandas as pd
import os
import glob
import re

# --- CONFIGURAZIONE ---
DATASET_DIR = os.path.join('dataset', 'annotations')

# Nomi dei file di output
OUT_FILE_1 = 'ski_dataset_skeleton_sum_normalized.csv'       # Schiena + Coscia + Gamba
OUT_FILE_2 = 'ski_dataset_torso_shoulder_normalized.csv'   # Schiena + Gamba + Spalle
OUT_FILE_3 = 'ski_dataset_unit_normalized.csv'             # Unit Norm (Statistica)

# ID Keypoints (COCO standard per Roboflow)
# 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder, 6:RShoulder
# 7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist, 11:LHip, 12:RHip
# 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
# Nota: Adatta questi ID se il tuo JSON usa stringhe o indici diversi.
# Qui uso i nomi stringa basati sui tuoi script precedenti.
ANCHOR_IDS = {
    'neck': '2',          # Collo/Nose (o punto alto busto)
    'r_shoulder': '3', 'l_shoulder': '6',
    'r_hip': '17',     'l_hip': '10',
    'r_knee': '18',    'l_knee': '11',
    'r_ankle': '19',   'l_ankle': '12'
}

def get_point(kpts, name, kpt_names):
    """Recupera (x, y) dato il nome del keypoint."""
    try:
        idx = kpt_names.index(name)
    except ValueError:
        return None
    
    start = idx * 3
    if start + 2 >= len(kpts): return None
    x, y, v = kpts[start], kpts[start+1], kpts[start+2]
    if v == 0: return None
    return np.array([x, y])

def get_dist(p1, p2):
    if p1 is None or p2 is None: return 0.0
    return np.linalg.norm(p1 - p2)

def process_frame(ann, kpt_names, img_name, jump_id):
    kpts = ann['keypoints']
    
    # 1. Recupera Punti Chiave
    pts = {k: get_point(kpts, v, kpt_names) for k, v in ANCHOR_IDS.items()}
    
    # Calcolo Centro Bacino (Root)
    root = None
    if pts['r_hip'] is not None and pts['l_hip'] is not None:
        root = (pts['r_hip'] + pts['l_hip']) / 2.0
    elif pts['r_hip'] is not None: root = pts['r_hip']
    elif pts['l_hip'] is not None: root = pts['l_hip']
    
    if root is None: return None, None, None # Frame invalido senza bacino

    # --- CALCOLO LUNGHEZZE SEGMENTI ---
    # Torso (Collo -> Bacino)
    torso_top = pts['neck'] if pts['neck'] is not None else (
        (pts['r_shoulder'] + pts['l_shoulder'])/2 if (pts['r_shoulder'] is not None and pts['l_shoulder'] is not None) else None
    )
    len_torso = get_dist(torso_top, root)
    if len_torso == 0: len_torso = 1.0 # Fallback minimo

    # Gambe (Media Destra/Sinistra per stabilità)
    len_thigh_r = get_dist(pts['r_hip'], pts['r_knee'])
    len_shin_r  = get_dist(pts['r_knee'], pts['r_ankle'])
    len_leg_r = len_thigh_r + len_shin_r
    
    len_thigh_l = get_dist(pts['l_hip'], pts['l_knee'])
    len_shin_l  = get_dist(pts['l_knee'], pts['l_ankle'])
    len_leg_l = len_thigh_l + len_shin_l
    
    # Usa la gamba più lunga visibile (spesso una è nascosta o piegata)
    len_leg_max = max(len_leg_r, len_leg_l)
    if len_leg_max == 0: len_leg_max = len_torso * 1.2 # Fallback euristico

    # Spalle
    len_shoulders = get_dist(pts['r_shoulder'], pts['l_shoulder'])

    # --- DEFINIZIONE SCALE ---
    
    # SCALA 1: Skeleton Sum (Schiena + Gamba)
    scale_1 = len_torso + len_leg_max
    
    # SCALA 2: Torso + Shoulder (Schiena + Gamba + Spalle)
    scale_2 = len_torso + len_leg_max + len_shoulders

    # SCALA 3: Unit Norm (Statistica)
    # Centra tutti i punti sulla media (Baricentro geometrico, non bacino)
    all_points = []
    for i in range(0, len(kpts), 3):
        if kpts[i+2] > 0: # Se visibile
            all_points.append([kpts[i], kpts[i+1]])
    all_points = np.array(all_points)
    
    if len(all_points) < 3: return None, None, None
    
    mean_center = np.mean(all_points, axis=0)
    centered_pts = all_points - mean_center
    # Norma di Frobenius (radice somma quadrati)
    scale_3 = np.linalg.norm(centered_pts) 
    if scale_3 == 0: scale_3 = 1.0

    # --- CREAZIONE RIGHE NORMALIZZATE ---
    row_common = {'jump_id': jump_id, 'frame_name': img_name}
    
    data_1, data_2, data_3 = row_common.copy(), row_common.copy(), row_common.copy()

    for i, name in enumerate(kpt_names):
        pt = get_point(kpts, name, kpt_names)
        col_x, col_y = f"kpt_{name}_x", f"kpt_{name}_y"
        
        if pt is not None:
            # Metodo 1 & 2: Origine = Bacino (root)
            vec = pt - root
            data_1[col_x], data_1[col_y] = vec[0]/scale_1, vec[1]/scale_1
            data_2[col_x], data_2[col_y] = vec[0]/scale_2, vec[1]/scale_2
            
            # Metodo 3: Origine = Baricentro (mean_center)
            vec_stat = pt - mean_center
            data_3[col_x], data_3[col_y] = vec_stat[0]/scale_3, vec_stat[1]/scale_3
        else:
            for d in [data_1, data_2, data_3]:
                d[col_x], d[col_y] = np.nan, np.nan

    return data_1, data_2, data_3

def main():
    files = glob.glob(os.path.join(DATASET_DIR, "**", "*interpolated*.coco.json"), recursive=True)
    print(f"Trovati {len(files)} file JSON.")

    rows_1, rows_2, rows_3 = [], [], []

    for json_file in files:
        folder = os.path.basename(os.path.dirname(json_file))
        fname = os.path.basename(json_file)
        # Estrazione ID (es. JP05 o jump5)
        match = re.search(r'(JP\d+|jump\d+)', fname, re.IGNORECASE) or re.search(r'(JP\d+|jump\d+)', folder, re.IGNORECASE)
        jump_id = match.group(1) if match else folder

        try:
            with open(json_file, 'r') as f: data = json.load(f)
            
            # Trova categoria con keypoints
            cat = next((c for c in data['categories'] if 'keypoints' in c), None)
            if not cat: continue
            
            kpt_names = cat['keypoints']
            img_map = {img['id']: img['file_name'] for img in data['images']}

            for ann in data['annotations']:
                if ann['category_id'] != cat['id']: continue
                
                r1, r2, r3 = process_frame(ann, kpt_names, img_map.get(ann['image_id'], 'unk'), jump_id)
                if r1:
                    rows_1.append(r1)
                    rows_2.append(r2)
                    rows_3.append(r3)

        except Exception as e:
            print(f"Errore {fname}: {e}")

    # Salvataggio
    pd.DataFrame(rows_1).to_csv(OUT_FILE_1, index=False)
    pd.DataFrame(rows_2).to_csv(OUT_FILE_2, index=False)
    pd.DataFrame(rows_3).to_csv(OUT_FILE_3, index=False)

    print("\n✅ Generazione completata:")
    print(f"  1. Skeleton Sum (Back+Leg) -> {OUT_FILE_1}")
    print(f"  2. Torso+Shoulder          -> {OUT_FILE_2}")
    print(f"  3. Unit Norm (Statistica)  -> {OUT_FILE_3}")

if __name__ == "__main__":
    main()