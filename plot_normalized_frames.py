import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
import math

# --- CONFIGURAZIONE ---
CSV_FILE = 'ski_dataset_normalized_adaptive.csv' 
ANNOTATIONS_DIR = os.path.join('dataset', 'annotations')
OUTPUT_DIR = 'sequence_check_adaptive'

# Se None, plotta il primo salto nel CSV. 
# Se vuoi uno specifico, metti es: 'JP0009' (o 'jump9' a seconda di come è salvato nel CSV)
JUMP_TO_PLOT = 'jump9' 

# Metti True se vuoi salvare ogni singolo frame come immagine separata
SAVE_INDIVIDUAL_FRAMES = True 

def load_skeleton_structure(jump_id):
    """Cerca il file JSON originale per recuperare le connessioni."""
    # Pattern di ricerca più robusto: cerca jump_id nel nome file o nella cartella
    # Pulisci jump_id per cercare file (es. da JP0006 a jump6 se necessario, o viceversa)
    # Qui cerchiamo genericamente *jump_id* nel path
    pattern = os.path.join(ANNOTATIONS_DIR, "**", f"*{jump_id}*.json")
    files = glob.glob(pattern, recursive=True)
    
    json_file = None
    # Priorità ai file interpolati
    for f in files:
        if 'interpolated' in f:
            json_file = f
            break
    if not json_file and files:
        json_file = files[0] # Fallback
        
    if not json_file:
        return [], []

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Cerca la categoria giusta
            for cat in data['categories']:
                if 'keypoints' in cat and len(cat['keypoints']) > 0:
                    return cat['keypoints'], cat.get('skeleton', [])
    except Exception:
        return [], []
    return [], []

def plot_frame_on_ax(ax, row, kpt_names, skeleton_pairs, frame_idx):
    """Disegna un frame su un asse specifico (riutilizzabile per grid e single plot)"""
    pts = {}
    
    # 1. Disegna Punti
    for i, name in enumerate(kpt_names):
        col_x = f"kpt_{name}_x"
        col_y = f"kpt_{name}_y"
        
        if col_x in row and not pd.isna(row[col_x]):
            x, y = row[col_x], row[col_y]
            pts[i] = (x, y) 
            
            # Colore Blu per sci, Rosso per corpo
            is_ski = name in ['15', '16', '22', '23', '12', '13', '14', '21', '20', '19']
            color = 'blue' if is_ski else 'red'
            ax.scatter(x, y, c=color, s=40, zorder=5) # s=40 punti più grandi

    # 2. Disegna Linee
    if skeleton_pairs:
        for p1_idx, p2_idx in skeleton_pairs:
            i1, i2 = p1_idx - 1, p2_idx - 1
            if i1 in pts and i2 in pts:
                p1, p2 = pts[i1], pts[i2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.6, lw=2)

    # --- SETUP GRAFICO (Unità Busto) ---
    ax.set_xlim(-6, 6)
    ax.set_ylim(5, -3) # Y invertita
    
    # Assi cartesiani centrati su Bacino (0,0)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')

def main():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Errore: File {CSV_FILE} non trovato. Esegui prima lo script di normalizzazione.")
        return

    print(f"Lettura CSV: {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    target_jump = JUMP_TO_PLOT
    if target_jump is None:
        if df.empty:
            print("CSV vuoto.")
            return
        # Prende il primo ID disponibile
        target_jump = df['jump_id'].unique()[0]
    
    print(f"Generazione sequenza per: {target_jump}")
    # Filtra e ordina per nome frame (assumendo contenga numeri ordinabili o sia già ordinato)
    jump_data = df[df['jump_id'] == target_jump]
    
    # Ordinamento robusto basato sui numeri nel nome file
    try:
        jump_data['sort_key'] = jump_data['frame_name'].apply(lambda x: int(''.join(filter(str.isdigit, str(x)))))
        jump_data = jump_data.sort_values('sort_key')
    except:
        jump_data = jump_data.sort_values('frame_name')
    
    if jump_data.empty:
        print(f"❌ Nessun dato per {target_jump}")
        return

    # Carica scheletro
    kpt_names, skeleton = load_skeleton_structure(target_jump)
    if not kpt_names:
        # Fallback nomi colonne
        cols = [c for c in df.columns if c.startswith('kpt_') and c.endswith('_x')]
        kpt_names = [c.replace('kpt_', '').replace('_x', '') for c in cols]

    # Crea cartelle output
    jump_out_dir = os.path.join(OUTPUT_DIR, target_jump)
    frames_out_dir = os.path.join(jump_out_dir, "frames_individuali")
    
    if not os.path.exists(jump_out_dir): os.makedirs(jump_out_dir)
    if SAVE_INDIVIDUAL_FRAMES and not os.path.exists(frames_out_dir): os.makedirs(frames_out_dir)

    print(f"Salvataggio output in: {jump_out_dir}")

    # --- 1. SALVATAGGIO FRAME SINGOLI ---
    if SAVE_INDIVIDUAL_FRAMES:
        print(f"Generating single frames in {frames_out_dir} ...")
        for idx, (i, row) in enumerate(jump_data.iterrows()):
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_frame_on_ax(ax, row, kpt_names, skeleton, idx)
            ax.set_title(f"{target_jump} | {row['frame_name']}")
            
            # Salva
            fname = os.path.splitext(row['frame_name'])[0]
            out_path = os.path.join(frames_out_dir, f"{fname}_norm.png")
            plt.savefig(out_path)
            plt.close(fig) # Chiudi per liberare memoria
            
            if idx % 10 == 0: print(f"  Plot {idx}/{len(jump_data)}...", end='\r')
        print("\nFrame singoli completati.")

    # --- 2. SALVATAGGIO GRIGLIA (RIASSUNTO) ---
    print("Generazione griglia riassuntiva...")
    num_frames = len(jump_data)
    cols = 8
    rows = math.ceil(num_frames / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    axes = axes.flatten()
    
    for idx, (i, row) in enumerate(jump_data.iterrows()):
        plot_frame_on_ax(axes[idx], row, kpt_names, skeleton, idx)
        axes[idx].set_title(f"{idx}", fontsize=8)
        # Rimuovi etichette assi per pulizia nella griglia
        axes[idx].set_xticklabels([])
        axes[idx].set_yticklabels([])
        
    # Spegni assi vuoti
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    grid_path = os.path.join(jump_out_dir, f"{target_jump}_grid_adaptive.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    
    print(f"✅ Fatto! Griglia salvata: {grid_path}")

if __name__ == "__main__":
    main()