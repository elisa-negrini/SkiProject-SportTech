import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
import math
import sys

# --- CONFIGURAZIONE ---
CSV_FILE = 'ski_dataset_final_scaled.csv' 
ANNOTATIONS_DIR = os.path.join('dataset', 'annotations')
OUTPUT_DIR = 'plot_scaled_jumps'

# DEFAULT
DEFAULT_JUMP = 'jump6' 

def load_skeleton_structure(jump_id):
    """Cerca il file JSON originale per recuperare le connessioni."""
    pattern = os.path.join(ANNOTATIONS_DIR, "**", f"*{jump_id}*.json")
    files = glob.glob(pattern, recursive=True)
    
    json_file = None
    for f in files:
        if 'interpolated' in f:
            json_file = f
            break
    if not json_file and files:
        json_file = files[0] 
        
    if not json_file:
        return [], []

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            for cat in data['categories']:
                if 'keypoints' in cat and len(cat['keypoints']) > 0:
                    return cat['keypoints'], cat.get('skeleton', [])
    except Exception:
        return [], []
    return [], []

def plot_frame_on_ax(ax, row, kpt_names, skeleton_pairs):
    """Disegna un frame su un asse specifico in coordinate 0-1"""
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
            ax.scatter(x, y, c=color, s=20, zorder=5) 

    # 2. Disegna Linee
    if skeleton_pairs:
        for p1_idx, p2_idx in skeleton_pairs:
            i1, i2 = p1_idx - 1, p2_idx - 1
            if i1 in pts and i2 in pts:
                p1, p2 = pts[i1], pts[i2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.6, lw=1)

    # --- SETUP GRAFICO 0-1 ---
    # Impostiamo i limiti fissi del "box" normalizzato
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0) # Invertiamo Y perché nelle immagini (0,0) è in alto a sinistra.
    
    # Centro (Bacino)
    ax.axhline(0.5, color='green', linewidth=0.5, alpha=0.3)
    ax.axvline(0.5, color='green', linewidth=0.5, alpha=0.3)
    
    ax.grid(True, linestyle='--', alpha=0.2)
    ax.set_aspect('equal')
    # Rimuovi tick per pulizia
    ax.set_xticks([])
    ax.set_yticks([])

def process_single_jump(df, target_jump):
    print(f"\n--- Generazione plot per: {target_jump} ---")
    
    jump_data = df[df['jump_id'] == target_jump].copy()
    
    # Ordinamento
    try:
        jump_data['sort_key'] = jump_data['frame_name'].apply(lambda x: int(''.join(filter(str.isdigit, str(x)))))
        jump_data = jump_data.sort_values('sort_key')
    except:
        jump_data = jump_data.sort_values('frame_name')
    
    if jump_data.empty:
        print(f"⚠️  Dati non trovati per {target_jump}.")
        return False

    # Carica scheletro
    kpt_names, skeleton = load_skeleton_structure(target_jump)
    if not kpt_names:
        cols = [c for c in df.columns if c.startswith('kpt_') and c.endswith('_x')]
        kpt_names = [c.replace('kpt_', '').replace('_x', '') for c in cols]

    # Crea cartelle output
    jump_out_dir = os.path.join(OUTPUT_DIR, target_jump)
    if not os.path.exists(jump_out_dir): os.makedirs(jump_out_dir)

    # --- SALVATAGGIO GRIGLIA RIASSUNTIVA ---
    print(f"  > Creazione griglia in {jump_out_dir}...")
    num_frames = len(jump_data)
    cols = 8
    rows = math.ceil(num_frames / cols)
    
    # Calcola dimensione figura in base al numero di righe
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()
    
    for idx, (i, row) in enumerate(jump_data.iterrows()):
        plot_frame_on_ax(axes[idx], row, kpt_names, skeleton)
        axes[idx].set_title(f"{idx}", fontsize=6)
        
    # Spegni assi vuoti
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    grid_path = os.path.join(jump_out_dir, f"{target_jump}_scaled_grid.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    
    print(f"✅ Salvato: {grid_path}")
    return True

def main():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Errore: Esegui prima 'scale_data.py' per generare {CSV_FILE}.")
        return

    print(f"Lettura CSV Scalato: {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    # --- INPUT UTENTE ---
    input_str = ""
    if len(sys.argv) > 1:
        input_str = sys.argv[1]
    else:
        print("-" * 50)
        print(f"Inserisci quali salti verificare.")
        print(f"  - Esempio:    5")
        print(f"  - Intervallo: 5-10")
        print(f"  - Invio:      Default ({DEFAULT_JUMP})")
        print("-" * 50)
        input_str = input(">> Scelta: ").strip()

    jumps_to_process = []
    if not input_str:
        jumps_to_process = [DEFAULT_JUMP]
    else:
        try:
            if '-' in input_str:
                start, end = map(int, input_str.split('-'))
                jumps_to_process = [f"jump{i}" for i in range(start, end + 1)]
            elif input_str.isdigit():
                jumps_to_process = [f"jump{input_str}"]
            else:
                jumps_to_process = [input_str]
        except ValueError:
            print("❌ Formato non valido.")
            return

    # ESECUZIONE
    for jump_id in jumps_to_process:
        process_single_jump(df, jump_id)

if __name__ == "__main__":
    main()