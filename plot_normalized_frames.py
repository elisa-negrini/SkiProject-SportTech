import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import glob
import math
import sys

# --- CONFIGURAZIONE ---
CSV_FILE = 'ski_dataset_normalized_adaptive.csv' 
ANNOTATIONS_DIR = os.path.join('dataset', 'annotations')
OUTPUT_DIR = 'sequence_check_adaptive'

# DEFAULT (usato se premi invio senza scrivere nulla)
DEFAULT_JUMP = 'jump6' 
SAVE_INDIVIDUAL_FRAMES = True 

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

def plot_frame_on_ax(ax, row, kpt_names, skeleton_pairs, frame_idx):
    """Disegna un frame su un asse specifico"""
    pts = {}
    
    # 1. Disegna Punti
    for i, name in enumerate(kpt_names):
        col_x = f"kpt_{name}_x"
        col_y = f"kpt_{name}_y"
        
        if col_x in row and not pd.isna(row[col_x]):
            x, y = row[col_x], row[col_y]
            pts[i] = (x, y) 
            
            is_ski = name in ['15', '16', '22', '23', '12', '13', '14', '21', '20', '19']
            color = 'blue' if is_ski else 'red'
            ax.scatter(x, y, c=color, s=40, zorder=5) 

    # 2. Disegna Linee
    if skeleton_pairs:
        for p1_idx, p2_idx in skeleton_pairs:
            i1, i2 = p1_idx - 1, p2_idx - 1
            if i1 in pts and i2 in pts:
                p1, p2 = pts[i1], pts[i2]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', alpha=0.6, lw=2)

    # --- SETUP GRAFICO ---
    ax.set_xlim(-6, 6)
    ax.set_ylim(5, -3) 
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_aspect('equal')

def process_single_jump(df, target_jump):
    """
    Elabora un singolo salto. Ritorna True se OK, False se fallisce.
    """
    print(f"\n--- Elaborazione salto: {target_jump} ---")
    
    jump_data = df[df['jump_id'] == target_jump].copy()
    
    # Ordinamento robusto
    try:
        jump_data['sort_key'] = jump_data['frame_name'].apply(lambda x: int(''.join(filter(str.isdigit, str(x)))))
        jump_data = jump_data.sort_values('sort_key')
    except:
        jump_data = jump_data.sort_values('frame_name')
    
    if jump_data.empty:
        print(f"⚠️  ATTENZIONE: Nessun dato trovato nel CSV per {target_jump}. Salto ignorato.")
        return False

    # Carica scheletro
    kpt_names, skeleton = load_skeleton_structure(target_jump)
    if not kpt_names:
        cols = [c for c in df.columns if c.startswith('kpt_') and c.endswith('_x')]
        kpt_names = [c.replace('kpt_', '').replace('_x', '') for c in cols]

    # Crea cartelle output
    jump_out_dir = os.path.join(OUTPUT_DIR, target_jump)
    frames_out_dir = os.path.join(jump_out_dir, "frames_individuali")
    
    if not os.path.exists(jump_out_dir): os.makedirs(jump_out_dir)
    if SAVE_INDIVIDUAL_FRAMES and not os.path.exists(frames_out_dir): os.makedirs(frames_out_dir)

    # --- 1. SALVATAGGIO FRAME SINGOLI ---
    if SAVE_INDIVIDUAL_FRAMES:
        print(f"  > Generazione frame singoli in {frames_out_dir} ...")
        for idx, (i, row) in enumerate(jump_data.iterrows()):
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_frame_on_ax(ax, row, kpt_names, skeleton, idx)
            ax.set_title(f"{target_jump} | {row['frame_name']}")
            
            fname = os.path.splitext(row['frame_name'])[0]
            out_path = os.path.join(frames_out_dir, f"{fname}_norm.png")
            plt.savefig(out_path)
            plt.close(fig) 
            if idx % 10 == 0: print(f"    Plot {idx}/{len(jump_data)}...", end='\r')
        print("\n  > Frame singoli completati.")

    # --- 2. SALVATAGGIO GRIGLIA ---
    print("  > Generazione griglia riassuntiva...")
    num_frames = len(jump_data)
    cols = 8
    rows = math.ceil(num_frames / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    axes = axes.flatten()
    
    for idx, (i, row) in enumerate(jump_data.iterrows()):
        plot_frame_on_ax(axes[idx], row, kpt_names, skeleton, idx)
        axes[idx].set_title(f"{idx}", fontsize=8)
        axes[idx].set_xticklabels([])
        axes[idx].set_yticklabels([])
        
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    grid_path = os.path.join(jump_out_dir, f"{target_jump}_grid_adaptive.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    
    print(f"✅ Completato: {target_jump}")
    return True

def main():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Errore: File {CSV_FILE} non trovato.")
        return

    print(f"Lettura CSV: {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    input_str = ""

    # 1. Controlla se ci sono argomenti da riga di comando (priorità)
    if len(sys.argv) > 1:
        input_str = sys.argv[1]
    else:
        # 2. Se non ci sono argomenti, CHIEDI ALL'UTENTE
        print("-" * 50)
        print(f"Inserisci quali salti processare.")
        print(f"  - Esempio singolo:    5")
        print(f"  - Esempio intervallo: 5-12")
        print(f"  - Premi INVIO vuoto per usare il default ({DEFAULT_JUMP})")
        print("-" * 50)
        input_str = input(">> Scelta: ").strip()

    # 3. Parsing dell'input
    jumps_to_process = []
    
    if not input_str:
        # Utente ha premuto invio vuoto
        jumps_to_process = [DEFAULT_JUMP]
    else:
        try:
            if '-' in input_str:
                # Caso intervallo: 5-22
                parts = input_str.split('-')
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                jumps_to_process = [f"jump{i}" for i in range(start, end + 1)]
            else:
                # Caso singolo
                if input_str.isdigit():
                    jumps_to_process = [f"jump{input_str}"]
                else:
                    jumps_to_process = [input_str] # Caso 'jump5' esplicito
        except ValueError:
            print("❌ Errore: Formato non valido. Riprova usando numeri (es. 5) o intervalli (es. 5-10).")
            return

    print(f"Lista salti da elaborare: {jumps_to_process}")

    # --- CICLO DI ELABORAZIONE ---
    success_count = 0
    fail_count = 0

    for jump_id in jumps_to_process:
        result = process_single_jump(df, jump_id)
        if result:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "="*30)
    print(f"FINE ELABORAZIONE.")
    print(f"Successi: {success_count}")
    print(f"Falliti/Non trovati: {fail_count}")
    print("="*30)

if __name__ == "__main__":
    main()