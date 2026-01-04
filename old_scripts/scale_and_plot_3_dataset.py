import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import sys
import glob
import json

# --- CONFIGURAZIONE ---
INPUT_FILES = {
    'skeleton_sum':   'ski_dataset_skeleton_sum_normalized.csv',
    'torso_shoulder': 'ski_dataset_torso_shoulder_normalized.csv',
    'unit_norm':      'ski_dataset_unit_normalized.csv'
}
BASE_OUTPUT_DIR = 'old_plots'
ANNOTATIONS_DIR = os.path.join('dataset', 'annotations')

# PARAMETRI DI SCALING (0-1)
# Per i metodi fisici, l'unità è "lunghezza corpo". 
# Un box da -1.5 a +1.5 copre braccia alzate e sci lunghi.
LIMIT_PHYSICAL = 1.5 
# Per unit norm, i valori sono tipicamente piccoli (norma 1).
LIMIT_UNIT = 1.0     

def load_skeleton_structure(jump_id):
    """Recupera la struttura per disegnare le linee."""
    pattern = os.path.join(ANNOTATIONS_DIR, "**", f"*{jump_id}*.json")
    files = glob.glob(pattern, recursive=True)
    skel = []
    # Cerca skeleton nel primo file valido
    for f in files:
        if 'interpolated' in f:
            try:
                with open(f) as jf:
                    d = json.load(jf)
                    for c in d['categories']:
                        if 'skeleton' in c: return c['keypoints'], c['skeleton']
            except: pass
    return [], []

def scale_dataframe_01(df, limit):
    """Porta i dati dal range [-limit, +limit] a [0, 1]."""
    cols = [c for c in df.columns if c.startswith('kpt_') and (c.endswith('_x') or c.endswith('_y'))]
    df_scaled = df.copy()
    
    # Clipping per sicurezza (evita crash grafici)
    df_scaled[cols] = df_scaled[cols].clip(-limit, limit)
    
    # Formula: (val + limit) / (2*limit) -> sposta 0 a 0.5
    df_scaled[cols] = (df_scaled[cols] + limit) / (2 * limit)
    return df_scaled

def plot_jump_grid(df, jump_id, method_name, out_folder):
    # Dati del salto
    jump_data = df[df['jump_id'] == jump_id].copy()
    if jump_data.empty: return

    # Ordinamento frame
    try:
        jump_data['sort'] = jump_data['frame_name'].apply(lambda x: int(''.join(filter(str.isdigit, str(x)))))
        jump_data = jump_data.sort_values('sort')
    except:
        jump_data = jump_data.sort_values('frame_name')

    # Scheletro
    kpt_names, skeleton = load_skeleton_structure(jump_id)
    if not kpt_names: # Fallback se non trova il json
        cols = [c for c in df.columns if 'kpt_' in c and '_x' in c]
        kpt_names = [c.replace('kpt_','').replace('_x','') for c in cols]

    # Setup Griglia
    n_frames = len(jump_data)
    cols_grid = 8
    rows_grid = math.ceil(n_frames / cols_grid)
    
    fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(cols_grid*2, rows_grid*2))
    axes = axes.flatten()

    for idx, ( _, row) in enumerate(jump_data.iterrows()):
        ax = axes[idx]
        pts = {}
        # Plot Punti
        for i, name in enumerate(kpt_names):
            cx, cy = f"kpt_{name}_x", f"kpt_{name}_y"
            if cx in row and not pd.isna(row[cx]):
                x, y = row[cx], row[cy]
                pts[i] = (x, y)
                is_ski = name in ['15','16','22','23','12','13','14','21','20','19'] # Adatta se necessario
                ax.scatter(x, y, s=10, c='blue' if is_ski else 'red')
        
        # Plot Linee
        if skeleton:
            for p1, p2 in skeleton:
                if (p1-1) in pts and (p2-1) in pts:
                    ax.plot([pts[p1-1][0], pts[p2-1][0]], [pts[p1-1][1], pts[p2-1][1]], 'gray', lw=1, alpha=0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0) # Y invertita
        ax.axis('off')
        ax.set_title(str(idx), fontsize=6)
        
        # Centro Croce (0.5, 0.5)
        ax.axhline(0.5, c='green', alpha=0.1)
        ax.axvline(0.5, c='green', alpha=0.1)

    # Pulisci assi vuoti
    for j in range(idx+1, len(axes)): axes[j].axis('off')
    
    save_path = os.path.join(out_folder, f"{jump_id}_grid.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"    --> Salvato: {save_path}")

def main():
    # Input salti da terminale
    jumps_arg = []
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if '-' in arg:
            s, e = map(int, arg.split('-'))
            jumps_arg = [f"jump{i}" for i in range(s, e+1)]
        else:
            jumps_arg = [f"jump{arg}" if arg.isdigit() else arg]
    else:
        print("Usa: python scale_and_plot.py 5-10  (o jump5)")
        print("Defaulting to jump6 per test.")
        jumps_arg = ['jump6']

    # Ciclo sui 3 dataset
    for method, csv_file in INPUT_FILES.items():
        if not os.path.exists(csv_file):
            print(f"⚠️ Manca {csv_file}, salto.")
            continue
            
        print(f"\n🔵 Elaborazione Metodo: {method.upper()}")
        df = pd.read_csv(csv_file)
        
        # Scaling 0-1
        limit = LIMIT_UNIT if method == 'unit_norm' else LIMIT_PHYSICAL
        df_scaled = scale_dataframe_01(df, limit)
        
        # Plotting
        for jump in jumps_arg:
            # Crea cartella specifica: plots/metodo/salto/
            out_folder = os.path.join(BASE_OUTPUT_DIR, method, jump)
            os.makedirs(out_folder, exist_ok=True)
            
            plot_jump_grid(df_scaled, jump, method, out_folder)

if __name__ == "__main__":
    main()