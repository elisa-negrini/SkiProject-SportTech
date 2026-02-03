import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# CONFIGURAZIONE
INPUT_FILE = "test_results_full.pkl"
OUTPUT_DIR = "results_mask_cav_long_run"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- DEFINIZIONE INDICI E CONNESSIONI ---
# Mappatura: 0=Collo, 1=Testa, 11=CavigliaSX, 18=CavigliaDX

# INDICI PUNTI SCI (Solo quelli PREDETTI dal modello)
# NOTA: Abbiamo tolto 11 e 18 perché sono caviglie (Corpo) e non vengono predette
IDX_SKI_SX = [12, 13, 14, 15] # PuntaScarpa, Tallone, CodaSci, PuntaSci
IDX_SKI_DX = [19, 20, 21, 22] 

# Connessioni Corpo (Disegnate in NERO)
CONN_BODY = [
    (1, 0), (1, 8),            # Testa-Collo-Bacino
    (1, 2), (2, 3), (3, 4),    # Braccio DX
    (1, 5), (5, 6), (6, 7),    # Braccio SX
    (8, 9), (9, 10), (10, 11), # Gamba SX (fino alla caviglia 11)
    (8, 16), (16, 17), (17, 18) # Gamba DX (fino alla caviglia 18)
]

# Connessioni Sci (Disegnate in VERDE/ROSSO)
# Includiamo le connessioni che partono dalla caviglia (11/18) verso lo sci
CONN_SKI_SX = [
    (11, 12), (11, 13), # Caviglia -> Punta/Tallone Scarpa
    (12, 15),           # Punta Scarpa -> Punta Sci
    (13, 14)            # Tallone Scarpa -> Coda Sci
]

CONN_SKI_DX = [
    (18, 19), (18, 20), # Caviglia -> Punta/Tallone Scarpa
    (19, 22),           # Punta Scarpa -> Punta Sci
    (20, 21)            # Tallone Scarpa -> Coda Sci
]

def plot_single_frame(gt, pred, frame_idx):
    fig, ax = plt.subplots(figsize=(8, 10))
    
    x_gt = gt[:, 0]
    y_gt = -gt[:, 1] # Inversione Y
    
    x_pred = pred[:, 0]
    y_pred = -pred[:, 1]

    # 1. DISEGNA CORPO (SOLO GT - NERO)
    for start, end in CONN_BODY:
        ax.plot([x_gt[start], x_gt[end]], [y_gt[start], y_gt[end]], 'k-', linewidth=2, alpha=0.6, zorder=1)
    
    # Pallini caviglie (Essendo corpo, li facciamo neri o blu, NON rossi)
    ax.scatter([x_gt[11], x_gt[18]], [y_gt[11], y_gt[18]], c='black', s=30, zorder=2, label="Caviglie (Fisse)")

    # 2. DISEGNA SCI GROUND TRUTH (VERDE)
    all_ski_conns = CONN_SKI_SX + CONN_SKI_DX
    for start, end in all_ski_conns:
        ax.plot([x_gt[start], x_gt[end]], [y_gt[start], y_gt[end]], 
                'g-', linewidth=3, alpha=0.8, zorder=3)
    
    # Punti sci GT (Verdi) - UNA SOLA VOLTA
    ski_indices = IDX_SKI_SX + IDX_SKI_DX
    ax.scatter(x_gt[ski_indices], y_gt[ski_indices], 
               c='green', s=40, marker='o', zorder=4, label="GT Sci")
    
    # 3. DISEGNA SCI PREDETTI COME LINEE RETTE
    # Sci SX: traccia retta da coda (14) a punta (15)
    ski_sx_pred_pts = pred[[12, 13, 14, 15], :] # Get all predicted points for left ski
    coda_sx_pred = ski_sx_pred_pts[2] # Predicted CodaSci
    punta_sx_pred = ski_sx_pred_pts[3] # Predicted PuntaSci
    
    ax.plot([coda_sx_pred[0], punta_sx_pred[0]], [-coda_sx_pred[1], -punta_sx_pred[1]], 
            'r-', linewidth=3, alpha=0.9, zorder=5, label='Sci SX Predetto')
    
    # Sci DX
    ski_dx_pred_pts = pred[[19, 20, 21, 22], :] # Get all predicted points for right ski
    coda_dx_pred = ski_dx_pred_pts[2] # Predicted CodaSci
    punta_dx_pred = ski_dx_pred_pts[3] # Predicted PuntaSci
    
    ax.plot([coda_dx_pred[0], punta_dx_pred[0]], [-coda_dx_pred[1], -punta_dx_pred[1]], 
            'orange', linewidth=3, alpha=0.9, zorder=5, label='Sci DX Predetto')
    
    # Punti predetti - UNA SOLA VOLTA (solo per gli indici dello sci, non le caviglie)
    ax.scatter(x_pred[ski_indices], y_pred[ski_indices], 
               c='red', s=40, marker='x', zorder=6, label="Punti Predetti")

    # 4. NUOVE CONNESSIONI: CAVIGLIA GT ALLA PUNTA/TALLONE SCARPA PREDETTI (MAGENTA)
    # Caviglia SX (GT) a Punta Scarpa SX (Predetta)
    ax.plot([x_gt[11], x_pred[12]], [y_gt[11], y_pred[12]], 
            'm--', linewidth=2, alpha=0.7, zorder=5, label='GT Caviglia -> Pred Scarpa')
    # Caviglia SX (GT) a Tallone Scarpa SX (Predetto)
    ax.plot([x_gt[11], x_pred[13]], [y_gt[11], y_pred[13]], 
            'm--', linewidth=2, alpha=0.7, zorder=5)

    # Caviglia DX (GT) a Punta Scarpa DX (Predetta)
    ax.plot([x_gt[18], x_pred[19]], [y_gt[18], y_pred[19]], 
            'm--', linewidth=2, alpha=0.7, zorder=5)
    # Caviglia DX (GT) a Tallone Scarpa DX (Predetto)
    ax.plot([x_gt[18], x_pred[20]], [y_gt[18], y_pred[20]], 
            'm--', linewidth=2, alpha=0.7, zorder=5)


    ax.set_title(f"Test Frame: {frame_idx}")
    ax.legend(loc='upper right')
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([]) 
    
    filename = os.path.join(OUTPUT_DIR, f"test_{frame_idx:04d}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Errore: Non trovo {INPUT_FILE}. Esegui prima il test del modello!")
        return

    print(f"📂 Caricamento {INPUT_FILE}...")
    with open(INPUT_FILE, 'rb') as f:
        data = pkl.load(f)
    
    preds = data['pred']
    gts = data['gt']
    
    num_to_plot = min(100, len(gts))  # massimo 100 immagini, o meno se ci sono meno frame
    print(f"📸 Generazione di {num_to_plot} immagini...")

    for i in tqdm(range(num_to_plot)):
        plot_single_frame(gts[i], preds[i], i)
        
    print(f"\n✅ Finito! Controlla la cartella: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()