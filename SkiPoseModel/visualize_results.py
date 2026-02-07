import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

INPUT_FILE = "test_results_full.pkl"
OUTPUT_DIR = "results_best_lr0001"

# INPUT_FILE = "test_results_linearized.pkl"
# OUTPUT_DIR = "results_best_linear"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

IDX_SKI_SX = [12, 13, 14, 15] 
IDX_SKI_DX = [19, 20, 21, 22] 

CONN_BODY = [
    (1, 0), (1, 8),            
    (1, 2), (2, 3), (3, 4),    
    (1, 5), (5, 6), (6, 7),   
    (8, 9), (9, 10), (10, 11), 
    (8, 16), (16, 17), (17, 18) 
]

CONN_SKI_SX = [
    (11, 12), (11, 13), 
    (12, 15),         
    (13, 14)  
]

CONN_SKI_DX = [
    (18, 19), (18, 20), 
    (19, 22),          
    (20, 21)       
]

def plot_single_frame(gt, pred, frame_idx):
    fig, ax = plt.subplots(figsize=(8, 10))
    
    x_gt = gt[:, 0]
    y_gt = -gt[:, 1] 
    
    x_pred = pred[:, 0]
    y_pred = -pred[:, 1]

    for start, end in CONN_BODY:
        ax.plot([x_gt[start], x_gt[end]], [y_gt[start], y_gt[end]], 'k-', linewidth=2, alpha=0.6, zorder=1)
    
    ax.scatter([x_gt[11], x_gt[18]], [y_gt[11], y_gt[18]], c='black', s=30, zorder=2, label="Ankles (GT)")

    all_ski_conns = CONN_SKI_SX + CONN_SKI_DX
    for start, end in all_ski_conns:
        ax.plot([x_gt[start], x_gt[end]], [y_gt[start], y_gt[end]], 
                'g-', linewidth=3, alpha=0.8, zorder=3)
    
    ski_indices = IDX_SKI_SX + IDX_SKI_DX
    ax.scatter(x_gt[ski_indices], y_gt[ski_indices], 
               c='green', s=40, marker='o', zorder=4, label="GT Ski Joints")
    
    
    ax.plot([x_pred[15], x_pred[12]], [y_pred[15], y_pred[12]], 
            'r-', linewidth=3, alpha=0.9, zorder=5, label='Sci L Predicted')
    ax.plot([x_pred[12], x_pred[13]], [y_pred[12], y_pred[13]], 
            'r-', linewidth=3, alpha=0.9, zorder=5)
    ax.plot([x_pred[13], x_pred[14]], [y_pred[13], y_pred[14]], 
            'r-', linewidth=3, alpha=0.9, zorder=5)

    ax.plot([x_pred[22], x_pred[19]], [y_pred[22], y_pred[19]], 
            'orange', linewidth=3, alpha=0.9, zorder=5, label='Sci R Predicted')
    ax.plot([x_pred[19], x_pred[20]], [y_pred[19], y_pred[20]], 
            'orange', linewidth=3, alpha=0.9, zorder=5)
    ax.plot([x_pred[20], x_pred[21]], [y_pred[20], y_pred[21]], 
            'orange', linewidth=3, alpha=0.9, zorder=5)

    ski_indices = IDX_SKI_SX + IDX_SKI_DX
    ax.scatter(x_pred[ski_indices], y_pred[ski_indices], 
            c='red', s=40, marker='x', zorder=6, label="Predicted")
    
    ax.plot([x_gt[11], x_pred[12]], [y_gt[11], y_pred[12]], 
            'm--', linewidth=2, alpha=0.7, zorder=5, label='GT Ankle -> Pred Shoe')
    ax.plot([x_gt[11], x_pred[13]], [y_gt[11], y_pred[13]], 
            'm--', linewidth=2, alpha=0.7, zorder=5)

    ax.plot([x_gt[18], x_pred[19]], [y_gt[18], y_pred[19]], 
            'm--', linewidth=2, alpha=0.7, zorder=5)
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
        print(f"❌ Error: not found {INPUT_FILE}")
        return

    print(f"📂  {INPUT_FILE}...")
    with open(INPUT_FILE, 'rb') as f:
        data = pkl.load(f)
    
    preds = data['pred']
    gts = data['gt']
    
    num_to_plot = min(100, len(gts))
    print(f"{num_to_plot} frames to plot...")

    for i in tqdm(range(num_to_plot)):
        plot_single_frame(gts[i], preds[i], i)
        
    print(f"\n Done! Check folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()