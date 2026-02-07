import pickle as pkl
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

INPUT_FILE = "test_results_full.pkl"
OUTPUT_FILE = "test_results_linearized.pkl"
VIS_DIR = "comparison_linearization"

if not os.path.exists(VIS_DIR):
    os.makedirs(VIS_DIR)

IDX_SKI_SX = [12, 13, 14, 15] 
IDX_SKI_DX = [19, 20, 21, 22] 

def linearize_ski_points(points):
    
    if points.shape[0] != 4:
        return points 

    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    pca = PCA(n_components=1)
    pca.fit(centered_points)
    
    direction_vector = pca.components_[0]
    projections = np.dot(centered_points, direction_vector)
    
    linearized_centered = np.outer(projections, direction_vector)
    linearized_points = linearized_centered + centroid
    
    return linearized_points

def plot_comparison(gt, raw_pred, lin_pred, frame_idx):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_gt = -gt[:, 1]
    y_raw = -raw_pred[:, 1]
    y_lin = -lin_pred[:, 1]
    
    x_gt, x_raw, x_lin = gt[:, 0], raw_pred[:, 0], lin_pred[:, 0]

    indices = IDX_SKI_SX + IDX_SKI_DX
    ax.scatter(x_gt[indices], y_gt[indices], c='green', alpha=0.3, label='Ground Truth')

    ax.scatter(x_raw[indices], y_raw[indices], c='red', marker='x', alpha=0.6, label='Raw Prediction (Model)')

    sx_idx = IDX_SKI_SX
    ax.plot(x_lin[sx_idx], y_lin[sx_idx], 'b-o', linewidth=2, label='Linearized (Post-Process)')
    
    dx_idx = IDX_SKI_DX
    ax.plot(x_lin[dx_idx], y_lin[dx_idx], 'b-o', linewidth=2)

    ax.set_title(f"Post-Processing Linearization - Frame {frame_idx}")
    ax.legend()
    ax.axis('equal')
    
    plt.savefig(os.path.join(VIS_DIR, f"compare_{frame_idx:04d}.png"))
    plt.close()

def main():
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return

    with open(INPUT_FILE, 'rb') as f:
        data = pkl.load(f)

    raw_preds = data['pred']
    gts = data['gt']
    
    linearized_preds = raw_preds.copy()
    
    print("(PCA)...")
    
    for i in tqdm(range(len(raw_preds))):
        ski_sx = raw_preds[i, IDX_SKI_SX, :]
        ski_dx = raw_preds[i, IDX_SKI_DX, :]
        
        lin_sx = linearize_ski_points(ski_sx)
        lin_dx = linearize_ski_points(ski_dx)
        
        linearized_preds[i, IDX_SKI_SX, :] = lin_sx
        linearized_preds[i, IDX_SKI_DX, :] = lin_dx
        
        if i < 20:
            plot_comparison(gts[i], raw_preds[i], linearized_preds[i], i)

    print(f"Saved in {OUTPUT_FILE}...")
    save_dict = {'pred': linearized_preds, 'gt': gts}
    
    with open(OUTPUT_FILE, 'wb') as f:
        pkl.dump(save_dict, f)
    
if __name__ == "__main__":
    main()