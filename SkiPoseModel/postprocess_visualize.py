import pickle as pkl
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import domainadapt_flags
from absl import app, flags
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os

FLAGS = flags.FLAGS

INPUT_FILE = "test_results.pkl"
OUTPUT_LINEARIZED = "test_results_linearized.pkl"

# Dynamically from FLAGS
IDX_SKI_SX = []
IDX_SKI_DX = []

def init_ski_indices():
    """Initialize ski indices from FLAGS"""
    global IDX_SKI_SX, IDX_SKI_DX
    ski_joints = FLAGS.ski_joints if isinstance(FLAGS.ski_joints, list) else list(map(int, FLAGS.ski_joints))
    IDX_SKI_SX = ski_joints[:4]   # Left ski
    IDX_SKI_DX = ski_joints[4:]   # Right ski

CONN_BODY = [
    (1, 0), (1, 8),             # Head-Neck to Pelvis
    (1, 2), (2, 3), (3, 4),     # Right arm
    (1, 5), (5, 6), (6, 7),     # Left arm
    (8, 9), (9, 10), (10, 11),  # Right leg
    (8, 16), (16, 17), (17, 18) # Left leg
]

CONN_SKI_SX = [
    (11, 12), (11, 13),  # Right ankle to ski
    (12, 15),            # Ski structure
    (13, 14)
]

CONN_SKI_DX = [
    (18, 19), (18, 20),  # Left ankle to ski
    (19, 22),            # Ski structure
    (20, 21)
]


def linearize_ski_points(points):
    """
    Linearizes ski points using PCA to fit them to a line.
    """
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


def apply_linearization(raw_preds, gts, create_comparison_plots=True):
    """
    Apply linearization to all ski points in all frames.
    """
    linearized_preds = raw_preds.copy()
    
    print(" Linearizing ski points (PCA)...")
    
    comparison_dir = "comparison_linearization"
    if create_comparison_plots and not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    for i in tqdm(range(len(raw_preds)), desc="Linearizing"):
        ski_sx = raw_preds[i, IDX_SKI_SX, :]
        ski_dx = raw_preds[i, IDX_SKI_DX, :]
        
        lin_sx = linearize_ski_points(ski_sx)
        lin_dx = linearize_ski_points(ski_dx)
        
        linearized_preds[i, IDX_SKI_SX, :] = lin_sx
        linearized_preds[i, IDX_SKI_DX, :] = lin_dx
        
        if create_comparison_plots and i < 20:
            plot_linearization_comparison(gts[i], raw_preds[i], linearized_preds[i], i, comparison_dir)
    
    print(f" Linearization complete. Comparison plots saved in '{comparison_dir}'")
    return linearized_preds


def plot_linearization_comparison(gt, raw_pred, lin_pred, frame_idx, output_dir):
    """
    Plot comparison of GT vs raw prediction vs linearized prediction.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_gt = -gt[:, 1]
    y_raw = -raw_pred[:, 1]
    y_lin = -lin_pred[:, 1]
    
    x_gt, x_raw, x_lin = gt[:, 0], raw_pred[:, 0], lin_pred[:, 0]

    indices = IDX_SKI_SX + IDX_SKI_DX
    
    ax.scatter(x_gt[indices], y_gt[indices], c='green', alpha=0.3, s=50, label='Ground Truth')

    ax.scatter(x_raw[indices], y_raw[indices], c='red', marker='x', alpha=0.6, s=80, label='Raw Prediction')

    sx_idx = IDX_SKI_SX
    ax.plot(x_lin[sx_idx], y_lin[sx_idx], 'b-o', linewidth=2, markersize=6, label='Linearized (Post-Process)')
    
    dx_idx = IDX_SKI_DX
    ax.plot(x_lin[dx_idx], y_lin[dx_idx], 'b-o', linewidth=2, markersize=6)

    ax.set_title(f"Post-Processing Linearization - Frame {frame_idx}")
    ax.legend(loc='upper right')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    filename = os.path.join(output_dir, f"compare_{frame_idx:04d}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()

def plot_single_frame(gt, pred, frame_idx, output_dir):
    """
    Visualize a single frame with GT and predictions overlaid.
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    
    x_gt = gt[:, 0]
    y_gt = -gt[:, 1]  # Flip Y
    
    x_pred = pred[:, 0]
    y_pred = -pred[:, 1]

    for start, end in CONN_BODY:
        ax.plot([x_gt[start], x_gt[end]], [y_gt[start], y_gt[end]], 
                'k-', linewidth=2, alpha=0.6, zorder=1)
    
    ax.scatter([x_gt[11], x_gt[18]], [y_gt[11], y_gt[18]], 
               c='black', s=30, zorder=2, label="Ankles (GT)")

    all_ski_conns = CONN_SKI_SX + CONN_SKI_DX
    for start, end in all_ski_conns:
        ax.plot([x_gt[start], x_gt[end]], [y_gt[start], y_gt[end]], 
                'g-', linewidth=3, alpha=0.8, zorder=3)
    
    ski_indices = IDX_SKI_SX + IDX_SKI_DX
    ax.scatter(x_gt[ski_indices], y_gt[ski_indices], 
               c='green', s=40, marker='o', zorder=4, label="GT Ski Joints")
    
    ax.plot([x_pred[15], x_pred[12]], [y_pred[15], y_pred[12]], 
            'r-', linewidth=3, alpha=0.9, zorder=5, label='Ski L Predicted')
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
    
    filename = os.path.join(output_dir, f"test_{frame_idx:04d}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close(fig)


def visualize_results(preds, gts, output_dir, num_frames=None):
    """
    Generate visualization plots for all frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if num_frames is None:
        num_frames = len(gts)
    else:
        num_frames = min(num_frames, len(gts))
    
    print(f"\n Visualizing {num_frames} frames...")
    
    for i in tqdm(range(num_frames), desc="Visualizing"):
        plot_single_frame(gts[i], preds[i], i, output_dir)
    
    print(f" Visualization complete. Images saved in '{output_dir}'")

def main(argv):
    """Main entry point for post-processing and visualization."""
    init_ski_indices()
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        return False
    
    print("=" * 80)
    print("SKI JUMPING MODEL - POST-PROCESSING & VISUALIZATION")
    print("=" * 80)
    
    print(f"\n Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'rb') as f:
        data = pkl.load(f)
    
    preds = data['pred']
    gts = data['gt']
    
    print(f"    Loaded {len(gts)} frames with {preds.shape[1]} keypoints")
    
    print("\n--- OPTIONS ---")
    do_linearize = input("Apply linearization post-processing? (y/n): ").lower().startswith('y')
    
    if do_linearize:
        linearized_preds = apply_linearization(preds, gts, create_comparison_plots=True)
        
        save_linearized = input("\nSave linearized results to pickle? (y/n): ").lower().startswith('y')
        if save_linearized:
            print(f"\n Saving linearized predictions to {OUTPUT_LINEARIZED}...")
            save_dict = {'pred': linearized_preds, 'gt': gts}
            with open(OUTPUT_LINEARIZED, 'wb') as f:
                pkl.dump(save_dict, f)
            print("    Saved!")
            preds = linearized_preds
    
    do_visualize = input("\nGenerate visualization plots? (y/n): ").lower().startswith('y')
    
    if do_visualize:
        if do_linearize and input("Use 'results_best_linear' output folder? (y/n): ").lower().startswith('y'):
            output_dir = "results"
        else:
            output_dir = "results_tmp"
        
        visualize_results(preds, gts, output_dir, num_frames=100)
    
    print("\n" + "=" * 80)
    print("✅ POST-PROCESSING & VISUALIZATION COMPLETE")
    print("=" * 80 + "\n")
    
    return True


if __name__ == "__main__":
    app.run(main)