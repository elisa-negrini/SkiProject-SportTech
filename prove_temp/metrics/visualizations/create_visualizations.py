"""
Ski Jumping Metrics Visualization Suite
=======================================

This script generates comprehensive visualizations for ski jumping analysis.

VISUALIZATIONS GENERATED (by priority):
---------------------------------------
1. Frame Overlays (PRIORITY 1)
   - Skeleton + angle measurements on original frames
   
2. Feature Importance (PRIORITY 1)
   - Bar charts from ML models
   
3. Metric Distributions (PRIORITY 2)
   - Histograms for each metric
   
4. Correlation Plots (PRIORITY 2)
   - Scatterplots: metrics vs scores
   
5. Heatmaps (PRIORITY 3)
   - Correlation matrix between metrics
   
6. Clustering Comparison (PRIORITY 3)
   - DTW vs CDTW visualization
   
7. Timelines (PRIORITY 4)
   - Per-jump metric evolution

Output: metrics/visualizations/
All images are in .gitignore (not committed to git)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import cv2
from typing import Dict, List, Optional, Tuple
import warnings
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class VisualizationSuite:
    """
    Comprehensive visualization generator for ski jumping metrics.
    """
    
    # Keypoint mapping (same as metrics_calculator)
    KPT_MAP = {
        'head': '1', 'neck': '2', 'center_pelvis': '9',
        'r_shoulder': '3', 'l_shoulder': '6',
        'r_hip': '17', 'l_hip': '10',
        'r_knee': '18', 'l_knee': '11',
        'r_ankle': '19', 'l_ankle': '12',
        'r_ski_tip': '23', 'r_ski_tail': '22',
        'l_ski_tip': '16', 'l_ski_tail': '15'
    }
    
    # Skeleton connections for drawing
    SKELETON_CONNECTIONS = [
        ('head', 'neck'),
        ('neck', 'r_shoulder'), ('neck', 'l_shoulder'),
        ('r_shoulder', 'r_hip'), ('l_shoulder', 'l_hip'),
        ('center_pelvis', 'r_hip'), ('center_pelvis', 'l_hip'),
        ('r_hip', 'r_knee'), ('l_hip', 'l_knee'),
        ('r_knee', 'r_ankle'), ('l_knee', 'l_ankle'),
        ('r_ankle', 'r_ski_tip'), ('r_ankle', 'r_ski_tail'),
        ('l_ankle', 'l_ski_tip'), ('l_ankle', 'l_ski_tail'),
    ]
    
    def __init__(self):
        """Initialize paths and load data."""
        self.metrics_path = Path(__file__).parent.parent  # metrics/
        self.project_root = self.metrics_path.parent  # project root
        
        # Data paths (dataset is in project root, not metrics)
        self.dataset_path = self.project_root / 'dataset'
        self.frames_path = self.dataset_path / 'frames'
        self.annotations_path = self.dataset_path / 'annotations'
        
        # Metrics paths
        self.timeseries_file = self.metrics_path / 'timeseries_metrics' / 'timeseries_summary.csv'
        self.core_metrics_file = self.metrics_path / 'core_metrics' / 'metrics_summary_per_jump.csv'
        self.core_per_frame_file = self.metrics_path / 'core_metrics' / 'metrics_per_frame.csv'
        self.correlations_file = self.metrics_path / 'correlations' / 'correlations.csv'
        self.merged_file = self.metrics_path / 'correlations' / 'merged_scores_metrics.csv'
        
        # Output paths
        self.output_dir = self.metrics_path / 'visualizations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.frame_overlay_dir = self.output_dir / 'frame_overlays'
        self.distributions_dir = self.output_dir / 'distributions'
        self.correlations_dir = self.output_dir / 'correlations'
        self.timelines_dir = self.output_dir / 'timelines'
        self.importance_dir = self.output_dir / 'importance'
        
        for d in [self.frame_overlay_dir, self.distributions_dir, 
                  self.correlations_dir, self.timelines_dir, self.importance_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_metrics(self) -> Dict[str, pd.DataFrame]:
        """Load all metrics dataframes."""
        data = {}
        
        if self.timeseries_file.exists():
            data['timeseries'] = pd.read_csv(self.timeseries_file)
            print(f"✅ Loaded timeseries: {len(data['timeseries'])} jumps")
        
        if self.core_metrics_file.exists():
            data['core'] = pd.read_csv(self.core_metrics_file)
            print(f"✅ Loaded core metrics: {len(data['core'])} jumps")
        
        if self.core_per_frame_file.exists():
            data['per_frame'] = pd.read_csv(self.core_per_frame_file)
            print(f"✅ Loaded per-frame metrics: {len(data['per_frame'])} rows")
        
        if self.merged_file.exists():
            data['merged'] = pd.read_csv(self.merged_file)
            print(f"✅ Loaded merged data: {len(data['merged'])} jumps")
        
        if self.correlations_file.exists():
            data['correlations'] = pd.read_csv(self.correlations_file)
            print(f"✅ Loaded correlations")
        
        return data
    
    def load_coco_annotations(self, jump_id: str) -> Optional[dict]:
        """Load COCO annotations for a specific jump."""
        # Try normalized annotations first
        jump_num = int(jump_id.replace('JP', ''))
        ann_file = self.annotations_path / jump_id / 'train' / f'annotations_normalized_jump{jump_num}.coco.json'
        
        if not ann_file.exists():
            # Try interpolated
            ann_file = self.annotations_path / jump_id / 'train' / f'annotations_interpolated_jump{jump_num}.coco.json'
        
        if not ann_file.exists():
            print(f"⚠️ No annotations found for {jump_id}")
            return None
        
        with open(ann_file, 'r') as f:
            return json.load(f)
    
    # =========================================================================
    # PRIORITY 1: FRAME OVERLAYS
    # =========================================================================
    
    def create_frame_overlay(self, jump_id: str, frame_name: str, 
                            keypoints: np.ndarray, metrics: dict,
                            img_width: int, img_height: int) -> np.ndarray:
        """
        Create a frame overlay with skeleton and metric annotations.
        
        Args:
            jump_id: Jump identifier (e.g., 'JP0009')
            frame_name: Frame filename (e.g., '00476.jpg')
            keypoints: Array of (x, y, visibility) normalized coordinates
            metrics: Dict of metric values for this frame
            img_width, img_height: Frame dimensions
        
        Returns:
            Annotated image array
        """
        # Load original frame
        frame_path = self.frames_path / jump_id / frame_name
        
        if not frame_path.exists():
            # Try alternative naming
            frame_path = self.frames_path / jump_id / frame_name.replace('.jpg', '.png')
        
        if not frame_path.exists():
            print(f"⚠️ Frame not found: {frame_path}")
            return None
        
        img = cv2.imread(str(frame_path))
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Parse keypoints (format: x, y, v, x, y, v, ...)
        kpt_dict = {}
        kpt_names = list(self.KPT_MAP.keys())
        
        for i, name in enumerate(kpt_names):
            idx = i * 3
            if idx + 2 < len(keypoints):
                x = keypoints[idx] * img_width
                y = keypoints[idx + 1] * img_height
                v = keypoints[idx + 2]
                if v > 0:  # Visible
                    kpt_dict[name] = (int(x), int(y))
        
        # Draw skeleton
        for p1_name, p2_name in self.SKELETON_CONNECTIONS:
            if p1_name in kpt_dict and p2_name in kpt_dict:
                p1 = kpt_dict[p1_name]
                p2 = kpt_dict[p2_name]
                cv2.line(img, p1, p2, (0, 255, 0), 2)
        
        # Draw keypoints
        for name, (x, y) in kpt_dict.items():
            color = (255, 0, 0)  # Red for body
            if 'ski' in name:
                color = (0, 0, 255)  # Blue for skis
            cv2.circle(img, (x, y), 5, color, -1)
        
        # Draw BSA angle if we have the points
        if all(k in kpt_dict for k in ['neck', 'center_pelvis', 'r_ski_tip', 'l_ski_tip', 'r_ski_tail', 'l_ski_tail',
                                         'r_shoulder', 'l_shoulder', 'r_hip', 'l_hip', 'r_knee', 'l_knee']):
            
            # RED VECTOR: Ski direction (from ski tail midpoint to ski tip midpoint)
            ski_tip_r = np.array(kpt_dict['r_ski_tip'], dtype=np.float32)
            ski_tip_l = np.array(kpt_dict['l_ski_tip'], dtype=np.float32)
            ski_tail_r = np.array(kpt_dict['r_ski_tail'], dtype=np.float32)
            ski_tail_l = np.array(kpt_dict['l_ski_tail'], dtype=np.float32)
            
            ski_tip_mid = (ski_tip_r + ski_tip_l) / 2
            ski_tail_mid = (ski_tail_r + ski_tail_l) / 2
            
            cv2.arrowedLine(img, tuple(ski_tail_mid.astype(int)), tuple(ski_tip_mid.astype(int)), 
                            (0, 0, 255), 3, tipLength=0.1)  # Red
            
            # BLUE VECTOR: Body regression line (neck, shoulders, pelvis, hips, knees)
            body_points = np.array([
                kpt_dict['neck'],
                (np.array(kpt_dict['r_shoulder'], dtype=np.float32) + np.array(kpt_dict['l_shoulder'], dtype=np.float32)) / 2,
                kpt_dict['center_pelvis'],
                (np.array(kpt_dict['r_hip'], dtype=np.float32) + np.array(kpt_dict['l_hip'], dtype=np.float32)) / 2,
                (np.array(kpt_dict['r_knee'], dtype=np.float32) + np.array(kpt_dict['l_knee'], dtype=np.float32)) / 2
            ], dtype=np.float32)
            
            # Fit a line using least squares regression
            x_coords = body_points[:, 0]
            y_coords = body_points[:, 1]
            z = np.polyfit(x_coords, y_coords, 1)
            p = np.poly1d(z)
            
            # Get start and end points for the line (extend from top to bottom of body)
            x_min, x_max = body_points[:, 0].min(), body_points[:, 0].max()
            y_start = p(x_min)
            y_end = p(x_max)
            
            cv2.arrowedLine(img, tuple([int(x_min), int(y_start)]), tuple([int(x_max), int(y_end)]), 
                            (255, 0, 0), 3, tipLength=0.1)  # Blue
            
            # Calculate angle between ski vector and body vector
            ski_vector = ski_tip_mid - ski_tail_mid
            body_vector = np.array([x_max - x_min, y_end - y_start])
            
            cos_angle = np.dot(ski_vector, body_vector) / (np.linalg.norm(ski_vector) * np.linalg.norm(body_vector) + 1e-6)
            angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
            angle_deg = np.degrees(angle_rad)
            
            # Add angle text comparing calculated vs CSV value
            mid_point = (ski_tip_mid + ski_tail_mid) / 2
            csv_value = metrics.get('body_ski_inclination', None)
            if pd.notna(csv_value):
                text = f"BSA Calc: {angle_deg:.1f}° | CSV: {csv_value:.1f}°"
            else:
                text = f"BSA Calc: {angle_deg:.1f}°"
            cv2.putText(img, text, tuple(mid_point.astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add V-style angle if available
        if 'avg_v_style_front' in metrics and pd.notna(metrics.get('avg_v_style_front')):
            cv2.putText(img, f"V-Style: {metrics['avg_v_style_front']:.1f}°", 
                      (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (255, 200, 0), 2)
        
        return img
    
    def generate_frame_overlays(self, jump_ids: List[str] = None, max_frames: int = 5):
        """
        Generate frame overlay visualizations for specified jumps.
        
        Args:
            jump_ids: List of jump IDs to process (None = all available)
            max_frames: Maximum frames per jump to generate
        """
        print("\n" + "=" * 60)
        print("GENERATING FRAME OVERLAYS (Priority 1)")
        print("=" * 60)
        
        if jump_ids is None:
            # Get all available jumps
            jump_ids = [d.name for d in self.annotations_path.iterdir() 
                       if d.is_dir() and d.name.startswith('JP')]
        
        # Load per-frame metrics if available
        df_frames = None
        if self.core_per_frame_file.exists():
            df_frames = pd.read_csv(self.core_per_frame_file)
        
        for jump_id in sorted(jump_ids)[:10]:  # Limit to first 10 jumps
            print(f"\n📸 Processing {jump_id}...")
            
            coco = self.load_coco_annotations(jump_id)
            if coco is None:
                continue
            
            # Create jump output directory
            jump_output = self.frame_overlay_dir / jump_id
            jump_output.mkdir(exist_ok=True)
            
            # Get image info
            images = {img['id']: img for img in coco['images']}
            
            # Process annotations
            frame_count = 0
            for ann_idx, ann in enumerate(coco['annotations'][:max_frames]):
                img_id = ann['image_id']
                if img_id not in images:
                    continue
                
                img_info = images[img_id]
                frame_name = img_info.get('extra', {}).get('name', img_info['file_name'])
                
                # Get metrics for this frame using frame_idx
                metrics = {}
                if df_frames is not None:
                    frame_metrics = df_frames[
                        (df_frames['jump_id'] == jump_id) & 
                        (df_frames['frame_idx'] == ann_idx)
                    ]
                    if len(frame_metrics) > 0:
                        metrics = frame_metrics.iloc[0].to_dict()
                
                # Create overlay
                img = self.create_frame_overlay(
                    jump_id, frame_name,
                    ann['keypoints'],
                    metrics,
                    img_info['width'],
                    img_info['height']
                )
                
                if img is not None:
                    output_path = jump_output / f"{frame_name.split('.')[0]}_overlay.png"
                    plt.figure(figsize=(12, 8))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"{jump_id} - {frame_name}", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    frame_count += 1
            
            print(f"   ✅ Generated {frame_count} overlays")
    
    # =========================================================================
    # PRIORITY 1: FEATURE IMPORTANCE
    # =========================================================================
    
    def generate_feature_importance_plots(self):
        """Generate feature importance bar charts from ML models."""
        print("\n" + "=" * 60)
        print("GENERATING FEATURE IMPORTANCE PLOTS (Priority 1)")
        print("=" * 60)
        
        models_dir = self.metrics_path / 'models'
        
        for target in ['Style_Score', 'Physical_Score']:
            for model in ['Random_Forest', 'Gradient_Boosting']:
                imp_file = models_dir / f'importance_{target}_{model}.csv'
                
                if not imp_file.exists():
                    continue
                
                df_imp = pd.read_csv(imp_file)
                
                # Determine which importance column to use
                if 'importance_builtin' in df_imp.columns:
                    imp_col = 'importance_builtin'
                elif 'importance_perm_mean' in df_imp.columns:
                    imp_col = 'importance_perm_mean'
                elif 'importance' in df_imp.columns:
                    imp_col = 'importance'
                else:
                    print(f"   ⚠️ No importance column in {imp_file.name}")
                    continue
                
                # Take top 15 features
                df_top = df_imp.nlargest(15, imp_col)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 8))
                
                colors = sns.color_palette("viridis", len(df_top))
                bars = ax.barh(range(len(df_top)), df_top[imp_col], color=colors)
                
                ax.set_yticks(range(len(df_top)))
                ax.set_yticklabels(df_top['feature'])
                ax.invert_yaxis()
                ax.set_xlabel('Importance', fontsize=12)
                ax.set_title(f'{target} - {model.replace("_", " ")}\nTop 15 Features', fontsize=14)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, df_top[imp_col])):
                    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', fontsize=9)
                
                plt.tight_layout()
                output_path = self.importance_dir / f'{target}_{model}_importance.png'
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Saved: {output_path.name}")
    
    # =========================================================================
    # PRIORITY 2: DISTRIBUTIONS
    # =========================================================================
    
    def generate_distribution_plots(self, data: Dict[str, pd.DataFrame]):
        """Generate histograms for all metrics."""
        print("\n" + "=" * 60)
        print("GENERATING DISTRIBUTION PLOTS (Priority 2)")
        print("=" * 60)
        
        if 'merged' not in data:
            print("⚠️ No merged data available")
            return
        
        df = data['merged']
        
        # Get numeric columns (excluding IDs and scores)
        exclude = ['jump_id', 'Style_Score', 'Physical_Score', 'AthleteScore', 
                   'AthleteDistance', 'HillHS', 'HillK']
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                       if c not in exclude]
        
        # Create multi-panel figure
        n_cols = 4
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            values = df[col].dropna()
            
            if len(values) < 3:
                ax.text(0.5, 0.5, f'{col}\n(insufficient data)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            ax.hist(values, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'mean={values.mean():.2f}')
            ax.set_title(col, fontsize=10)
            ax.set_xlabel('')
            ax.legend(fontsize=8)
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Metric Distributions', fontsize=16, y=1.02)
        plt.tight_layout()
        
        output_path = self.distributions_dir / 'all_distributions.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {output_path}")
    
    # =========================================================================
    # PRIORITY 2: CORRELATION SCATTERPLOTS
    # =========================================================================
    
    def generate_correlation_scatterplots(self, data: Dict[str, pd.DataFrame]):
        """Generate scatterplots of metrics vs scores."""
        print("\n" + "=" * 60)
        print("GENERATING CORRELATION SCATTERPLOTS (Priority 2)")
        print("=" * 60)
        
        if 'merged' not in data:
            return
        
        df = data['merged']
        
        # Key metrics to plot
        key_metrics = [
            'flight_std', 'flight_jitter', 'landing_hip_velocity',
            'knee_peak_velocity', 'avg_body_ski_angle', 'avg_v_style_front',
            'telemark_scissor_mean', 'telemark_stability'
        ]
        
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        for target in ['Style_Score', 'Physical_Score']:
            if target not in df.columns:
                continue
            
            n_metrics = len(available_metrics)
            n_cols = 4
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
            axes = axes.flatten()
            
            for i, metric in enumerate(available_metrics):
                ax = axes[i]
                
                # Get valid pairs
                mask = df[metric].notna() & df[target].notna()
                x = df.loc[mask, metric]
                y = df.loc[mask, target]
                
                if len(x) < 3:
                    ax.text(0.5, 0.5, f'{metric}\n(insufficient data)', 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Scatter plot
                ax.scatter(x, y, alpha=0.6, s=60, edgecolor='white', linewidth=0.5)
                
                # Fit line
                
                r, p = pearsonr(x, y)
                z = np.polyfit(x, y, 1)
                line_x = np.linspace(x.min(), x.max(), 100)
                ax.plot(line_x, np.poly1d(z)(line_x), 'r--', linewidth=2)
                
                ax.set_xlabel(metric, fontsize=10)
                ax.set_ylabel(target, fontsize=10)
                ax.set_title(f'r={r:.2f}, p={p:.3f}', fontsize=10)
            
            # Hide empty
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Metrics vs {target}', fontsize=16, y=1.02)
            plt.tight_layout()
            
            output_path = self.correlations_dir / f'{target}_scatterplots.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved: {output_path}")
    
    # =========================================================================
    # PRIORITY 3: HEATMAP
    # =========================================================================
    
    def generate_correlation_heatmap(self, data: Dict[str, pd.DataFrame]):
        """Generate correlation matrix heatmap."""
        print("\n" + "=" * 60)
        print("GENERATING CORRELATION HEATMAP (Priority 3)")
        print("=" * 60)
        
        if 'merged' not in data:
            return
        
        df = data['merged']
        
        # Select numeric columns
        exclude = ['jump_id']
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                       if c not in exclude]
        
        # Compute correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 14))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                   vmin=-1, vmax=1, center=0, square=True,
                   linewidths=0.5, cbar_kws={'shrink': 0.8})
        
        plt.title('Metric Correlation Matrix', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        output_path = self.correlations_dir / 'correlation_heatmap.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {output_path}")
    
    # =========================================================================
    # PRIORITY 4: TIMELINES
    # =========================================================================
    
    def generate_timeline_plots(self, data: Dict[str, pd.DataFrame], 
                               jump_ids: List[str] = None):
        """Generate per-jump metric evolution plots."""
        print("\n" + "=" * 60)
        print("GENERATING TIMELINE PLOTS (Priority 4)")
        print("=" * 60)
        
        if 'per_frame' not in data:
            print("⚠️ No per-frame data available")
            return
        
        df = data['per_frame']
        
        if jump_ids is None:
            jump_ids = df['jump_id'].unique()[:5]  # First 5 jumps
        
        for jump_id in jump_ids:
            df_jump = df[df['jump_id'] == jump_id].sort_values('frame_idx')
            
            if len(df_jump) < 5:
                continue
            
            # Create 3-panel figure
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # BSA
            if 'body_ski_angle' in df_jump.columns:
                ax = axes[0]
                values = df_jump['body_ski_angle'].dropna()
                if len(values) > 0:
                    ax.plot(range(len(values)), values, 'b-', linewidth=2)
                    ax.fill_between(range(len(values)), 5, 25, alpha=0.2, color='green', label='Optimal')
                    ax.set_ylabel('Body-Ski Angle (°)')
                    ax.legend()
                    ax.grid(alpha=0.3)
            
            # V-style
            if 'v_style_angle_front' in df_jump.columns:
                ax = axes[1]
                values = df_jump['v_style_angle_front'].dropna()
                if len(values) > 0:
                    ax.plot(range(len(values)), values, 'r-', linewidth=2)
                    ax.set_ylabel('V-Style Angle (°)')
                    ax.grid(alpha=0.3)
            
            # Telemark
            telemark_col = None
            for col in ['telemark_offset_x_raw', 'telemark_leg_angle']:
                if col in df_jump.columns:
                    telemark_col = col
                    break
            
            if telemark_col:
                ax = axes[2]
                values = df_jump[telemark_col].dropna()
                if len(values) > 0:
                    ax.plot(range(len(values)), values, 'g-', linewidth=2)
                    ax.set_ylabel(telemark_col)
                    ax.set_xlabel('Frame')
                    ax.grid(alpha=0.3)
            
            plt.suptitle(f'{jump_id} - Metric Evolution', fontsize=14)
            plt.tight_layout()
            
            output_path = self.timelines_dir / f'{jump_id}_timeline.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved: {output_path.name}")
    
    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================
    
    def run(self, priorities: List[int] = None):
        """
        Run visualization generation.
        
        Args:
            priorities: List of priorities to generate (1-4). None = all
        """
        print("=" * 60)
        print("SKI JUMPING VISUALIZATION SUITE")
        print("=" * 60)
        
        # Load data
        data = self.load_metrics()
        
        if priorities is None:
            priorities = [1, 2, 3, 4]
        
        # Priority 1
        if 1 in priorities:
            self.generate_frame_overlays()
            self.generate_feature_importance_plots()
        
        # Priority 2
        if 2 in priorities:
            self.generate_distribution_plots(data)
            self.generate_correlation_scatterplots(data)
        
        # Priority 3
        if 3 in priorities:
            self.generate_correlation_heatmap(data)
        
        # Priority 4
        if 4 in priorities:
            self.generate_timeline_plots(data)
        
        print("\n" + "=" * 60)
        print("VISUALIZATION COMPLETE")
        print(f"Output: {self.output_dir}")
        print("=" * 60)
    
    def run_interactive(self):
        """
        Interactive menu for selecting which visualizations to generate.
        """
        print("\n" + "=" * 60)
        print("SKI JUMPING VISUALIZATION SUITE - INTERACTIVE MODE")
        print("=" * 60)
        
        print("\nAvailable visualizations:\n")
        print("  1. Frame Overlays + Feature Importance (PRIORITY 1)")
        print("     - Skeleton overlays with angle measurements")
        print("     - ML model feature importance charts\n")
        
        print("  2. Metric Distributions + Correlation Scatterplots (PRIORITY 2)")
        print("     - Histograms for each metric")
        print("     - Metrics vs scores scatterplots\n")
        
        print("  3. Correlation Heatmap (PRIORITY 3)")
        print("     - Correlation matrix between all metrics\n")
        
        print("  4. Timeline Plots (PRIORITY 4)")
        print("     - Per-jump metric evolution over frames\n")
        
        print("  5. ALL (runs everything)\n")
        
        while True:
            user_input = input("Select option(s) [1-5 or comma-separated, e.g., '1,3,5']: ").strip()
            
            if user_input == "5":
                selected = [1, 2, 3, 4]
                print("\n✓ Running ALL visualizations...\n")
                break
            
            try:
                selected = []
                for item in user_input.split(','):
                    item = item.strip()
                    if item in ['1', '2', '3', '4']:
                        selected.append(int(item))
                    elif item == '5':
                        selected = [1, 2, 3, 4]
                        break
                    else:
                        raise ValueError("Invalid option")
                
                if selected:
                    print(f"\n✓ Running: {', '.join([f'Priority {p}' for p in sorted(set(selected))])}\n")
                    break
                else:
                    print("❌ No valid options selected. Try again.\n")
            except (ValueError, IndexError):
                print("❌ Invalid input. Please enter numbers 1-5 separated by commas.\n")
        
        self.run(priorities=list(set(selected)))


if __name__ == "__main__":
    import sys
    
    suite = VisualizationSuite()
    
    # Check for command line arguments
    if "--skip-overlays" in sys.argv:
        print("Skipping frame overlays (--skip-overlays flag)")
        data = suite.load_metrics()
        suite.generate_feature_importance_plots()
        suite.generate_distribution_plots(data)
        suite.generate_correlation_scatterplots(data)
        suite.generate_correlation_heatmap(data)
        suite.generate_timeline_plots(data)
        print("\n" + "=" * 60)
        print("VISUALIZATION COMPLETE")
        print(f"Output: {suite.output_dir}")
        print("=" * 60)
    elif "--interactive" in sys.argv or "-i" in sys.argv:
        suite.run_interactive()
    else:
        # Ask user: all or interactive?
        print("\n" + "=" * 60)
        print("SKI JUMPING VISUALIZATION SUITE")
        print("=" * 60)
        choice = input("\nRun ALL visualizations or choose SPECIFIC ones? (all/specific): ").strip().lower()
        
        if choice in ['specific', 'spec', 's']:
            suite.run_interactive()
        else:
            suite.run()
