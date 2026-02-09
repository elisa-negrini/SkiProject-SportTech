import json
import numpy as np
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt        
import math
import pandas as pd
import glob


class Normalizer:
    """Normalize annotations using:
    1. Hybrid scale (torso + leg + shoulders)
    2. Fixed scaling to map physical range into [0,1]
    """
    
    def __init__(self):
        self.dataset_dir = Path('dataset/annotations')
        
        self.anchor_ids = {
            'neck': '2',
            'r_shoulder': '3', 'l_shoulder': '6',
            'r_hip': '17', 'l_hip': '10',
            'r_knee': '18', 'l_knee': '11',
            'r_ankle': '19', 'l_ankle': '12'
        }
    
    def get_point(self, kpts, name, kpt_names):
        """Return (x,y) for a keypoint name."""
        try:
            idx = kpt_names.index(name)
        except ValueError:
            return None
        
        start = idx * 3
        if start + 2 >= len(kpts):
            return None
        
        x, y, v = kpts[start], kpts[start+1], kpts[start+2]
        if v == 0:
            return None
        
        return np.array([x, y])
    
    def get_dist(self, p1, p2):
        """Compute Euclidean distance between two points."""
        if p1 is None or p2 is None:
            return 0.0
        return np.linalg.norm(p1 - p2)
    
    def calculate_scale_and_root(self, kpts, kpt_names):
        """Compute hybrid scale and root (pelvis)."""
        pts = {k: self.get_point(kpts, v, kpt_names) 
               for k, v in self.anchor_ids.items()}
        
        root = None
        if pts['r_hip'] is not None and pts['l_hip'] is not None:
            root = (pts['r_hip'] + pts['l_hip']) / 2.0
        elif pts['r_hip'] is not None:
            root = pts['r_hip']
        elif pts['l_hip'] is not None:
            root = pts['l_hip']
        
        if root is None:
            return None, None
        
        torso_top = pts['neck']
        if torso_top is None and pts['r_shoulder'] is not None and pts['l_shoulder'] is not None:
            torso_top = (pts['r_shoulder'] + pts['l_shoulder']) / 2
        
        len_torso = self.get_dist(torso_top, root)
        if len_torso == 0: len_torso = 1.0 # minimal fallback
        
        # Legs (sum of thigh+shin, take longest visible)
        len_thigh_r = self.get_dist(pts['r_hip'], pts['r_knee'])
        len_shin_r = self.get_dist(pts['r_knee'], pts['r_ankle'])
        len_leg_r = len_thigh_r + len_shin_r
        
        len_thigh_l = self.get_dist(pts['l_hip'], pts['l_knee'])
        len_shin_l = self.get_dist(pts['l_knee'], pts['l_ankle'])
        len_leg_l = len_thigh_l + len_shin_l
        
        len_leg_max = max(len_leg_r, len_leg_l)
        if len_leg_max == 0: len_leg_max = len_torso * 1.2 # fallback
        
        # Shoulders
        len_shoulders = self.get_dist(pts['r_shoulder'], pts['l_shoulder'])
        
        # Hybrid sum
        scale_val = len_torso + len_leg_max + len_shoulders
        
        if scale_val == 0: scale_val = 1.0
        
        return root, scale_val
    
    def normalize_annotation(self, ann, kpt_names, root, scale):
        """Apply anatomical normalization and fixed scaling."""
        kpts = ann['keypoints'].copy()
        
        # FIXED_LIMIT defines range [-2.0, +2.0] mapped to [0,1]
        FIXED_LIMIT = 2.0          
        # Formula: new_val = 0.5 + (old_val / (2 * LIMIT))
        scale_factor = 1.0 / (2 * FIXED_LIMIT)
        
        for i in range(0, len(kpts), 3):
            x, y, v = kpts[i], kpts[i+1], kpts[i+2]
            
            if v > 0:
                x_rel = (x - root[0]) / scale
                y_rel = (y - root[1]) / scale
                
                x_final = 0.5 + (x_rel * scale_factor)
                y_final = 0.5 + (y_rel * scale_factor)
                
                x_final = np.clip(x_final, 0.0, 1.0)
                y_final = np.clip(y_final, 0.0, 1.0)
                
                kpts[i] = x_final
                kpts[i + 1] = y_final
            else:
                kpts[i] = 0.5
                kpts[i + 1] = 0.5
        
        ann['keypoints'] = kpts
        return ann
    
    def process(self, jump_num):
        """Process a single jump: read interpolated JSON -> write normalized JSON."""
        jump_id = f"JP{jump_num:04d}"
        jump_dir = self.dataset_dir / jump_id
        
        input_file = jump_dir / "train" / f"annotations_interpolated_jump{jump_num}.coco.json"
        output_file = jump_dir / "train" / f"annotations_normalized_jump{jump_num}.coco.json"
        
        if not input_file.exists():
            print(f"   ⚠️  File not found: {input_file}")
            return False
        
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            cat = next((c for c in data['categories'] if 'keypoints' in c), None)
            if not cat: return False
            
            kpt_names = cat['keypoints']
            
            normalized_count = 0
            
            for ann in data['annotations']:
                if ann['category_id'] != cat['id']: continue
                
                root, scale = self.calculate_scale_and_root(ann['keypoints'], kpt_names)
                
                if root is None: continue
                
                self.normalize_annotation(ann, kpt_names, root, scale)
                normalized_count += 1
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"    Normalized {normalized_count} frames -> {output_file.name}")
            return True
            
        except Exception as e:
            print(f"   ❌ Normalization error: {e}")
            return False

    def visualize_normalization(self, jump_num):

        jump_id = f"JP{jump_num:04d}"
        jump_dir = self.dataset_dir / jump_id
        
        input_file = jump_dir / "train" / f"annotations_normalized_jump{jump_num}.coco.json"
        output_image = jump_dir / "train" / f"visualization_normalized_grid_jump{jump_num}.png"
        
        if not input_file.exists():
            print(f"   ⚠️  Normalized file not found: {input_file}")
            return False
        
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            cat = next((c for c in data['categories'] if 'keypoints' in c), None)
            if not cat: return False
            
            kpt_names = cat['keypoints']
            skeleton = cat.get('skeleton', [])
            
            annotations = [ann for ann in data['annotations'] if ann['category_id'] == cat['id']]
            num_frames = len(annotations)
            
            if num_frames == 0:
                print("   ⚠️  No annotations found.")
                return False

            cols = 8
            rows = math.ceil(num_frames / cols)
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
            axes = axes.flatten()
            
            print(f"   Generating grid for {num_frames} frames...")

            for idx, ann in enumerate(annotations):
                ax = axes[idx]
                kpts = ann['keypoints']
                
                pts_map = {}
                
                for i in range(0, len(kpts), 3):
                    x, y, v = kpts[i], kpts[i+1], kpts[i+2]
                    
                    pts_map[i//3] = (x, y)
                    
                    
                    name = kpt_names[i//3]
                    is_ski = name in ['15','16','22','23','12','13','14','21','20','19']
                    color = 'blue' if is_ski else 'red'
                    
                    alpha = 1.0 if v > 0 else 0.3
                    ax.scatter(x, y, s=15, c=color, alpha=alpha, zorder=5)

                for p1, p2 in skeleton:
                    idx1, idx2 = p1 - 1, p2 - 1
                    if idx1 in pts_map and idx2 in pts_map:
                        x1, y1 = pts_map[idx1]
                        x2, y2 = pts_map[idx2]
                        ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1, alpha=0.6)

                ax.set_xlim(0, 1)
                ax.set_ylim(1, 0) 
                
                ax.grid(True, linestyle=':', alpha=0.5)
                ax.set_xticks([0, 0.5, 1])
                ax.set_yticks([0, 0.5, 1])
                ax.tick_params(labelsize=6)
                
                ax.set_title(f"Fr {idx}", fontsize=8)
                
                ax.axhline(0.5, color='green', linewidth=0.5, alpha=0.3)
                ax.axvline(0.5, color='green', linewidth=0.5, alpha=0.3)

            for j in range(idx + 1, len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            plt.savefig(str(output_image), dpi=100)
            plt.close()
            
            print(f"    Grid visualization saved: {output_image.name}")
            return True
            
        except Exception as e:
            print(f"   ❌ Visualization error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def create_dataset_csv(self, output_name='dataset/keypoints_dataset.csv'):
            
            print(f"\n--- Creating dataset CSV ({output_name}) ---")
            
            pattern = str(self.dataset_dir / "**" / "*annotations_normalized*.json")
            files = glob.glob(pattern, recursive=True)
            
            if not files:
                print("❌ No normalized files found. Run normalization first.")
                return False
                
            print(f"   Found {len(files)} JSON files.")
            
            all_rows = []
            
            for json_file in files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    path_parts = Path(json_file).parts
                    jump_id = "unknown"
                    for part in path_parts:
                        if part.startswith("JP") or part.startswith("jump"):
                            jump_id = part
                            break
                    
                    cat = next((c for c in data['categories'] if 'keypoints' in c), None)
                    if not cat: continue
                    
                    kpt_names = cat['keypoints']
                    img_map = {img['id']: img['file_name'] for img in data['images']}
                    
                    for ann in data['annotations']:
                        if ann['category_id'] != cat['id']: continue
                        
                        row = {
                            'jump_id': jump_id,
                            'frame_name': img_map.get(ann['image_id'], 'unknown'),
                            'image_id': ann['image_id']
                        }
                        
                        kpts = ann['keypoints']
                        for i, name in enumerate(kpt_names):
                            x, y, v = kpts[i*3], kpts[i*3+1], kpts[i*3+2]
                            row[f'kpt_{name}_x'] = x
                            row[f'kpt_{name}_y'] = y
                            row[f'kpt_{name}_v'] = v
                        
                        all_rows.append(row)
                        
                except Exception as e:
                    print(f"   ❌ Error reading {Path(json_file).name}: {e}")

            if not all_rows:
                print("⚠️  No data extracted.")
                return False

            df = pd.DataFrame(all_rows)
            
            try:
                df['sort_frame'] = df['frame_name'].apply(lambda x: int(''.join(filter(str.isdigit, str(x)))))
                df = df.sort_values(by=['jump_id', 'sort_frame'])
                df = df.drop(columns=['sort_frame'])
            except:
                df = df.sort_values(by=['jump_id', 'frame_name'])
            
            # Create dataset directory if it doesn't exist
            Path(output_name).parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(output_name, index=False)
            print(f"✅ Dataset saved: {output_name}")
            print(f"   Total frames: {len(df)}")
            print(f"   Columns: {list(df.columns)[:5]} ...")
            return True