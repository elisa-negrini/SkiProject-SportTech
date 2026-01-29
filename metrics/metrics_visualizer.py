import pandas as pd
import numpy as np
import cv2
import json
import os
from pathlib import Path

class MetricsVisualizer:
    def __init__(self, dataset_root='dataset', metrics_file='metrics/metrics_per_frame.csv'):
        self.root = Path(dataset_root)
        self.metrics_path = Path(metrics_file)
        
        # Logical name -> ID string in Roboflow JSON
        self.kpt_id_map = {
            # V-Style
            'head': '1', 'neck': '2',
            'center_pelvis': '9',
            'r_ski_tip': '23', 'r_ski_tail': '22',
            'l_ski_tip': '16', 'l_ski_tail': '15',
            # Body-Ski Angle & Altro
            'r_shoulder': '3', 'r_ankle': '19',
            'l_shoulder': '6', 'l_ankle': '12',
            'r_hip': '17', 'r_knee': '18',
            'l_hip': '10', 'l_knee': '11'
        }

    def load_data(self):
        if not self.metrics_path.exists():
            print(f"❌ Metrics file not found: {self.metrics_path}")
            return False
        self.df = pd.read_csv(self.metrics_path)
        return True

    def get_jump_ids(self):
        return sorted(self.df['jump_id'].unique())

    def get_available_metrics(self, jump_id):
        df_jump = self.df[self.df['jump_id'] == jump_id]
        valid_cols = []
        
        # Check if at least one of the two columns has valid data
        has_v_front = 'v_style_angle_front' in df_jump.columns and df_jump['v_style_angle_front'].notna().any()
        has_v_back = 'v_style_angle_back' in df_jump.columns and df_jump['v_style_angle_back'].notna().any()
        
        if has_v_front or has_v_back:
            valid_cols.append('v_style_angle')

        # Other standard metrics
        candidates = [
            'symmetry_index_back', 
            'body_ski_angle', 
            'takeoff_knee_angle', 
            'telemark_leg_angle', 
            'telemark_proj_ski_raw'
        ]
        for col in candidates:
            if col in df_jump.columns and df_jump[col].notna().any():
                valid_cols.append(col)
                
        return valid_cols

    def get_kpt_pos(self, keypoints, logical_name, kpt_list_order):
        real_id = self.kpt_id_map.get(logical_name)
        if not real_id: return None
        try:
            idx = kpt_list_order.index(real_id)
            base = idx * 3
            if base + 2 >= len(keypoints): return None
            x, y, v = keypoints[base], keypoints[base+1], keypoints[base+2]
            if v == 0: return None
            return np.array([x, y])
        except (ValueError, IndexError):
            return None

    def crop_to_skier(self, img, ann, margin=0.4):
        h_img, w_img = img.shape[:2]
        bx, by, bw, bh = ann['bbox']
        cx, cy = bx + bw/2, by + bh/2
        max_side = max(bw, bh)
        crop_size = int(max_side * (1 + margin))
        
        x1 = int(max(0, cx - crop_size/2))
        y1 = int(max(0, cy - crop_size/2))
        x2 = int(min(w_img, cx + crop_size/2))
        y2 = int(min(h_img, cy + crop_size/2))
        
        cropped = img[y1:y2, x1:x2]
        offset = np.array([x1, y1])
        return cropped, offset

    def find_intersection(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if denom == 0: return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return np.array([px, py])
    
    def draw_dashed_line(self, img, p1, p2, color, thickness=2, dash_len=10):
        """Draw a dashed line between p1 and p2."""
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        if dist == 0: return img
        
        tangent = vec / dist
        segments = int(dist / dash_len)
        
        for i in range(segments):
            start = p1 + tangent * (i * dash_len)
            end = p1 + tangent * ((i + 0.5) * dash_len)
            # Check not to exceed p2
            if np.linalg.norm(start - p1) > dist: break
            if np.linalg.norm(end - p1) > dist: end = p2
            
            cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)
        return img
    
    def draw_symmetry(self, img, offset, keypoints, kpt_names, metric_val, f_idx):
        # Retrieve axis points and calculate regression line
        pts_axis = []
        for name in ['head', 'neck', 'center_pelvis']:
            pt = self.get_kpt_pos(keypoints, name, kpt_names)
            if pt is not None: pts_axis.append(pt - offset)
            
        # New geometric calculation
        p_top_final, p_pelvis_core, p_bottom_final = None, None, None
        
        if len(pts_axis) >= 2:
            data = np.array(pts_axis)
            x, y = data[:, 0], data[:, 1]
            try:
                # Calculate regression line
                slope, intercept = np.polyfit(x, y, 1)
                
                # Find real extreme points on the back
                y_min, y_max = np.min(y), np.max(y)
                x_at_ymin = (y_min - intercept) / slope
                x_at_ymax = (y_max - intercept) / slope
                
                p_head_core = np.array([x_at_ymin, y_min])   # Highest point on back
                p_pelvis_core = np.array([x_at_ymax, y_max]) # Lowest point (pelvis)
                
                # Calculate back vector and length
                vec_core = p_pelvis_core - p_head_core
                len_core = np.linalg.norm(vec_core)
                vec_unit = vec_core / len_core if len_core > 0 else np.array([0,1])
                
                # Calculate extensions
                # Top extension (10% solid): extend upward
                p_top_final = p_head_core - vec_unit * (len_core * 0.10)
                # Bottom extension (100% dashed): extend downward
                p_bottom_final = p_pelvis_core + vec_unit * (len_core * 2.00)
                
            except: pass
        
        # Retrieve skis
        tr = self.get_kpt_pos(keypoints, 'r_ski_tail', kpt_names)
        pr = self.get_kpt_pos(keypoints, 'r_ski_tip', kpt_names)
        tl = self.get_kpt_pos(keypoints, 'l_ski_tail', kpt_names)
        pl = self.get_kpt_pos(keypoints, 'l_ski_tip', kpt_names)
        
        if p_top_final is not None and all(x is not None for x in [tr, pr, tl, pl]):
            tr, pr, tl, pl = tr - offset, pr - offset, tl - offset, pl - offset
            
            col_axis = (255, 0, 0) # Blue in BGR
            thickness = 2

            # Drawing sections
            # A. Solid part (Blue): From top extension to pelvis
            cv2.line(img, tuple(p_top_final.astype(int)), tuple(p_pelvis_core.astype(int)), col_axis, thickness, cv2.LINE_AA)
            
            # B. Dashed part (Blue): From pelvis to bottom extension
            self.draw_dashed_line(img, p_pelvis_core, p_bottom_final, col_axis, thickness, dash_len=15)

            # C. Draw skis (Red)
            cv2.line(img, tuple(tr.astype(int)), tuple(pr.astype(int)), (0, 0, 255), 3)
            cv2.line(img, tuple(tl.astype(int)), tuple(pl.astype(int)), (0, 0, 255), 3)
            
            # D. Label
            label_lines = [f"Symmetry Index: {metric_val:.2f}", f"Frame: {f_idx}"]
            self.draw_label_with_box(img, label_lines, (int(p_pelvis_core[0]), int(p_pelvis_core[1]) + 50))
        return img

    def draw_label_with_box(self, img, text_lines, pos_target):
        if isinstance(text_lines, str): text_lines = [text_lines]
        
        h_img, w_img = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        pad = 5
        line_spacing = 6
        
        max_w = 0
        total_h = 0
        
        for line in text_lines:
            (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            max_w = max(max_w, w)
            total_h += h + baseline + line_spacing
        
        total_h -= line_spacing
        
        w_box = max_w + 2*pad
        h_box = total_h + 2*pad
        
        x, y = pos_target
        x = max(0, min(x, w_img - w_box))
        y = max(0, min(y, h_img - h_box))
        
        cv2.rectangle(img, (int(x), int(y)), (int(x + w_box), int(y + h_box)), (255, 255, 255), -1)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w_box), int(y + h_box)), (0, 0, 0), 1)
        
        curr_y = y + pad
        for line in text_lines:
            (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            center_x = int(x + (w_box / 2) - (w / 2))
            
            curr_y += h + baseline
            cv2.putText(img, line, (center_x, int(curr_y - baseline + 2)), 
                       font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            curr_y += line_spacing

    def draw_v_style(self, img, offset, keypoints, kpt_names, metric_val, f_idx, view_type=""):
        tr = self.get_kpt_pos(keypoints, 'r_ski_tail', kpt_names)
        pr = self.get_kpt_pos(keypoints, 'r_ski_tip', kpt_names)
        tl = self.get_kpt_pos(keypoints, 'l_ski_tail', kpt_names)
        pl = self.get_kpt_pos(keypoints, 'l_ski_tip', kpt_names)
        
        if any(x is None for x in [tr, pr, tl, pl]): return img
        
        tr, pr, tl, pl = tr - offset, pr - offset, tl - offset, pl - offset
        
        col_ski = (0, 0, 255)   # Red
        col_fill = (0, 255, 0)  # Green
        
        cv2.line(img, tuple(tr.astype(int)), tuple(pr.astype(int)), col_ski, 3)
        cv2.line(img, tuple(tl.astype(int)), tuple(pl.astype(int)), col_ski, 3)
        
        vertex = self.find_intersection(tr, pr, tl, pl)
        
        # Dynamic label: FRONT or BACK
        label_text = f"V-Style: {metric_val:.1f} deg"
        if view_type:
            label_text = f"V-Style ({view_type}): {metric_val:.1f} deg"
            
        label_lines = [label_text, f"Frame: {f_idx}"]
        
        if vertex is not None:
            v_int = vertex.astype(int)
            dist_tr = np.linalg.norm(tr - v_int)
            dist_tl = np.linalg.norm(tl - v_int)
            radius = int(min(dist_tr, dist_tl))
            
            angle_r = np.degrees(np.arctan2(tr[1]-v_int[1], tr[0]-v_int[0]))
            angle_l = np.degrees(np.arctan2(tl[1]-v_int[1], tl[0]-v_int[0]))
            start_ang = min(angle_r, angle_l)
            end_ang = max(angle_r, angle_l)
            if end_ang - start_ang > 180: start_ang, end_ang = end_ang, start_ang + 360

            if 5 < radius < 5000:
                overlay = img.copy()
                try:
                    cv2.ellipse(overlay, tuple(v_int), (radius, radius), 0, start_ang, end_ang, col_fill, -1)
                    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
                except: pass
            
            # Box fisso in basso a sinistra (come richiesto precedentemente)
            h_img = img.shape[0]
            fixed_pos = (10, h_img - 60)
            self.draw_label_with_box(img, label_lines, fixed_pos)
        else:
            self.draw_label_with_box(img, label_lines, (10, 10))

        return img
    
    def get_regression_line(self, points, img_shape):
        """Calculate start/end points to draw regression line."""
        if len(points) < 2: return None
        data = np.array(points)
        x, y = data[:, 0], data[:, 1]
        try:
            slope, intercept = np.polyfit(x, y, 1)
            # Create two points to draw the line
            y_min, y_max = np.min(y), np.max(y)
            x_min = (y_min - intercept) / slope
            x_max = (y_max - intercept) / slope
            return (np.array([x_min, y_min]), np.array([x_max, y_max]))
        except: return None

    def draw_body_ski(self, img, offset, keypoints, kpt_names, metric_val, f_idx):
        # 1. Raccogli punti per la regressione (CORPO)
        # Nota: usiamo get_kpt_pos che restituisce coordinate reali nell'immagine
        pts_body = []
        # Raccogliamo sia destra che sinistra insieme per fare una "super regressione" media?
        # Oppure facciamo media di due rette? La tua richiesta era "media tra i due vettori".
        # Visivamente è più pulito fare una regressione su TUTTI i punti validi (DX + SX) insieme.
        # Questo dà automaticamente la linea media.
        for side in ['r_', 'l_']:
            for part in ['shoulder', 'hip', 'knee', 'ankle']:
                pt = self.get_kpt_pos(keypoints, side + part, kpt_names)
                if pt is not None: pts_body.append(pt - offset) # Applica offset crop
        
        # 2. Raccogli punti per gli SCI (Punte e Code)
        # Qui calcoliamo i vettori medi sommando punta media e coda media
        tips = []
        tails = []
        for side in ['r_', 'l_']:
            p_t = self.get_kpt_pos(keypoints, side + 'ski_tip', kpt_names)
            p_ta = self.get_kpt_pos(keypoints, side + 'ski_tail', kpt_names)
            if p_t is not None: tips.append(p_t - offset)
            if p_ta is not None: tails.append(p_ta - offset)
            
        # --- DISEGNO ---
        
        # A. Linea Corpo (Regressione)
        body_line = self.get_regression_line(pts_body, img.shape)
        center_body = None
        if body_line:
            p1, p2 = body_line
            cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 255, 0), 3) # Ciano
            center_body = (p1 + p2) / 2 # Punto medio per ancorare l'angolo
            
        # B. Linea Sci (Media)
        center_ski = None
        if tips and tails:
            # Calcola media delle punte e delle code
            avg_tip = np.mean(np.array(tips), axis=0)
            avg_tail = np.mean(np.array(tails), axis=0)
            
            cv2.line(img, tuple(avg_tip.astype(int)), tuple(avg_tail.astype(int)), (0, 0, 255), 3) # Rosso
            
            # Punto di intersezione fittizio per disegnare l'angolo?
            # O usiamo la caviglia media come perno?
            # Usiamo la media delle caviglie come perno visivo, è più naturale.
            ank_r = self.get_kpt_pos(keypoints, 'r_ankle', kpt_names)
            ank_l = self.get_kpt_pos(keypoints, 'l_ankle', kpt_names)
            label_lines = [f"BSA: {metric_val:.1f} deg", f"Frame: {f_idx}"]
            h_img = img.shape[0]
            fixed_pos = (10, h_img - 60) # Coordinate fisse
            
            self.draw_label_with_box(img, label_lines, fixed_pos)
        return img

    def visualize_interactive(self):
        if not self.load_data(): return

        jump_ids = self.get_jump_ids()
        print("\n--- AVAILABLE JUMPS ---")
        for j in jump_ids: print(f" - {j}")
        
        sel_jump = input("\nEnter jump ID (e.g. JP0005 or just 5): ").strip()
        if not sel_jump.startswith("JP"): sel_jump = f"JP{int(sel_jump):04d}"
        
        if sel_jump not in jump_ids:
            print("❌ Jump not found in metrics CSV.")
            return

        metrics = self.get_available_metrics(sel_jump)
        if not metrics:
            print("❌ No valid metrics found for this jump.")
            return

        print(f"\n--- AVAILABLE METRICS FOR {sel_jump} ---")
        for i, m in enumerate(metrics): print(f" {i+1}) {m}")
        
        try:
            m_idx = int(input("Choose metric number: ")) - 1
            metric_name = metrics[m_idx]
        except:
            print("❌ Invalid selection."); return

        ann_path = self.root / 'dataset' / 'annotations' / sel_jump / 'train' / f'annotations_interpolated_jump{int(sel_jump[2:])}.coco.json'
        frames_dir = self.root / 'dataset' / 'frames' / sel_jump
        save_dir = self.root / 'metrics' / 'visualizations' / sel_jump / metric_name
        save_dir.mkdir(parents=True, exist_ok=True)

        if not ann_path.exists():
            print(f"❌ Annotations not found: {ann_path}"); return

        with open(ann_path, 'r') as f: coco = json.load(f)
        
        kpt_names = None
        cat_with_kpts = next((c for c in coco.get('categories', []) if 'keypoints' in c), None)
        if cat_with_kpts: kpt_names = cat_with_kpts['keypoints']
        else: print("❌ CRITICAL ERROR: JSON without keypoints."); return
        
        # Prepare data for unified V-Style
        if metric_name == 'v_style_angle':
            # Filter rows where at least one of the two columns is valid
            mask = self.df['v_style_angle_front'].notna() | self.df['v_style_angle_back'].notna()
            df_view = self.df[(self.df['jump_id'] == sel_jump) & mask].copy()
        else:
            # Standard filter for other metrics
            df_view = self.df[(self.df['jump_id'] == sel_jump) & (self.df[metric_name].notna())].copy()
            
        df_view = df_view.sort_values('frame_idx')
        
        print(f"\n--- VISUALIZING: {metric_name} ({len(df_view)} frames) ---")
        print("Commands: [SPACE]=Next, [B]=Back, [ESC]=Exit")
        
        ann_map = {a['image_id']: a for a in coco['annotations']}
        img_map = {img['file_name']: img['id'] for img in coco['images']}
        frame_lookup = {}
        for fname in img_map.keys():
            import re
            m = re.search(r"(\d+)", fname)
            if m: frame_lookup[int(m.group(1))] = fname

        indices = list(df_view.index)
        curr_ptr = 0
        
        while curr_ptr < len(indices):
            row = df_view.loc[indices[curr_ptr]]
            f_idx = int(row['frame_idx'])
            
            # Determine value and type for V-Style
            val = 0.0
            view_str = ""
            
            if metric_name == 'v_style_angle':
                if pd.notna(row.get('v_style_angle_front')):
                    val = row['v_style_angle_front']
                    view_str = "FRONT"
                elif pd.notna(row.get('v_style_angle_back')):
                    val = row['v_style_angle_back']
                    view_str = "BACK"
            else:
                val = row[metric_name]
            
            fname = frame_lookup.get(f_idx)
            if not fname: curr_ptr += 1; continue
                
            img_path = frames_dir / fname
            if not img_path.exists(): curr_ptr += 1; continue
                
            img = cv2.imread(str(img_path))
            if img is None: curr_ptr += 1; continue

            img_id = img_map[fname]
            ann = ann_map.get(img_id)
            
            if ann and 'bbox' in ann:
                crop_img, offset = self.crop_to_skier(img, ann, margin=0.5)
                kpts = ann['keypoints']
                
                if metric_name == 'v_style_angle':
                    crop_img = self.draw_v_style(crop_img, offset, kpts, kpt_names, val, f_idx, view_str)
                
                elif 'symmetry_index_back' in metric_name:
                    crop_img = self.draw_symmetry(crop_img, offset, kpts, kpt_names, val, f_idx)
                
                elif 'body_ski' in metric_name:
                    crop_img = self.draw_body_ski(crop_img, offset, kpts, kpt_names, val, f_idx)
                else:
                    # Generic case
                    h_img = crop_img.shape[0]
                    fixed_pos = (10, h_img - 60)                    
                    self.draw_label_with_box(crop_img, [f"{metric_name}: {val:.2f}", f"Frame: {f_idx}"], fixed_pos)

                save_path = save_dir / f"viz_{f_idx:05d}.jpg"
                cv2.imwrite(str(save_path), crop_img)
                cv2.imshow("Metrics Visualizer", crop_img)
                
                k = cv2.waitKey(0)
                if k == 27: break
                elif k == ord('b'): curr_ptr = max(0, curr_ptr - 1)
                else: curr_ptr += 1
            else:
                curr_ptr += 1

        cv2.destroyAllWindows()
        print(f"\n✅ Visualization completed. Saved in: {save_dir}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent 
    viz = MetricsVisualizer(dataset_root=root_dir)
    viz.visualize_interactive()