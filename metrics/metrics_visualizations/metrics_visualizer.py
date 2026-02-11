import pandas as pd
import numpy as np
import cv2
import json
import os
from pathlib import Path

class MetricsVisualizer:
    def __init__(self, dataset_root='.', metrics_file='metrics/core_metrics/metrics_per_frame.csv'):
        self.root = Path(__file__).parent.parent.parent
        
        self.metrics_path = self.root / metrics_file
        self.summary_metrics_path = self.root / 'metrics' / 'core_metrics' / 'metrics_summary_per_jump.csv'
        self.phases_path = self.root / 'dataset' / 'jump_phases_SkiTB.csv'
        
        if not self.phases_path.exists():
            print(f" Warning: {self.phases_path} not found. Landing metrics visualization may fail.")
        else:
            print(f" Phases file found at: {self.phases_path}")

        self.kpt_id_map = {
            'head': '1', 'neck': '2',
            'center_pelvis': '9',
            'r_ski_tip': '23', 'r_ski_tail': '22',
            'l_ski_tip': '16', 'l_ski_tail': '15',
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
        if hasattr(self, 'df'):
            return sorted(self.df['jump_id'].unique())
        return []

    def get_available_metrics(self, jump_id):
        """Returns the list of available metrics for a jump."""
        available = []
        df_jump = self.df[self.df['jump_id'] == jump_id]
        if not df_jump.empty:
            for col in df_jump.columns:
                if df_jump[col].notna().any():
                    available.append(col)
            
            has_front = 'v_style_angle_front' in df_jump.columns and df_jump['v_style_angle_front'].notna().any()
            has_back = 'v_style_angle_back' in df_jump.columns and df_jump['v_style_angle_back'].notna().any()
            
            if (has_front or has_back) and 'v_style_angle' not in available:
                available.append('v_style_angle')
            
            # Check for telemark_scissor: use telemark_scissor_ratio column or landing phase
            has_scissor_ratio = 'telemark_scissor_ratio' in df_jump.columns and df_jump['telemark_scissor_ratio'].notna().any()
            has_landing_phase = 'is_landing_phase' in df_jump.columns and (df_jump['is_landing_phase'] == 1).any()
            
            if (has_scissor_ratio or has_landing_phase) and 'telemark_scissor' not in available:
                available.append('telemark_scissor')
        
        if self.summary_metrics_path.exists():
            try:
                df_sum = pd.read_csv(self.summary_metrics_path)
                if jump_id in df_sum['jump_id'].values:
                    df_sum_jump = df_sum[df_sum['jump_id'] == jump_id]
                else:
                    df_sum_jump = df_sum[df_sum['jump_id'].apply(lambda x: self._normalize_jid(x)) == jump_id]

                if not df_sum_jump.empty:
                    for col in df_sum_jump.columns:
                        if df_sum_jump[col].notna().any() and col not in available:
                            available.append(col)
            except Exception as e:
                print(f"Error reading summary metrics: {e}")
                        
        return available

    def _normalize_jid(self, val):
        import re
        s = str(val)
        m = re.search(r'(\d+)', s)
        if m:
            return f"JP{int(m.group(1)):04d}"
        return str(val)

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
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        if dist == 0: return img
        
        tangent = vec / dist
        segments = int(dist / dash_len)
        
        for i in range(segments):
            start = p1 + tangent * (i * dash_len)
            end = p1 + tangent * ((i + 0.5) * dash_len)
            if np.linalg.norm(start - p1) > dist: break
            if np.linalg.norm(end - p1) > dist: end = p2
            cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)
        return img
    
    def draw_symmetry(self, img, offset, keypoints, kpt_names, metric_val, f_idx, scale=1.0):
        p_neck = self.get_kpt_pos(keypoints, 'neck', kpt_names)
        p_knee_r = self.get_kpt_pos(keypoints, 'r_knee', kpt_names)
        p_knee_l = self.get_kpt_pos(keypoints, 'l_knee', kpt_names)
        
        if p_neck is not None: p_neck = (p_neck - offset) * scale
        if p_knee_r is not None: p_knee_r = (p_knee_r - offset) * scale
        if p_knee_l is not None: p_knee_l = (p_knee_l - offset) * scale
            
        tr = self.get_kpt_pos(keypoints, 'r_ski_tail', kpt_names)
        pr = self.get_kpt_pos(keypoints, 'r_ski_tip', kpt_names)
        tl = self.get_kpt_pos(keypoints, 'l_ski_tail', kpt_names)
        pl = self.get_kpt_pos(keypoints, 'l_ski_tip', kpt_names)
        
        skis_ok = all(x is not None for x in [tr, pr, tl, pl])
        if skis_ok:
            tr = (tr - offset) * scale; pr = (pr - offset) * scale
            tl = (tl - offset) * scale; pl = (pl - offset) * scale

        p_ref_core, vec_body_axis = None, None
        p_bottom_extended = None
        
        if p_neck is not None and p_knee_r is not None and p_knee_l is not None:
            p_mid_knee = (p_knee_r + p_knee_l) / 2
            vec_core = p_mid_knee - p_neck 
            len_core = np.linalg.norm(vec_core)
            
            if len_core > 0:
                vec_body_axis = vec_core / len_core
                p_ref_core = p_mid_knee
                p_top_draw = p_neck - vec_body_axis * (len_core * 0.20)
                p_bottom_draw = p_mid_knee + vec_body_axis * (len_core * 3.00) 
                p_bottom_extended = p_mid_knee + vec_body_axis * 10000 

                cv2.line(img, tuple(p_top_draw.astype(int)), tuple(p_mid_knee.astype(int)), (255, 0, 0), 2, cv2.LINE_AA)
                self.draw_dashed_line(img, p_mid_knee, p_bottom_draw, (255, 0, 0), 2, dash_len=15)

        if skis_ok and p_ref_core is not None and p_bottom_extended is not None:
            def get_angle_between(v1, v2):
                v1_u = v1 / np.linalg.norm(v1)
                v2_u = v2 / np.linalg.norm(v2)
                dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
                deg = np.degrees(np.arccos(dot))
                if deg > 90: deg = 180 - deg
                return deg

            int_r = self.find_intersection(tr, pr, p_ref_core, p_bottom_extended)
            int_l = self.find_intersection(tl, pl, p_ref_core, p_bottom_extended)
            
            len_ski = (np.linalg.norm(pr - tr) + np.linalg.norm(pl - tl)) / 2
            dist_target_from_ref = len_ski * 0.4
            
            def get_dist_on_axis(p):
                return np.dot(p - p_ref_core, vec_body_axis)

            vec_up_axis = -vec_body_axis 

            if int_r is not None:
                self.draw_dashed_line(img, tr, int_r, (0, 255, 255), 1, dash_len=8)
                radius_r = max(30, abs(get_dist_on_axis(int_r) - dist_target_from_ref))
                ang_val_r = get_angle_between(pr - tr, vec_body_axis)
                self.draw_angle_arc_with_label(img, int_r, vec_up_axis, tr - int_r, (0, 255, 255), ang_val_r, radius=int(radius_r))

            if int_l is not None:
                self.draw_dashed_line(img, tl, int_l, (255, 0, 255), 1, dash_len=8)
                radius_l = max(30, abs(get_dist_on_axis(int_l) - dist_target_from_ref))
                ang_val_l = get_angle_between(pl - tl, vec_body_axis)
                self.draw_angle_arc_with_label(img, int_l, vec_up_axis, tl - int_l, (255, 0, 255), ang_val_l, radius=int(radius_l))

            cv2.line(img, tuple(tr.astype(int)), tuple(pr.astype(int)), (0, 0, 255), 3)
            cv2.line(img, tuple(tl.astype(int)), tuple(pl.astype(int)), (0, 0, 255), 3)
            
        return img
    
    def draw_angle_arc_with_label(self, img, center, vec_start, vec_end, color_fill, val_deg, radius=50):
        ang_start = np.degrees(np.arctan2(vec_start[1], vec_start[0]))
        ang_end = np.degrees(np.arctan2(vec_end[1], vec_end[0]))
        
        if ang_end - ang_start > 180: ang_end -= 360
        if ang_start - ang_end > 180: ang_start -= 360
        
        start_draw = min(ang_start, ang_end)
        end_draw = max(ang_start, ang_end)
        
        overlay = img.copy()
        cv2.ellipse(overlay, tuple(center.astype(int)), (radius, radius), 
                   0, start_draw, end_draw, color_fill, -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        mid_ang_rad = np.radians((start_draw + end_draw) / 2)
        label_dist = radius + 20
        lx = center[0] + np.cos(mid_ang_rad) * label_dist
        ly = center[1] + np.sin(mid_ang_rad) * label_dist
        
        text = f"{val_deg:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        tl = (int(lx - w/2 - 2), int(ly - h/2 - 2))
        br = (int(lx + w/2 + 2), int(ly + h/2 + 2))
        cv2.rectangle(img, tl, br, (0,0,0), -1)
        
        cv2.putText(img, text, (int(lx - w/2), int(ly + h/2)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_fill, 2, cv2.LINE_AA)
        return img

    def draw_v_style(self, img, offset, keypoints, kpt_names, metric_val, f_idx, view_type="", scale=1.0):
        tr = self.get_kpt_pos(keypoints, 'r_ski_tail', kpt_names)
        pr = self.get_kpt_pos(keypoints, 'r_ski_tip', kpt_names)
        tl = self.get_kpt_pos(keypoints, 'l_ski_tail', kpt_names)
        pl = self.get_kpt_pos(keypoints, 'l_ski_tip', kpt_names)
        
        if any(x is None for x in [tr, pr, tl, pl]): return img
        
        tr = (tr - offset) * scale; pr = (pr - offset) * scale
        tl = (tl - offset) * scale; pl = (pl - offset) * scale
        
        cv2.line(img, tuple(tr.astype(int)), tuple(pr.astype(int)), (0, 0, 255), 3)
        cv2.line(img, tuple(tl.astype(int)), tuple(pl.astype(int)), (0, 0, 255), 3)
        
        vertex = self.find_intersection(tr, pr, tl, pl)
        
        if vertex is not None:
            self.draw_dashed_line(img, tr, vertex, (100, 100, 255), 1, dash_len=8)
            self.draw_dashed_line(img, tl, vertex, (100, 100, 255), 1, dash_len=8)
            
            dist_pr = np.linalg.norm(pr - vertex)
            dist_pl = np.linalg.norm(pl - vertex)
            radius = int(min(dist_pr, dist_pl) * 0.6) 
            
            vec_r = pr - vertex
            vec_l = pl - vertex
            self.draw_angle_arc_with_label(img, vertex, vec_r, vec_l, (0, 255, 0), metric_val, radius)
        return img
    
    def draw_body_ski(self, img, offset, keypoints, kpt_names, metric_val, f_idx, scale=1.0):
        shoulders = [self.get_kpt_pos(keypoints, s+'shoulder', kpt_names) for s in ['r_', 'l_']]
        ankles = [self.get_kpt_pos(keypoints, s+'ankle', kpt_names) for s in ['r_', 'l_']]
        
        shoulders = [p for p in shoulders if p is not None]
        ankles = [p for p in ankles if p is not None]
        
        p_body_top, p_body_bot = None, None
        if shoulders and ankles:
            avg_sh = np.mean(shoulders, axis=0)
            avg_ank = np.mean(ankles, axis=0)
            p_body_top = (avg_sh - offset) * scale
            p_body_bot = (avg_ank - offset) * scale
            cv2.line(img, tuple(p_body_top.astype(int)), tuple(p_body_bot.astype(int)), (255, 255, 0), 3)

        tips = [self.get_kpt_pos(keypoints, s+'ski_tip', kpt_names) for s in ['r_', 'l_']]
        tails = [self.get_kpt_pos(keypoints, s+'ski_tail', kpt_names) for s in ['r_', 'l_']]
        tips = [p for p in tips if p is not None]
        tails = [p for p in tails if p is not None]
        
        p_ski_tip, p_ski_tail = None, None
        if tips and tails:
            avg_tip = np.mean(tips, axis=0)
            avg_tail = np.mean(tails, axis=0)
            p_ski_tip = (avg_tip - offset) * scale
            p_ski_tail = (avg_tail - offset) * scale
            cv2.line(img, tuple(p_ski_tip.astype(int)), tuple(p_ski_tail.astype(int)), (0, 0, 255), 3)

        if p_body_top is not None and p_ski_tip is not None:
            vertex = self.find_intersection(p_body_top, p_body_bot, p_ski_tip, p_ski_tail)
            if vertex is not None:
                self.draw_dashed_line(img, p_body_bot, vertex, (255, 255, 0), 1, dash_len=6)
                self.draw_dashed_line(img, p_ski_tail, vertex, (0, 0, 255), 1, dash_len=6)
                vec_body = p_body_top - vertex
                vec_ski = p_ski_tip - vertex
                dist_to_tip = np.linalg.norm(vec_ski)
                dynamic_radius = max(int(dist_to_tip * 0.6), 50)
                self.draw_angle_arc_with_label(img, vertex, vec_body, vec_ski, (0, 255, 255), metric_val, radius=dynamic_radius)
        return img
    
    def draw_knee_compression(self, img, offset, keypoints, kpt_names, metric_val, f_idx, scale=1.0):
        legs = [
            {'hip': 'r_hip', 'knee': 'r_knee', 'ank': 'r_ankle', 'color': (0, 255, 255)}, 
            {'hip': 'l_hip', 'knee': 'l_knee', 'ank': 'l_ankle', 'color': (0, 165, 255)} 
        ]
        
        found_any = False
        for leg in legs:
            ph = self.get_kpt_pos(keypoints, leg['hip'], kpt_names)
            pk = self.get_kpt_pos(keypoints, leg['knee'], kpt_names)
            pa = self.get_kpt_pos(keypoints, leg['ank'], kpt_names)
            if all(p is not None for p in [ph, pk, pa]):
                found_any = True
                ph, pk, pa = (ph - offset) * scale, (pk - offset) * scale, (pa - offset) * scale
                cv2.line(img, tuple(ph.astype(int)), tuple(pk.astype(int)), leg['color'], 4, cv2.LINE_AA)
                cv2.line(img, tuple(pk.astype(int)), tuple(pa.astype(int)), leg['color'], 4, cv2.LINE_AA)

        if found_any:
            h, w = img.shape[:2]
            bar_w, bar_h = 20, int(h * 0.5)
            x_start, y_start = w - 40, int((h - bar_h) / 2)
            cv2.rectangle(img, (x_start, y_start), (x_start + bar_w, y_start + bar_h), (50, 50, 50), -1)
            max_comp_ref = 10.0 
            fill_ratio = min(metric_val / max_comp_ref, 1.0)
            fill_h = int(bar_h * fill_ratio)
            color_impact = (0, 255, 0) if fill_ratio < 0.5 else (0, 0, 255)
            cv2.rectangle(img, (x_start, y_start + bar_h - fill_h), (x_start + bar_w, y_start + bar_h), color_impact, -1)
            cv2.putText(img, "COMP", (x_start - 10, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img
    
    def draw_takeoff_knee(self, img, offset, keypoints, kpt_names, metric_val, f_idx, scale=1.0):
        legs = [
            {'hip': 'r_hip', 'knee': 'r_knee', 'ank': 'r_ankle', 'color': (0, 255, 0)},
            {'hip': 'l_hip', 'knee': 'l_knee', 'ank': 'l_ankle', 'color': (255, 255, 0)}
        ]
        found_any = False
        label_drawn = False
        for leg in legs:
            p_hip = self.get_kpt_pos(keypoints, leg['hip'], kpt_names)
            p_knee = self.get_kpt_pos(keypoints, leg['knee'], kpt_names)
            p_ank = self.get_kpt_pos(keypoints, leg['ank'], kpt_names)
            if all(p is not None for p in [p_hip, p_knee, p_ank]):
                found_any = True
                ph = (p_hip - offset) * scale
                pk = (p_knee - offset) * scale
                pa = (p_ank - offset) * scale
                cv2.line(img, tuple(ph.astype(int)), tuple(pk.astype(int)), leg['color'], 3, cv2.LINE_AA)
                cv2.line(img, tuple(pk.astype(int)), tuple(pa.astype(int)), leg['color'], 3, cv2.LINE_AA)
                cv2.circle(img, tuple(pk.astype(int)), 5, (255, 255, 255), -1)
                v_femur = ph - pk
                v_tibia = pa - pk
                if not label_drawn:
                    self.draw_angle_arc_with_label(img, pk, v_femur, v_tibia, leg['color'], metric_val, radius=40)
                    label_drawn = True
                else:
                    self.draw_only_arc_no_label(img, pk, v_femur, v_tibia, leg['color'], radius=40)
        return img

    def draw_only_arc_no_label(self, img, center, vec_start, vec_end, color_fill, radius=40):
        ang_start = np.degrees(np.arctan2(vec_start[1], vec_start[0]))
        ang_end = np.degrees(np.arctan2(vec_end[1], vec_end[0]))
        if ang_end - ang_start > 180: ang_end -= 360
        if ang_start - ang_end > 180: ang_start -= 360
        start_draw = min(ang_start, ang_end)
        end_draw = max(ang_start, ang_end)
        overlay = img.copy()
        cv2.ellipse(overlay, tuple(center.astype(int)), (radius, radius), 0, start_draw, end_draw, color_fill, -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    def draw_telemark_scissor(self, img, offset, keypoints, kpt_names, f_idx, scale=1.0):
        """Visualize telemark scissor: vertical offset between ankles (normalized by leg height).
        
        Scissor formula:
        - ankle_y_diff = |ankle_r.y - ankle_l.y|  (vertical pixel distance)
        - leg_height = |avg_hip.y - avg_ankle.y|  (leg length in pixels)
        - scissor_ratio = ankle_y_diff / leg_height  (0.0 to ~0.30)
        
        Draws full leg skeleton (hip→knee→ankle) with vertical offset line between ankles.
        Color: forward leg (lower ankle) = green, back leg (higher ankle) = orange.
        """
        # --- Get keypoints ---
        p_hip_r = self.get_kpt_pos(keypoints, 'r_hip', kpt_names)
        p_hip_l = self.get_kpt_pos(keypoints, 'l_hip', kpt_names)
        p_knee_r = self.get_kpt_pos(keypoints, 'r_knee', kpt_names)
        p_knee_l = self.get_kpt_pos(keypoints, 'l_knee', kpt_names)
        p_ankle_r = self.get_kpt_pos(keypoints, 'r_ankle', kpt_names)
        p_ankle_l = self.get_kpt_pos(keypoints, 'l_ankle', kpt_names)

        # Transform to crop coordinates
        def to_crop(p):
            return (p - offset) * scale if p is not None else None

        hip_r = to_crop(p_hip_r)
        hip_l = to_crop(p_hip_l)
        knee_r = to_crop(p_knee_r)
        knee_l = to_crop(p_knee_l)
        ankle_r = to_crop(p_ankle_r)
        ankle_l = to_crop(p_ankle_l)

        # Need all keypoints for proper calculation
        if any(p is None for p in [hip_r, hip_l, knee_r, knee_l, ankle_r, ankle_l]):
            # Draw warning
            cv2.putText(img, "Missing keypoints for scissor calculation", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return img

        # --- Calculate scissor_mean (frame-by-frame) ---
        ankle_y_diff = abs(ankle_r[1] - ankle_l[1])  # Vertical distance between ankles
        
        avg_hip_y = (hip_r[1] + hip_l[1]) / 2.0
        avg_ankle_y = (ankle_r[1] + ankle_l[1]) / 2.0
        leg_height = abs(avg_hip_y - avg_ankle_y)  # Leg length
        
        if leg_height < 10:  # Too small, invalid
            scissor_ratio = 0.0
        else:
            scissor_ratio = ankle_y_diff / leg_height
        
        # Clamp to valid range (0.0 to 0.50 max shown)
        scissor_ratio = np.clip(scissor_ratio, 0.0, 0.50)

        # --- Colors: Purple = Right leg, Orange = Left leg (fixed) ---
        color_r = (226, 43, 138)   # Purple (BGR) - right leg
        color_l = (0, 165, 255)    # Orange (BGR) - left leg

        # --- Draw leg skeletons ---
        # Right leg (purple)
        cv2.line(img, tuple(hip_r.astype(int)), tuple(knee_r.astype(int)), color_r, 4, cv2.LINE_AA)
        cv2.line(img, tuple(knee_r.astype(int)), tuple(ankle_r.astype(int)), color_r, 4, cv2.LINE_AA)
        
        # Left leg (orange)
        cv2.line(img, tuple(hip_l.astype(int)), tuple(knee_l.astype(int)), color_l, 4, cv2.LINE_AA)
        cv2.line(img, tuple(knee_l.astype(int)), tuple(ankle_l.astype(int)), color_l, 4, cv2.LINE_AA)

        # --- Draw keypoint circles ---
        for pt, color in [(hip_r, color_r), (hip_l, color_l), 
                          (knee_r, color_r), (knee_l, color_l),
                          (ankle_r, color_r), (ankle_l, color_l)]:
            cv2.circle(img, tuple(pt.astype(int)), 6, color, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(pt.astype(int)), 6, (255, 255, 255), 2, cv2.LINE_AA)

        # --- Draw vertical offset line between ankles (dashed) ---
        # Line connects the two ankles horizontally at the X position between them
        mid_ankle_x = int((ankle_r[0] + ankle_l[0]) / 2)
        ankle_lower_y = int(min(ankle_r[1], ankle_l[1]))
        ankle_higher_y = int(max(ankle_r[1], ankle_l[1]))
        
        # Draw vertical dashed line showing the offset
        self.draw_dashed_line(img, 
                              np.array([mid_ankle_x, ankle_lower_y]), 
                              np.array([mid_ankle_x, ankle_higher_y]),
                              (255, 255, 0), 2, dash_len=8)  # Cyan dashed line
        
        # Draw horizontal lines at each ankle to show levels
        line_len = 30
        cv2.line(img, (mid_ankle_x - line_len, ankle_lower_y), 
                 (mid_ankle_x + line_len, ankle_lower_y), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.line(img, (mid_ankle_x - line_len, ankle_higher_y), 
                 (mid_ankle_x + line_len, ankle_higher_y), (255, 255, 0), 2, cv2.LINE_AA)

        # --- Display metric value ---
        h_img, w_img = img.shape[:2]
        
        # Main value (ratio) - larger with outline
        label_main = f"Scissor Ratio: {scissor_ratio:.3f}"
        cv2.putText(img, label_main, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label_main, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Percentage - clearer formatting
        percentage = scissor_ratio * 100
        label_pct = f"= {percentage:.1f}% of leg height"
        cv2.putText(img, label_pct, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Frame number - larger and clearer, top right
        frame_label = f"Frame: {f_idx}"
        # Get text size for right alignment
        (text_w, text_h), _ = cv2.getTextSize(frame_label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        frame_x = w_img - text_w - 15
        cv2.putText(img, frame_label, (frame_x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, frame_label, (frame_x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return img
    
    def smart_resize(self, img, min_side=600):
        h, w = img.shape[:2]
        current_min = min(h, w)
        if current_min < min_side:
            scale = min_side / current_min
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC), scale
        return img, 1.0
    
    def process_jump(self, sel_jump, metric_name, interactive=True):
        ann_path = self.root / 'dataset' / 'annotations' / sel_jump / 'train' / f'annotations_interpolated_jump{int(sel_jump[2:])}.coco.json'
        frames_dir = self.root / 'dataset' / 'frames' / sel_jump
        save_dir = Path(__file__).parent / 'frame_overlays' / sel_jump / metric_name
        save_dir.mkdir(parents=True, exist_ok=True)

        if not ann_path.exists():
            print(f"❌ Annotations not found for {sel_jump}: {ann_path}")
            return

        with open(ann_path, 'r') as f: coco = json.load(f)
        
        kpt_names = None
        cat_with_kpts = next((c for c in coco.get('categories', []) if 'keypoints' in c), None)
        if cat_with_kpts: kpt_names = cat_with_kpts['keypoints']
        else: return
                # Initialize telemark_scissor value (used only for telemark_scissor metric)
        telemark_summary_val = 0.0
                # --- LOGICA SELEZIONE DATAFRAME ---
        if metric_name == 'v_style_angle':
            mask = self.df['v_style_angle_front'].notna() | self.df['v_style_angle_back'].notna()
            df_view = self.df[(self.df['jump_id'] == sel_jump) & mask].copy()

        elif metric_name == 'telemark_scissor':
            # Show 20-30 frames AFTER landing (athlete is in the air before)
            landing_frame = None
            if self.phases_path is not None and self.phases_path.exists():
                df_phases = pd.read_csv(self.phases_path)
                df_phases['jump_id'] = df_phases['jump_id'].apply(self._normalize_jid)
                row = df_phases[df_phases['jump_id'] == sel_jump]
                if not row.empty and pd.notna(row.iloc[0].get('landing')):
                    landing_frame = int(row.iloc[0]['landing'])

            if landing_frame is not None:
                start_f = landing_frame
                end_f = landing_frame + 30
                df_view = self.df[(self.df['jump_id'] == sel_jump) & 
                                  (self.df['frame_idx'] >= start_f) & 
                                  (self.df['frame_idx'] <= end_f)].copy()
            else:
                # Fallback: use landing phase flag
                df_view = self.df[(self.df['jump_id'] == sel_jump) & 
                                  (self.df['is_landing_phase'] == 1)].copy()

            # Read the summary value (one per jump) from metrics_summary_per_jump.csv
            telemark_summary_val = 0.0
            summary_path = self.summary_metrics_path
            if summary_path is not None and summary_path.exists():
                try:
                    df_summ = pd.read_csv(summary_path)
                    df_summ['jump_id'] = df_summ['jump_id'].apply(self._normalize_jid)
                    s_row = df_summ[df_summ['jump_id'] == sel_jump]
                    if not s_row.empty and 'telemark_scissor_mean' in df_summ.columns:
                        v = s_row.iloc[0]['telemark_scissor_mean']
                        if pd.notna(v):
                            telemark_summary_val = float(v)
                except Exception as e:
                    print(f"⚠️ Could not read telemark_scissor_mean from summary: {e}")

        else:
            df_view = self.df[(self.df['jump_id'] == sel_jump) & (self.df[metric_name].notna())].copy()
            
        df_view = df_view.sort_values('frame_idx')
        
        if df_view.empty:
            if interactive: print("No frames found for this metric.")
            return

        print(f" -> Processing {sel_jump}: {len(df_view)} frames...")

        # --- COSTRUZIONE MAPPE ---
        ann_map = {a['image_id']: a for a in coco['annotations']}
        img_map = {img['file_name']: img['id'] for img in coco['images']}
        
        # Mapping Frame Index -> JSON Filename
        frame_lookup = {}
        for fname in img_map.keys():
            import re
            m = re.search(r"(\d+)", fname)
            if m: 
                # Assumiamo che il primo numero trovato sia il frame index
                frame_lookup[int(m.group(1))] = fname

        indices = list(df_view.index)
        curr_ptr = 0
        
        while curr_ptr < len(indices):
            row = df_view.loc[indices[curr_ptr]]
            f_idx = int(row['frame_idx'])
            
            # --- RECUPERO VALORE METRICA ---
            val = 0.0
            view_str = ""
            if metric_name == 'v_style_angle':
                if pd.notna(row.get('v_style_angle_front')):
                    val = row['v_style_angle_front']; view_str = "FRONT"
                elif pd.notna(row.get('v_style_angle_back')):
                    val = row['v_style_angle_back']; view_str = "BACK"
            else:
                val = row.get(metric_name, 0.0)
            
            # --- FIX: GESTIONE NOME FILE E PATH IMMAGINE ---
            json_fname = frame_lookup.get(f_idx)
            
            # Creiamo una lista di candidati per il nome del file su disco
            file_candidates = []
            if json_fname: 
                file_candidates.append(json_fname) # 1. Prova il nome esatto del JSON
            
            # 2. Prova il formato standard a 5 cifre (es. 00413.jpg)
            file_candidates.append(f"{f_idx:05d}.jpg")
            # 3. Prova il formato semplice (es. 413.jpg)
            file_candidates.append(f"{f_idx}.jpg")
            
            found_img_path = None
            for cand in file_candidates:
                if (frames_dir / cand).exists():
                    found_img_path = frames_dir / cand
                    break
            
            if not found_img_path: 
                print(f"⚠️ SKIP: Immagine non trovata per frame {f_idx}. Cercato: {file_candidates}")
                curr_ptr += 1; continue
                
            img = cv2.imread(str(found_img_path))
            if img is None: 
                print(f"⚠️ SKIP: Impossibile leggere immagine {found_img_path}")
                curr_ptr += 1; continue

            # Se non abbiamo trovato il nome nel JSON, non possiamo trovare l'annotazione
            if not json_fname:
                print(f"⚠️ SKIP: Frame {f_idx} non trovato nel JSON delle annotazioni")
                curr_ptr += 1; continue

            img_id = img_map.get(json_fname) # Usiamo il nome del JSON per prendere l'ID
            ann = ann_map.get(img_id)
            
            if ann and 'bbox' in ann:
                # --- VISUALIZZAZIONE (INVARIATA) ---
                crop_img, offset = self.crop_to_skier(img, ann, margin=0.5)
                kpts = ann['keypoints']
                crop_img, scale_factor = self.smart_resize(crop_img, min_side=600)

                if metric_name == 'v_style_angle':
                    crop_img = self.draw_v_style(crop_img, offset, kpts, kpt_names, val, f_idx, view_str, scale=scale_factor)
                elif 'symmetry_index_back' in metric_name:
                    crop_img = self.draw_symmetry(crop_img, offset, kpts, kpt_names, val, f_idx, scale=scale_factor)
                elif 'body_ski' in metric_name:
                    crop_img = self.draw_body_ski(crop_img, offset, kpts, kpt_names, val, f_idx, scale=scale_factor)
                elif 'takeoff_knee' in metric_name:
                    crop_img = self.draw_takeoff_knee(crop_img, offset, kpts, kpt_names, val, f_idx, scale=scale_factor)
                elif metric_name == 'telemark_scissor':
                    crop_img = self.draw_telemark_scissor(crop_img, offset, kpts, kpt_names, f_idx,
                                                          scale=scale_factor)

                save_path = save_dir / f"viz_{f_idx:05d}.jpg"
                cv2.imwrite(str(save_path), crop_img)
                
                if interactive:
                    cv2.imshow("Metrics Visualizer", crop_img)
                    k = cv2.waitKey(0)
                    if k == 27: cv2.destroyAllWindows(); return "EXIT" 
                    elif k == ord('b'): curr_ptr = max(0, curr_ptr - 1)
                    else: curr_ptr += 1
                else:
                    curr_ptr += 1
            else:
                print(f"⚠️ SKIP: Annotazione o Bbox mancante per frame {f_idx}")
                curr_ptr += 1
        
        if interactive: cv2.destroyAllWindows()
        return "DONE"

    def visualize_interactive(self):
        if not self.load_data(): return

        print("\n=== METRICS VISUALIZER ===")
        print(" 1) Single Jump Mode (Select jump -> then metric)")
        print(" 2) Batch Mode (Select metric -> Save ALL jumps with that metric)")
        mode = input("Choice (1 or 2): ").strip()

        jump_ids = self.get_jump_ids()
        
        supported_metrics = [
            'v_style_angle',
            'symmetry_index_back', 
            'body_ski_angle', 
            'takeoff_knee_angle',
            'telemark_scissor'
        ]

        if mode == '1':
            print("\n--- AVAILABLE JUMPS ---")
            for j in jump_ids: print(f" - {j}")
            sel_jump = input("\nEnter jump ID (e.g. 5): ").strip()
            if not sel_jump.startswith("JP"): sel_jump = f"JP{int(sel_jump):04d}"
            
            if sel_jump not in jump_ids: print("❌ Jump not found."); return
            metrics = self.get_available_metrics(sel_jump)
            if not metrics: print("❌ No metrics available."); return

            print(f"\n--- METRICS FOR {sel_jump} ---")
            for i, m in enumerate(metrics): print(f" {i+1}) {m}")
            try:
                m_idx = int(input("Choose metric number: ")) - 1
                metric_name = metrics[m_idx]
            except: print("❌ Invalid selection."); return
            self.process_jump(sel_jump, metric_name, interactive=True)

        elif mode == '2':
            print("\n--- AVAILABLE METRICS (Global) ---")
            for i, m in enumerate(supported_metrics): print(f" {i+1}) {m}")
            try:
                m_idx = int(input("Choose metric number: ")) - 1
                metric_name = supported_metrics[m_idx]
            except: print("❌ Invalid selection."); return

            print(f"\n🔍 Finding jumps for '{metric_name}'...")
            valid_jumps = []
            for jid in jump_ids:
                if metric_name in self.get_available_metrics(jid):
                    valid_jumps.append(jid)
            
            if not valid_jumps: print(f"❌ No jumps found for {metric_name}."); return
            print(f"Found {len(valid_jumps)} jumps.")
            confirm = input("Start batch processing and saving? (y/n): ").lower()
            
            if confirm == 'y':
                for i, jid in enumerate(valid_jumps):
                    print(f"[{i+1}/{len(valid_jumps)}] Processing {jid}...")
                    self.process_jump(jid, metric_name, interactive=False)
                print("\n Batch processing complete!")
            else:
                print("Cancelled.")

if __name__ == "__main__":
    viz = MetricsVisualizer()
    viz.visualize_interactive()