import pandas as pd
import numpy as np
import cv2
import json
import os
from pathlib import Path

class MetricsVisualizer:
    def __init__(self, dataset_root='dataset', metrics_file='metrics/core_metrics/metrics_per_frame.csv'):
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
            'telemark_depth_back_ratio'
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
    
    def draw_symmetry(self, img, offset, keypoints, kpt_names, metric_val, f_idx, scale=1.0):
        # 1. Axis (neck -> pelvis)
        p_neck = self.get_kpt_pos(keypoints, 'neck', kpt_names)
        p_pelvis = self.get_kpt_pos(keypoints, 'center_pelvis', kpt_names)
        if p_neck is not None: p_neck = (p_neck - offset) * scale
        if p_pelvis is not None: p_pelvis = (p_pelvis - offset) * scale
            
        # 2. Skis
        tr = self.get_kpt_pos(keypoints, 'r_ski_tail', kpt_names)
        pr = self.get_kpt_pos(keypoints, 'r_ski_tip', kpt_names)
        tl = self.get_kpt_pos(keypoints, 'l_ski_tail', kpt_names)
        pl = self.get_kpt_pos(keypoints, 'l_ski_tip', kpt_names)
        
        skis_ok = all(x is not None for x in [tr, pr, tl, pl])
        if skis_ok:
            tr = (tr - offset) * scale; pr = (pr - offset) * scale
            tl = (tl - offset) * scale; pl = (pl - offset) * scale

        # 3. Build axis
        p_pelvis_core, vec_body_axis = None, None
        p_bottom_extended = None
        
        if p_neck is not None and p_pelvis is not None:
            vec_core = p_pelvis - p_neck # Verso il BASSO
            len_core = np.linalg.norm(vec_core)
            if len_core > 0:
                vec_body_axis = vec_core / len_core
                p_pelvis_core = p_pelvis
                p_top_draw = p_neck - vec_body_axis * (len_core * 0.30)
                p_bottom_draw = p_pelvis + vec_body_axis * (len_core * 4.00) 
                p_bottom_extended = p_pelvis + vec_body_axis * 10000 

                cv2.line(img, tuple(p_top_draw.astype(int)), tuple(p_pelvis.astype(int)), (255, 0, 0), 2, cv2.LINE_AA)
                self.draw_dashed_line(img, p_pelvis, p_bottom_draw, (255, 0, 0), 2, dash_len=15)

        if skis_ok and p_pelvis_core is not None and p_bottom_extended is not None:
            def get_angle_between(v1, v2):
                v1_u = v1 / np.linalg.norm(v1)
                v2_u = v2 / np.linalg.norm(v2)
                dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
                deg = np.degrees(np.arccos(dot))
                if deg > 90: deg = 180 - deg
                return deg

            # A. Intersezioni
            int_r = self.find_intersection(tr, pr, p_pelvis_core, p_bottom_extended)
            int_l = self.find_intersection(tl, pl, p_pelvis_core, p_bottom_extended)
            
            # B. Compute target height (mid-thigh) using average ski length
            len_ski = (np.linalg.norm(pr - tr) + np.linalg.norm(pl - tl)) / 2
            
            # Project 50% of ski length from pelvis as target distance
            dist_target_from_pelvis = len_ski * 0.5 
            
            # Funzione proiezione distanza su asse
            def get_dist_on_axis(p):
                # Distanza proiettata dal bacino (positiva verso il basso)
                return np.dot(p - p_pelvis_core, vec_body_axis)

            # Distanza target assoluta dal bacino
            target_proj_d = dist_target_from_pelvis

            # Calcolo raggi per raggiungere quel target dai rispettivi vertici
            # Radius = |Posizione Vertice - Posizione Target|
            radius_r = abs(get_dist_on_axis(int_r) - target_proj_d) if int_r is not None else 50
            radius_l = abs(get_dist_on_axis(int_l) - target_proj_d) if int_l is not None else 50
            
            # C. Disegno
            vec_up_axis = -vec_body_axis 

            if int_r is not None:
                self.draw_dashed_line(img, tr, int_r, (0, 255, 255), 1, dash_len=8)
                vec_ski_vis_r = tr - int_r 
                ang_val_r = get_angle_between(pr - tr, vec_body_axis)
                self.draw_angle_arc_with_label(img, int_r, vec_up_axis, vec_ski_vis_r, (0, 255, 255), ang_val_r, radius=int(radius_r))

            if int_l is not None:
                self.draw_dashed_line(img, tl, int_l, (255, 0, 255), 1, dash_len=8)
                vec_ski_vis_l = tl - int_l
                ang_val_l = get_angle_between(pl - tl, vec_body_axis)
                self.draw_angle_arc_with_label(img, int_l, vec_up_axis, vec_ski_vis_l, (255, 0, 255), ang_val_l, radius=int(radius_l))

            cv2.line(img, tuple(tr.astype(int)), tuple(pr.astype(int)), (0, 0, 255), 3)
            cv2.line(img, tuple(tl.astype(int)), tuple(pl.astype(int)), (0, 0, 255), 3)
            
            #self.draw_label_with_box(img, [f"Sym Index: {metric_val:.2f}", f"Frame: {f_idx}"], (10, img.shape[0] - 25))
            
        return img
    
    def draw_angle_arc_with_label(self, img, center, vec_start, vec_end, color_fill, val_deg, radius=50):
        """Draw a filled arc and a value label."""
        # Calcola angoli in gradi
        ang_start = np.degrees(np.arctan2(vec_start[1], vec_start[0]))
        ang_end = np.degrees(np.arctan2(vec_end[1], vec_end[0]))
        
        # Normalize for ellipse drawing
        if ang_end - ang_start > 180: ang_end -= 360
        if ang_start - ang_end > 180: ang_start -= 360
        
        start_draw = min(ang_start, ang_end)
        end_draw = max(ang_start, ang_end)
        
        # 1. Disegna Arco/Settore
        overlay = img.copy()
        cv2.ellipse(overlay, tuple(center.astype(int)), (radius, radius), 
                   0, start_draw, end_draw, color_fill, -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img) # Trasparenza
        
        # 2. Etichetta con sfondo nero
        # Posizione: a metà dell'arco, un po' più distante dal raggio
        mid_ang_rad = np.radians((start_draw + end_draw) / 2)
        label_dist = radius + 20
        lx = center[0] + np.cos(mid_ang_rad) * label_dist
        ly = center[1] + np.sin(mid_ang_rad) * label_dist
        
        text = f"{val_deg:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        # Sfondo Nero
        tl = (int(lx - w/2 - 2), int(ly - h/2 - 2))
        br = (int(lx + w/2 + 2), int(ly + h/2 + 2))
        cv2.rectangle(img, tl, br, (0,0,0), -1)
        
        # Testo del colore dell'area
        cv2.putText(img, text, (int(lx - w/2), int(ly + h/2)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_fill, 2, cv2.LINE_AA)
        return img

    def draw_label_with_box(self, img, text_lines, pos_target):
        if isinstance(text_lines, str): text_lines = [text_lines]
        
        h_img, w_img = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        pad = 4
        line_spacing = 4
        
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

    def draw_v_style(self, img, offset, keypoints, kpt_names, metric_val, f_idx, view_type="", scale=1.0):
        tr = self.get_kpt_pos(keypoints, 'r_ski_tail', kpt_names)
        pr = self.get_kpt_pos(keypoints, 'r_ski_tip', kpt_names)
        tl = self.get_kpt_pos(keypoints, 'l_ski_tail', kpt_names)
        pl = self.get_kpt_pos(keypoints, 'l_ski_tip', kpt_names)
        
        if any(x is None for x in [tr, pr, tl, pl]): return img
        
        # Applica Scala
        tr = (tr - offset) * scale; pr = (pr - offset) * scale
        tl = (tl - offset) * scale; pl = (pl - offset) * scale
        
        # Disegna Sci Solidi
        cv2.line(img, tuple(tr.astype(int)), tuple(pr.astype(int)), (0, 0, 255), 3)
        cv2.line(img, tuple(tl.astype(int)), tuple(pl.astype(int)), (0, 0, 255), 3)
        
        vertex = self.find_intersection(tr, pr, tl, pl)
        
        if vertex is not None:
            # --- DASHED LINES TO VERTEX ---
            # Draw dashed lines from tails (tr/tl) to intersection (vertex)
            self.draw_dashed_line(img, tr, vertex, (100, 100, 255), 1, dash_len=8)
            self.draw_dashed_line(img, tl, vertex, (100, 100, 255), 1, dash_len=8)
            
            ## --- MODIFICA RAGGIO (Verso le Punte) ---
            # Calcola distanza dal vertice alle PUNTE (pr, pl) invece che alle code
            dist_pr = np.linalg.norm(pr - vertex)
            dist_pl = np.linalg.norm(pl - vertex)
            
            # Usa il 60% della distanza verso la punta (o un'altra % a piacere)
            radius = int(min(dist_pr, dist_pl) * 0.6) 
            
            # (Opzionale) Limita il raggio se necessario, es. max 200px
            # radius = min(radius, 200)

            # Vettori direzione (dal vertice verso la punta)
            vec_r = pr - vertex
            vec_l = pl - vertex
            
            # Disegna Arco
            self.draw_angle_arc_with_label(img, vertex, vec_r, vec_l, (0, 255, 0), metric_val, radius)

        # Etichetta standard in basso
        label_lines = [f"V-Style: {metric_val:.2f} deg | Frame: {f_idx}"]
        h_img = img.shape[0]
        #self.draw_label_with_box(img, label_lines, (10, h_img - 25))

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

    def draw_body_ski(self, img, offset, keypoints, kpt_names, metric_val, f_idx, scale=1.0):
        # --- 1. Punti Corpo (Spalla -> Caviglia) ---
        shoulders = [self.get_kpt_pos(keypoints, s+'shoulder', kpt_names) for s in ['r_', 'l_']]
        ankles = [self.get_kpt_pos(keypoints, s+'ankle', kpt_names) for s in ['r_', 'l_']]
        
        # Filtra None e calcola medie
        shoulders = [p for p in shoulders if p is not None]
        ankles = [p for p in ankles if p is not None]
        
        p_body_top, p_body_bot = None, None
        if shoulders and ankles:
            avg_sh = np.mean(shoulders, axis=0)
            avg_ank = np.mean(ankles, axis=0)
            # Applica Offset e Scala
            p_body_top = (avg_sh - offset) * scale
            p_body_bot = (avg_ank - offset) * scale
            
            # Disegna linea solida Ciano
            cv2.line(img, tuple(p_body_top.astype(int)), tuple(p_body_bot.astype(int)), (255, 255, 0), 3)

        # --- 2. Punti Sci (Punta -> Coda) ---
        tips = [self.get_kpt_pos(keypoints, s+'ski_tip', kpt_names) for s in ['r_', 'l_']]
        tails = [self.get_kpt_pos(keypoints, s+'ski_tail', kpt_names) for s in ['r_', 'l_']]
        tips = [p for p in tips if p is not None]
        tails = [p for p in tails if p is not None]
        
        p_ski_tip, p_ski_tail = None, None
        if tips and tails:
            avg_tip = np.mean(tips, axis=0)
            avg_tail = np.mean(tails, axis=0)
            # Applica Offset e Scala
            p_ski_tip = (avg_tip - offset) * scale
            p_ski_tail = (avg_tail - offset) * scale
            
            # Disegna linea solida Rossa
            cv2.line(img, tuple(p_ski_tip.astype(int)), tuple(p_ski_tail.astype(int)), (0, 0, 255), 3)

        # --- 3. Calcolo Intersezione e Angolo Colorato ---
        if p_body_top is not None and p_ski_tip is not None:
            # Trova l'intersezione tra la retta del corpo e quella dello sci
            vertex = self.find_intersection(p_body_top, p_body_bot, p_ski_tip, p_ski_tail)
            
            if vertex is not None:
                # Disegna linee tratteggiate verso l'intersezione (per far capire dove si incontrano)
                # Dal piede al vertice
                self.draw_dashed_line(img, p_body_bot, vertex, (255, 255, 0), 1, dash_len=6)
                # Dalla coda dello sci al vertice
                self.draw_dashed_line(img, p_ski_tail, vertex, (0, 0, 255), 1, dash_len=6)
                
                # Definisci i vettori uscenti dal vertice per disegnare l'arco
                # Vettore 1: Verso la spalla
                vec_body = p_body_top - vertex
                # Vettore 2: Verso la punta dello sci
                vec_ski = p_ski_tip - vertex
                
                # --- MODIFICA QUI ---
                # Calcola la distanza tra l'intersezione e la punta dello sci
                dist_to_tip = np.linalg.norm(vec_ski)
                
                # Imposta il raggio al 60% della lunghezza disponibile verso la punta
                dynamic_radius = int(dist_to_tip * 0.6) 
                
                # Se preferisci un limite minimo (per non averlo minuscolo se l'intersezione è vicina)
                dynamic_radius = max(dynamic_radius, 50)

                # Disegna con il nuovo raggio
                self.draw_angle_arc_with_label(img, vertex, vec_body, vec_ski, (0, 255, 255), metric_val, radius=dynamic_radius)
        # --- 4. Etichetta Box ---
        label_lines = [f"BSA: {metric_val:.2f} deg | Frame: {f_idx}"]
        h_img = img.shape[0]
        fixed_pos = (10, h_img - 25) 
        #self.draw_label_with_box(img, label_lines, fixed_pos)
            
        return img
    
    def draw_telemark(self, img, offset, keypoints, kpt_names, metric_val, f_idx, scale=1.0):
        ank_r = self.get_kpt_pos(keypoints, 'r_ankle', kpt_names)
        ank_l = self.get_kpt_pos(keypoints, 'l_ankle', kpt_names)
        tips = [self.get_kpt_pos(keypoints, s+'ski_tip', kpt_names) for s in ['r_', 'l_']]
        tails = [self.get_kpt_pos(keypoints, s+'ski_tail', kpt_names) for s in ['r_', 'l_']]
        
        # Filtra None e applica offset
        tips = [(p - offset) * scale for p in tips if p is not None]
        tails = [(p - offset) * scale for p in tails if p is not None]

        if ank_r is not None and ank_l is not None:
            # Applica offset e scala alle caviglie
            ank_r = (ank_r - offset) * scale
            ank_l = (ank_l - offset) * scale
            
            # --- 1. SCI ROSSI TRASPARENTI ---
            if tips and tails:
                overlay = img.copy() # Crea un livello trasparente
                # Disegna linee rosse su overlay
                avg_tip = np.mean(tips, axis=0).astype(int)
                avg_tail = np.mean(tails, axis=0).astype(int)
                # Linea rossa spessa
                cv2.line(overlay, tuple(avg_tip), tuple(avg_tail), (0, 0, 255), 4) 
                
                # Fonde overlay e immagine originale (0.4 = 40% visibilità linee)
                cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

            # --- 2. GEOMETRIA (Cerchi e Linee Gialle) ---
            # Disegnati DOPO la trasparenza per rimanere brillanti
            ar, al = ank_r.astype(int), ank_l.astype(int)
            cv2.circle(img, tuple(ar), 6, (0, 255, 255), -1) 
            cv2.circle(img, tuple(al), 6, (0, 255, 255), -1)
            cv2.line(img, tuple(ar), tuple(al), (0, 255, 255), 1)

            # --- 3. BARRA LATERALE (Piccola a Destra) ---
            h, w = img.shape[:2]
            bar_w = 15           # Molto più stretta
            bar_h = int(h * 0.4) # Alta solo il 40% dell'immagine
            pad_right = 10
            
            x_start = w - bar_w - pad_right
            y_start = int((h - bar_h) / 2)
            y_end = y_start + bar_h
            
            # Sfondo barra
            cv2.rectangle(img, (x_start, y_start), (x_start + bar_w, y_end), (50, 50, 50), -1)
            
            # Riempimento dinamico
            max_val = 0.8 # Valore di riferimento per "ottimo"
            fill_ratio = min(metric_val / max_val, 1.0)
            fill_h = int(bar_h * fill_ratio)
            
            color_bar = (0, 0, 255) 
            if fill_ratio > 0.4: color_bar = (0, 255, 255)
            if fill_ratio > 0.7: color_bar = (0, 255, 0)
            
            # Disegna livello
            cv2.rectangle(img, (x_start, y_end - fill_h), (x_start + bar_w, y_end), color_bar, -1)
            cv2.rectangle(img, (x_start, y_start), (x_start + bar_w, y_end), (200, 200, 200), 1)

            # --- 4. TESTO UNICA RIGA (In basso a sinistra) ---
            # Unisco le stringhe con " | " per averle su una riga sola
            label_text = f"Telemark Ratio: {metric_val:.2f} | Frame: {f_idx}"
            
            # Posizione fissa in basso a sinistra (con font piccolo basta meno spazio dal bordo)
            fixed_pos = (10, h - 25) 
            #self.draw_label_with_box(img, [label_text], fixed_pos)
            
        return img
    
    def smart_resize(self, img, min_side=600):
        """Ingrandisce e restituisce anche il fattore di scala."""
        h, w = img.shape[:2]
        current_min = min(h, w)
        
        if current_min < min_side:
            scale = min_side / current_min
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC), scale
        return img, 1.0
    
    def process_jump(self, sel_jump, metric_name, interactive=True):
        """
        Processa un singolo salto.
        Se interactive=True: mostra le finestre e aspetta input (Spazio/B/Esc).
        Se interactive=False: salva tutti i frame e passa al successivo automaticamente.
        """
        # Paths setup
        # NOTA: Assicurati che il percorso 'annotations' sia corretto rispetto alla tua struttura
        ann_path = self.root / 'dataset' / 'annotations' / sel_jump / 'train' / f'annotations_interpolated_jump{int(sel_jump[2:])}.coco.json'
        frames_dir = self.root / 'dataset' / 'frames' / sel_jump
        
        # Cartella output: creiamo una sottocartella per ogni salto dentro la cartella della metrica
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
        
        # Prepare data filter
        if metric_name == 'v_style_angle':
            mask = self.df['v_style_angle_front'].notna() | self.df['v_style_angle_back'].notna()
            df_view = self.df[(self.df['jump_id'] == sel_jump) & mask].copy()
        else:
            df_view = self.df[(self.df['jump_id'] == sel_jump) & (self.df[metric_name].notna())].copy()
            
        df_view = df_view.sort_values('frame_idx')
        
        if df_view.empty:
            if interactive: print("No frames found for this metric.")
            return

        print(f" -> Processing {sel_jump}: {len(df_view)} frames...")
        
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
            
            # Determine value
            val = 0.0
            view_str = ""
            if metric_name == 'v_style_angle':
                if pd.notna(row.get('v_style_angle_front')):
                    val = row['v_style_angle_front']; view_str = "FRONT"
                elif pd.notna(row.get('v_style_angle_back')):
                    val = row['v_style_angle_back']; view_str = "BACK"
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
                # --- RENDER LOGIC ---
                crop_img, offset = self.crop_to_skier(img, ann, margin=0.5)
                kpts = ann['keypoints']
                crop_img, scale_factor = self.smart_resize(crop_img, min_side=600)

                if metric_name == 'v_style_angle':
                    crop_img = self.draw_v_style(crop_img, offset, kpts, kpt_names, val, f_idx, view_str, scale=scale_factor)
                elif 'symmetry_index_back' in metric_name:
                    crop_img = self.draw_symmetry(crop_img, offset, kpts, kpt_names, val, f_idx, scale=scale_factor)
                elif 'body_ski' in metric_name:
                    crop_img = self.draw_body_ski(crop_img, offset, kpts, kpt_names, val, f_idx, scale=scale_factor)
                elif 'telemark_depth' in metric_name: 
                    crop_img = self.draw_telemark(crop_img, offset, kpts, kpt_names, val, f_idx, scale=scale_factor)
                else:
                    h_img = crop_img.shape[0]
                    #self.draw_label_with_box(crop_img, [f"{metric_name}: {val:.2f}", f"Frame: {f_idx}"], (10, h_img - 25))

                # SAVE
                save_path = save_dir / f"viz_{f_idx:05d}.jpg"
                cv2.imwrite(str(save_path), crop_img)
                
                # INTERACTIVE CONTROL
                if interactive:
                    cv2.imshow("Metrics Visualizer", crop_img)
                    k = cv2.waitKey(0)
                    if k == 27: # ESC
                        cv2.destroyAllWindows()
                        return "EXIT" 
                    elif k == ord('b'): curr_ptr = max(0, curr_ptr - 1)
                    else: curr_ptr += 1
                else:
                    # Batch mode: avanti veloce
                    curr_ptr += 1
            else:
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
            'telemark_depth_back_ratio'
        ]

        # --- MODO 1: Interattivo Singolo ---
        if mode == '1':
            print("\n--- AVAILABLE JUMPS ---")
            for j in jump_ids: print(f" - {j}")
            
            sel_jump = input("\nEnter jump ID (e.g. 5): ").strip()
            if not sel_jump.startswith("JP"): sel_jump = f"JP{int(sel_jump):04d}"
            
            if sel_jump not in jump_ids:
                print("❌ Jump not found."); return

            metrics = self.get_available_metrics(sel_jump)
            if not metrics:
                print("❌ No metrics available."); return

            print(f"\n--- METRICS FOR {sel_jump} ---")
            for i, m in enumerate(metrics): print(f" {i+1}) {m}")
            
            try:
                m_idx = int(input("Choose metric number: ")) - 1
                metric_name = metrics[m_idx]
            except:
                print("❌ Invalid selection."); return
            
            # Chiama la funzione in modalità interattiva
            self.process_jump(sel_jump, metric_name, interactive=True)


        # --- MODO 2: Batch per Metrica ---
        elif mode == '2':
            print("\n--- AVAILABLE METRICS (Global) ---")
            for i, m in enumerate(supported_metrics): print(f" {i+1}) {m}")
            
            try:
                m_idx = int(input("Choose metric number: ")) - 1
                metric_name = supported_metrics[m_idx]
            except:
                print("❌ Invalid selection."); return

            print(f"\n🔍 Finding jumps for '{metric_name}'...")
            valid_jumps = []
            for jid in jump_ids:
                if metric_name in self.get_available_metrics(jid):
                    valid_jumps.append(jid)
            
            if not valid_jumps:
                print(f"❌ No jumps found for {metric_name}."); return

            print(f"Found {len(valid_jumps)} jumps: {valid_jumps}")
            confirm = input("Start batch processing and saving? (y/n): ").lower()
            
            if confirm == 'y':
                for i, jid in enumerate(valid_jumps):
                    print(f"[{i+1}/{len(valid_jumps)}] Processing {jid}...")
                    self.process_jump(jid, metric_name, interactive=False)
                print("\n✅ Batch processing complete!")
            else:
                print("Cancelled.")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent.parent
    viz = MetricsVisualizer(dataset_root=root_dir)
    viz.visualize_interactive()
