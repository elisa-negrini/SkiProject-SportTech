import json
import numpy as np
import os
import cv2
from pathlib import Path

class Normalizer:
    """
    Normalizza le annotazioni usando la tecnica scale_2 (Torso + Leg + Shoulders)
    con il bacino centrato in 0.5 e coordinate normalizzate tra 0 e 1.
    TUTTI i keypoints (inclusi gli sci) sono garantiti dentro [0, 1].
    """
    
    def __init__(self):
        self.dataset_dir = Path('dataset/annotations')
        
        # Mapping ID keypoints (basato sul tuo sistema)
        self.anchor_ids = {
            'neck': '2',
            'r_shoulder': '3', 'l_shoulder': '6',
            'r_hip': '17', 'l_hip': '10',
            'r_knee': '18', 'l_knee': '11',
            'r_ankle': '19', 'l_ankle': '12'
        }
    
    def get_point(self, kpts, name, kpt_names):
        """Recupera (x, y) dato il nome del keypoint."""
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
        """Calcola distanza euclidea tra due punti."""
        if p1 is None or p2 is None:
            return 0.0
        return np.linalg.norm(p1 - p2)
    
    def calculate_scale_and_root(self, kpts, kpt_names):
        """
        Calcola scale_2 (Torso + Leg + Shoulders) e il punto root (bacino).
        Ritorna: (root, scale_2) o (None, None) se il frame è invalido.
        """
        # Recupera punti chiave
        pts = {k: self.get_point(kpts, v, kpt_names) 
               for k, v in self.anchor_ids.items()}
        
        # Calcolo Centro Bacino (Root)
        root = None
        if pts['r_hip'] is not None and pts['l_hip'] is not None:
            root = (pts['r_hip'] + pts['l_hip']) / 2.0
        elif pts['r_hip'] is not None:
            root = pts['r_hip']
        elif pts['l_hip'] is not None:
            root = pts['l_hip']
        
        if root is None:
            return None, None
        
        # --- Calcolo Lunghezze Segmenti ---
        
        # Torso (Collo -> Bacino)
        torso_top = pts['neck']
        if torso_top is None and pts['r_shoulder'] is not None and pts['l_shoulder'] is not None:
            torso_top = (pts['r_shoulder'] + pts['l_shoulder']) / 2
        
        len_torso = self.get_dist(torso_top, root)
        if len_torso == 0:
            len_torso = 1.0  # Fallback minimo
        
        # Gambe (Media Destra/Sinistra)
        len_thigh_r = self.get_dist(pts['r_hip'], pts['r_knee'])
        len_shin_r = self.get_dist(pts['r_knee'], pts['r_ankle'])
        len_leg_r = len_thigh_r + len_shin_r
        
        len_thigh_l = self.get_dist(pts['l_hip'], pts['l_knee'])
        len_shin_l = self.get_dist(pts['l_knee'], pts['l_ankle'])
        len_leg_l = len_thigh_l + len_shin_l
        
        # Usa la gamba più lunga visibile
        len_leg_max = max(len_leg_r, len_leg_l)
        if len_leg_max == 0:
            len_leg_max = len_torso * 1.2  # Fallback euristico
        
        # Spalle
        len_shoulders = self.get_dist(pts['r_shoulder'], pts['l_shoulder'])
        
        # SCALA 2: Torso + Leg + Shoulders
        scale_2 = len_torso + len_leg_max + len_shoulders
        
        if scale_2 == 0:
            scale_2 = 1.0
        
        return root, scale_2
    
    def get_bounding_box(self, kpts):
        """
        Calcola il bounding box di tutti i keypoints visibili.
        Ritorna: (min_x, max_x, min_y, max_y)
        """
        visible_points = []
        for i in range(0, len(kpts), 3):
            x, y, v = kpts[i], kpts[i+1], kpts[i+2]
            if v > 0:
                visible_points.append([x, y])
        
        if not visible_points:
            return None
        
        visible_points = np.array(visible_points)
        min_x, min_y = visible_points.min(axis=0)
        max_x, max_y = visible_points.max(axis=0)
        
        return min_x, max_x, min_y, max_y
    
    def normalize_annotation(self, ann, kpt_names, root, scale):
        """
        Normalizza i keypoints di una singola annotazione.
        STRATEGIA CORRETTA:
        1. Normalizza usando scale_2 (corpo) centrato sul bacino → range circa [-1, 1]
        2. Trova il bounding box reale di TUTTI i punti (inclusi sci che possono sforare)
        3. Calcola il punto più estremo dal bacino (che è in 0,0)
        4. Scala tutto per far sì che il punto più lontano tocchi il bordo [0,1]
        5. Trasla tutto per mettere il bacino a (0.5, 0.5)
        """
        kpts = ann['keypoints'].copy()
        
        # FASE 1: Normalizzazione anatomica relativa al corpo
        # Bacino = (0, 0) nel sistema normalizzato
        normalized_points = []
        for i in range(0, len(kpts), 3):
            x, y, v = kpts[i], kpts[i+1], kpts[i+2]
            
            if v > 0:
                # Centra sul bacino e scala con scale_2
                x_rel = (x - root[0]) / scale
                y_rel = (y - root[1]) / scale
                normalized_points.append([x_rel, y_rel, v])
            else:
                normalized_points.append([0, 0, v])
        
        normalized_points = np.array(normalized_points)
        
        # FASE 2: Trova il bounding box di TUTTI i punti visibili
        visible_mask = normalized_points[:, 2] > 0
        if not np.any(visible_mask):
            # Nessun punto visibile
            for i in range(len(normalized_points)):
                kpts[i*3] = 0.5
                kpts[i*3 + 1] = 0.5
            ann['keypoints'] = kpts
            return ann
        
        visible_points = normalized_points[visible_mask, :2]
        
        min_x = visible_points[:, 0].min()
        max_x = visible_points[:, 0].max()
        min_y = visible_points[:, 1].min()
        max_y = visible_points[:, 1].max()
        
        # FASE 3: Calcola la distanza massima dal bacino (che è in 0,0)
        # Dobbiamo trovare quanto "spazio" serve in ogni direzione
        # considerando che il bacino deve stare a 0.5
        
        # Distanze dal bacino (0,0) ai bordi del bbox
        dist_left = abs(min_x)    # verso sinistra
        dist_right = abs(max_x)   # verso destra
        dist_top = abs(min_y)     # verso alto
        dist_bottom = abs(max_y)  # verso basso
        
        # Per centrare il bacino a 0.5, abbiamo 0.5 di spazio da ogni lato
        # Dobbiamo scalare in modo che la direzione più "affollata" 
        # usi al massimo 0.5 unità (con piccolo margine)
        
        # Margine di sicurezza (2% dello spazio disponibile)
        margin = 0.02
        available_half_space = 0.5 - margin  # 0.48
        
        # Trova la scala necessaria per ogni direzione
        # scala = spazio_disponibile / spazio_richiesto
        scale_x = available_half_space / max(dist_left, dist_right) if max(dist_left, dist_right) > 0 else 1.0
        scale_y = available_half_space / max(dist_top, dist_bottom) if max(dist_top, dist_bottom) > 0 else 1.0
        
        # Usa la scala più restrittiva (la minore) per mantenere proporzioni
        final_scale = min(scale_x, scale_y)
        
        # FASE 4: Applica trasformazione finale
        for i in range(len(normalized_points)):
            if normalized_points[i, 2] > 0:  # Se visibile
                x_norm = normalized_points[i, 0]
                y_norm = normalized_points[i, 1]
                
                # Scala (mantiene il bacino a 0,0)
                x_scaled = x_norm * final_scale
                y_scaled = y_norm * final_scale
                
                # Trasla per mettere il bacino a 0.5
                x_final = x_scaled + 0.5
                y_final = y_scaled + 0.5
                
                # Clamp per sicurezza (il margine dovrebbe prevenire sforamenti)
                x_final = np.clip(x_final, 0.0, 1.0)
                y_final = np.clip(y_final, 0.0, 1.0)
                
                kpts[i*3] = x_final
                kpts[i*3 + 1] = y_final
            else:
                # Punti non visibili al centro (bacino)
                kpts[i*3] = 0.5
                kpts[i*3 + 1] = 0.5
        
        ann['keypoints'] = kpts
        return ann
    
    def process(self, jump_num):
        """
        Processa un singolo salto: legge il JSON interpolato e crea quello normalizzato.
        
        Args:
            jump_num: Numero del salto (es. 8 per JP0008)
        
        Returns:
            bool: True se successo, False altrimenti
        """
        jump_id = f"JP{jump_num:04d}"
        jump_dir = self.dataset_dir / jump_id
        
        # File di input e output
        input_file = jump_dir / "train" / f"annotations_interpolated_jump{jump_num}.coco.json"
        output_file = jump_dir / "train" / f"annotations_normalized_jump{jump_num}.coco.json"
        
        if not input_file.exists():
            print(f"   ⚠️  File non trovato: {input_file}")
            return False
        
        try:
            # Carica il JSON
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            # Trova categoria con keypoints
            cat = next((c for c in data['categories'] if 'keypoints' in c), None)
            if not cat:
                print(f"   ⚠️  Nessuna categoria con keypoints trovata")
                return False
            
            kpt_names = cat['keypoints']
            
            # Normalizza ogni annotazione
            normalized_count = 0
            skipped_count = 0
            
            for ann in data['annotations']:
                if ann['category_id'] != cat['id']:
                    continue
                
                # Calcola root e scala per questo frame
                root, scale = self.calculate_scale_and_root(ann['keypoints'], kpt_names)
                
                if root is None:
                    print(f"   ⚠️  Frame invalido (nessun bacino): image_id={ann.get('image_id')}")
                    skipped_count += 1
                    continue
                
                # Normalizza l'annotazione
                self.normalize_annotation(ann, kpt_names, root, scale)
                normalized_count += 1
            
            # Salva il JSON normalizzato
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"   ✅ Normalizzati {normalized_count} frame -> {output_file.name}")
            if skipped_count > 0:
                print(f"   ⚠️  Saltati {skipped_count} frame invalidi")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Errore durante normalizzazione: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def visualize_normalization(self, jump_num):
        """
        Crea un'immagine di visualizzazione della normalizzazione simile al PNG fornito.
        Mostra tutti i frame normalizzati sovrapposti.
        
        Args:
            jump_num: Numero del salto
        
        Returns:
            bool: True se successo
        """
        jump_id = f"JP{jump_num:04d}"
        jump_dir = self.dataset_dir / jump_id
        
        input_file = jump_dir / "train" / f"annotations_normalized_jump{jump_num}.coco.json"
        output_image = jump_dir / "train" / f"visualization_normalized_jump{jump_num}.png"
        
        if not input_file.exists():
            print(f"   ⚠️  File normalizzato non trovato: {input_file}")
            return False
        
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            cat = next((c for c in data['categories'] if 'keypoints' in c), None)
            if not cat:
                return False
            
            kpt_names = cat['keypoints']
            skeleton = cat.get('skeleton', [])
            
            # Crea canvas (1000x1000 per buona risoluzione)
            img_size = 1000
            canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
            
            # Scala per convertire coordinate [0,1] in pixel
            def to_pixel(coord):
                return int(np.clip(coord, 0, 1) * (img_size - 1))
            
            # Colori diversi per ogni frame (ciclo attraverso una palette)
            num_frames = len([a for a in data['annotations'] if a['category_id'] == cat['id']])
            colors = []
            for i in range(num_frames):
                hue = int(180 * i / max(num_frames, 1))
                color = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
                colors.append(tuple(map(int, color)))
            
            # Disegna ogni frame
            color_idx = 0
            for ann in data['annotations']:
                if ann['category_id'] != cat['id']:
                    continue
                
                kpts = ann['keypoints']
                color = colors[color_idx % len(colors)]
                color_idx += 1
                
                # Converti keypoints in formato utilizzabile
                points = []
                for i in range(0, len(kpts), 3):
                    x, y, v = kpts[i], kpts[i+1], kpts[i+2]
                    if v > 0:
                        points.append((to_pixel(x), to_pixel(y)))
                    else:
                        points.append(None)
                
                # Disegna skeleton
                for connection in skeleton:
                    pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
                    if pt1_idx < len(points) and pt2_idx < len(points):
                        if points[pt1_idx] is not None and points[pt2_idx] is not None:
                            cv2.line(canvas, points[pt1_idx], points[pt2_idx], color, 1, cv2.LINE_AA)
                
                # Disegna keypoints
                for pt in points:
                    if pt is not None:
                        cv2.circle(canvas, pt, 2, color, -1, cv2.LINE_AA)
            
            # Aggiungi linee di riferimento per il centro (bacino a 0.5)
            center = img_size // 2
            cv2.line(canvas, (center, 0), (center, img_size), (200, 200, 200), 1, cv2.LINE_AA)
            cv2.line(canvas, (0, center), (img_size, center), (200, 200, 200), 1, cv2.LINE_AA)
            
            # Aggiungi bordi per il range [0, 1]
            border_color = (150, 150, 150)
            cv2.rectangle(canvas, (0, 0), (img_size-1, img_size-1), border_color, 2)
            
            # Aggiungi testo informativo
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, f"{jump_id} - Normalized", (10, 30), font, 0.7, (0, 0, 0), 2)
            cv2.putText(canvas, f"Frames: {color_idx}", (10, 60), font, 0.6, (0, 0, 0), 1)
            
            # Salva immagine
            cv2.imwrite(str(output_image), canvas)
            print(f"   ✅ Visualizzazione salvata: {output_image.name}")
            return True
            
        except Exception as e:
            print(f"   ❌ Errore durante visualizzazione: {e}")
            import traceback
            traceback.print_exc()
            return False