import json
import cv2
import numpy as np
import re
from pathlib import Path
import sys

class Visualizer:
    def __init__(self, dataset_root="./dataset"):
        self.root = Path(dataset_root)
        
        self.keypoint_number_map = {
            1: 1, 2: 6, 3: 3, 4: 4, 5: 5, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11,
            11: 12, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21, 17: 13, 18: 14,
            19: 2, 20: 16, 21: 15, 22: 22, 23: 23
        }
        
        self.number_color = (0, 255, 200)
        
        self.keypoint_colors = {
            1: (255, 210, 0),   2: (0, 0, 0),       3: (255, 128, 0),
            4: (255, 128, 0),   5: (255, 128, 0),   6: (0, 128, 255),
            7: (0, 128, 255),   8: (0, 128, 255),   9: (0, 0, 0),
            10: (0, 255, 255),  11: (0, 255, 255),  12: (0, 255, 255),
            13: (128, 0, 255),  14: (128, 0, 255),  15: (128, 0, 255),
            16: (128, 0, 255),  17: (255, 0, 128),  18: (255, 0, 128),
            19: (255, 0, 128),  20: (128, 0, 255),  21: (128, 0, 255),
            22: (128, 0, 255),  23: (128, 0, 255),
        }
        
        self.connection_colors = {
            (1, 2): (255, 210, 0),
            (2, 6): (0, 128, 255), (6, 7): (0, 128, 255), (7, 8): (0, 128, 255),
            (2, 3): (255, 128, 0), (3, 4): (255, 128, 0), (4, 5): (255, 128, 0),
            (2, 9): (0, 0, 0),
            (9, 10): (0, 255, 255), (10, 11): (0, 255, 255), (11, 12): (0, 255, 255),
            (12, 13): (0, 255, 255), (12, 14): (0, 255, 255),
            (9, 17): (255, 0, 128), (17, 18): (255, 0, 128), (18, 19): (255, 0, 128),
            (19, 20): (255, 0, 128), (19, 21): (255, 0, 128),
            (20, 23): (128, 0, 255), (21, 22): (128, 0, 255),
            (13, 16): (128, 0, 255), (14, 15): (128, 0, 255),
        }
        
        self.bbox_color = (0, 255, 0)
        self.interpolated_alpha = 0.6
        
        # Impostazioni visualizzazione
        self.target_height = 800  # Altezza fissa dell'immagine di output (alta risoluzione)
        self.line_thickness = 3   # Spessore linee
        self.point_radius = 6     # Raggio punti

    def _draw_skeleton(self, image, keypoints, skeleton, is_interpolated=False):
        kp = np.array(keypoints).reshape(-1, 3)
        overlay = image.copy() if is_interpolated else image
        
        # Disegna connessioni
        for connection in skeleton:
            pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1
            if pt1_idx < len(kp) and pt2_idx < len(kp):
                x1, y1, v1 = kp[pt1_idx]
                x2, y2, v2 = kp[pt2_idx]
                if v1 > 0 and v2 > 0:
                    display_num1 = self.keypoint_number_map.get(pt1_idx + 1, pt1_idx + 1)
                    display_num2 = self.keypoint_number_map.get(pt2_idx + 1, pt2_idx + 1)
                    
                    color = self.connection_colors.get((display_num1, display_num2))
                    if color is None:
                        color = self.connection_colors.get((display_num2, display_num1))
                    if color is None:
                        color = self.keypoint_colors.get(display_num1, (255, 255, 255))
                    
                    cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, self.line_thickness)
        
        # Disegna punti
        for i, (x, y, v) in enumerate(kp):
            if v > 0:
                original_num = i + 1
                display_num = self.keypoint_number_map.get(original_num, original_num)
                color = self.keypoint_colors.get(display_num, (255, 255, 255))
                
                cv2.circle(overlay, (int(x), int(y)), self.point_radius, color, -1)
                
                # Numero del punto (piccolo e vicino al punto)
                text_x = int(x) + 8
                text_y = int(y) - 8
                cv2.putText(overlay, str(display_num), (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.number_color, 1)
        
        if is_interpolated:
            cv2.addWeighted(overlay, self.interpolated_alpha, image, 1 - self.interpolated_alpha, 0, image)
        else:
            image[:] = overlay
        
        return image

    def _extract_frame_number(self, name: str) -> int:
        m = re.search(r"(\d+)", name)
        if not m: return 0
        return int(m.group(1))

    def _get_image_name(self, img):
        if "extra" in img and isinstance(img["extra"], dict) and "name" in img["extra"]:
            return img["extra"]["name"]
        return img["file_name"]

    def _get_crop_coordinates(self, image_shape, bbox, padding_percentage=0.50):
        img_h, img_w = image_shape[:2]
        x, y, w, h = [int(v) for v in bbox]

        pad_w = int(w * padding_percentage)
        pad_h = int(h * padding_percentage)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        return x1, y1, x2, y2

    def _adjust_and_scale_keypoints(self, keypoints, crop_offset_x, crop_offset_y, scale_factor):
        """
        1. Trasla i keypoint (sottrae coordinate crop)
        2. Scala i keypoint (moltiplica per fattore di zoom)
        """
        adjusted = []
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
            if v > 0:
                # Traslazione e Scalatura
                new_x = (x - crop_offset_x) * scale_factor
                new_y = (y - crop_offset_y) * scale_factor
                adjusted.extend([new_x, new_y, v])
            else:
                adjusted.extend([0, 0, 0])
        return adjusted

    def run_visualization(self):
        try:
            val = input("Inserisci il numero del salto da visualizzare (es. 5): ")
            jump_number = int(val)
        except ValueError:
            print("Errore: Inserisci un numero intero valido.")
            return

        jump_id = f"JP{jump_number:04d}"
        ann_path = self.root / f"annotations/{jump_id}/train/annotations_interpolated_jump{jump_number}.coco.json"
        frames_dir = self.root / f"frames/{jump_id}"
        out_dir = Path(f"./jump_{jump_number}_zoomed_clean")
        
        if not ann_path.exists():
            print(f"❌ Annotazioni non trovate: {ann_path}")
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Creazione immagini in: {out_dir.resolve()}")

        with open(ann_path, 'r') as f:
            coco = json.load(f)

        skeleton = []
        for cat in coco.get("categories", []):
            if cat["name"] == "skier":
                skeleton = cat.get("skeleton", [])
                break

        ann_map = {a["image_id"]: a for a in coco["annotations"]}
        frames_list = []
        
        # Raccolta frame
        for img in coco["images"]:
            try:
                img_name = self._get_image_name(img)
                frame_num = self._extract_frame_number(img_name)
                ann = ann_map.get(img["id"])
                if ann:
                    frames_list.append((frame_num, img, ann))
            except ValueError:
                continue
        
        frames_list.sort(key=lambda x: x[0])
        
        print(f"Processing {len(frames_list)} frames...")
        
        for frame_num, img, ann in frames_list:
            img_name = self._get_image_name(img)
            img_path = frames_dir / img_name
            
            if not img_path.exists():
                continue
            
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # --- PROCESSO: CROP -> INGRANDISCI -> OVERLAY ---
            if "bbox" in ann:
                # 1. Calcola le coordinate del ritaglio
                x1, y1, x2, y2 = self._get_crop_coordinates(image.shape, ann["bbox"], padding_percentage=0.50)
                
                # 2. Ritaglia l'immagine originale
                cropped_image = image[y1:y2, x1:x2].copy()
                
                # 3. Calcola il fattore di scala per arrivare all'altezza target (es. 800px)
                # Questo evita che immagini piccole abbiano scheletri giganti
                h, w = cropped_image.shape[:2]
                if h > 0 and w > 0:
                    scale_factor = self.target_height / float(h)
                    new_width = int(w * scale_factor)
                    new_height = self.target_height
                    
                    # 4. Ingrandisci l'immagine (Upscale con interpolazione cubica per qualità)
                    upscaled_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    
                    # 5. Adatta i keypoints alla nuova scala e disegna
                    if "keypoints" in ann and skeleton:
                        adjusted_kps = self._adjust_and_scale_keypoints(ann["keypoints"], x1, y1, scale_factor)
                        upscaled_image = self._draw_skeleton(upscaled_image, adjusted_kps, skeleton, is_interpolated=False)
                    
                    # Salva
                    output_file = out_dir / f"frame_{frame_num:05d}.jpg"
                    cv2.imwrite(str(output_file), upscaled_image)
            
            if frame_num % 20 == 0:
                print(f"✓ Frame {frame_num} elaborato")

        print("✅ Finito! Immagini salvate in:", out_dir)

if __name__ == "__main__":
    viz = Visualizer()
    viz.run_visualization()