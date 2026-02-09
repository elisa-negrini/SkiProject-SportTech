import json
import cv2
import numpy as np
import re
from pathlib import Path

class Visualizer:
    def __init__(self, dataset_root="./dataset"):
        self.root = Path(dataset_root)
        
        # Keypoint conversion map
        self.keypoint_number_map = {
            1: 1, 2: 6, 3: 3, 4: 4, 5: 5, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11,
            11: 12, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21, 17: 13, 18: 14,
            19: 2, 20: 16, 21: 15, 22: 22, 23: 23
        }
        
        self.number_color = (0, 255, 200)
        
        self.keypoint_colors = {
            1: (255, 210, 0),   # Cyan - head/neck
            2: (0, 0, 0),       # Black - center torso
            3: (255, 128, 0),   # Light Blue - left shoulder
            4: (255, 128, 0),   # Light Blue - left elbow
            5: (255, 128, 0),   # Light Blue - left wrist
            6: (0, 128, 255),   # Orange - right shoulder
            7: (0, 128, 255),   # Orange - right elbow
            8: (0, 128, 255),   # Orange - right wrist
            9: (0, 0, 0),       # Black - center pelvis
            10: (0, 255, 255),  # Yellow - left hip
            11: (0, 255, 255),  # Yellow - left knee
            12: (0, 255, 255),  # Yellow - left ankle
            13: (128, 0, 255),  # Magenta - right hip
            14: (128, 0, 255),  # Magenta - right knee
            15: (128, 0, 255),  # Magenta - right ankle
            16: (128, 0, 255),  # Dark Violet - right foot
            17: (255, 0, 128),  # Violet - left shoulder/side
            18: (255, 0, 128),  # Violet - left torso
            19: (255, 0, 128),  # Violet - left pelvis
            20: (128, 0, 255),  # Pink - posterior left hip
            21: (128, 0, 255),  # Pink - posterior left knee
            22: (128, 0, 255),  # Pink - posterior center
            23: (128, 0, 255),  # Pink - left foot
        }
        
        self.connection_colors = {
            # Head and Neck
            (1, 2): (255, 210, 0),      # Cyan
            # Right Arm (Orange)
            (2, 6): (0, 128, 255),
            (6, 7): (0, 128, 255),
            (7, 8): (0, 128, 255),
            # Left Arm (Light Blue)
            (2, 3): (255, 128, 0),
            (3, 4): (255, 128, 0),
            (4, 5): (255, 128, 0),
            # Center Torso
            (2, 9): (0, 0, 0),          # Black
            # Left Leg (Yellow)
            (9, 10): (0, 255, 255),
            (10, 11): (0, 255, 255),
            (11, 12): (0, 255, 255),
            (12, 13): (0, 255, 255),
            (12, 14): (0, 255, 255),
            # Right Leg (Violet)
            (9, 17): (255, 0, 128),
            (17, 18): (255, 0, 128),
            (18, 19): (255, 0, 128),
            (19, 20): (255, 0, 128),
            (19, 21): (255, 0, 128),
            # Ski/Foot Connections (Pink/Violet)
            (20, 23): (128, 0, 255),
            (21, 22): (128, 0, 255),
            (13, 16): (128, 0, 255),
            (14, 15): (128, 0, 255),
        }
        
        self.bbox_color = (0, 255, 0)  # Green
        self.interpolated_alpha = 0.6

    def _draw_skeleton(self, image, keypoints, skeleton, is_interpolated=False):
        """Draws the skeleton with the same design as visualize_interpolation.py"""
        kp = np.array(keypoints).reshape(-1, 3)
        
        overlay = image.copy() if is_interpolated else image
        
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
                    
                    cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        
        for i, (x, y, v) in enumerate(kp):
            if v > 0:
                original_num = i + 1
                display_num = self.keypoint_number_map.get(original_num, original_num)
                color = self.keypoint_colors.get(display_num, (255, 255, 255))
                
                cv2.circle(overlay, (int(x), int(y)), 7, color, -1)
                
                text_x = int(x) + 10
                text_y = int(y) - 10
                cv2.putText(overlay, str(display_num), (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.number_color, 1)
        
        if is_interpolated:
            cv2.addWeighted(overlay, self.interpolated_alpha, image, 1 - self.interpolated_alpha, 0, image)
        else:
            image[:] = overlay
        
        return image

    def _draw_bbox(self, image, bbox, is_interpolated=False):
        """Draws the bounding box"""
        x, y, w, h = [int(v) for v in bbox]
        
        if is_interpolated:
            overlay = image.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), self.bbox_color, 2)
            cv2.addWeighted(overlay, self.interpolated_alpha, image, 1 - self.interpolated_alpha, 0, image)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), self.bbox_color, 2)
        
        return image

    def _extract_frame_number(self, name: str) -> int:
        """Extracts the frame number from the file name"""
        m = re.search(r"(\d+)", name)
        if not m:
            raise ValueError(f"No number found in '{name}'")
        return int(m.group(1))

    def _get_image_name(self, img):
        """Returns the logical image name"""
        if "extra" in img and isinstance(img["extra"], dict) and "name" in img["extra"]:
            return img["extra"]["name"]
        return img["file_name"]

    def generate_images(self, jump_number):
        """Generate visualization images for a jump"""
        jump_id = f"JP{jump_number:04d}"
        ann_path = self.root / f"annotations/{jump_id}/train/annotations_interpolated_jump{jump_number}.coco.json"
        frames_dir = self.root / f"frames/{jump_id}"
        out_dir = self.root / f"annotations/{jump_id}/visualizations"
        
        if not ann_path.exists():
            print(f"❌ JSON not found: {ann_path}")
            return False
            
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading annotations for {jump_id}...")
        with open(ann_path, 'r') as f:
            coco = json.load(f)
        
        skeleton = []
        for cat in coco.get("categories", []):
            if cat["name"] == "skier":
                skeleton = cat.get("skeleton", [])
                break

        ann_map = {a["image_id"]: a for a in coco["annotations"]}
        
        original_images_ids = set()
        
        frames = []
        for img in coco["images"]:
            try:
                img_name = self._get_image_name(img)
                frame_num = self._extract_frame_number(img_name)
                ann = ann_map.get(img["id"])
                if ann:
                    frames.append((frame_num, img, ann))
            except ValueError as e:
                print(f"Skipping image {self._get_image_name(img)}: {e}")
        
        frames.sort(key=lambda x: x[0])
        
        print(f"Processing {len(frames)} frames for {jump_id}...")
        
        for frame_num, img, ann in frames:
            img_name = self._get_image_name(img)
            img_path = frames_dir / img_name
            
            if not img_path.exists():
                print(f"⚠️ Image not found: {img_path}")
                continue
            
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"⚠️ Could not load: {img_path}")
                continue
            
            is_interpolated = img["id"] not in original_images_ids if original_images_ids else False
            
            if "bbox" in ann:
                image = self._draw_bbox(image, ann["bbox"], is_interpolated)
            
            if "keypoints" in ann and skeleton:
                image = self._draw_skeleton(image, ann["keypoints"], skeleton, is_interpolated)
            
            label = f"Frame {frame_num}" + (" [INTERPOLATED]" if is_interpolated else "")
            font_scale = 1.0
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            cv2.rectangle(image, (5, 5), (15 + text_width, 35 + text_height), (0, 0, 0), -1)
            cv2.rectangle(image, (5, 5), (15 + text_width, 35 + text_height), (255, 255, 255), 2)
            
            color = (255, 165, 0) if is_interpolated else (0, 255, 0)  # Orange or Green
            cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, thickness)
            
            output_file_path = out_dir / f"frame_{frame_num:05d}.jpg"
            cv2.imwrite(str(output_file_path), image)
            
            if frame_num % 10 == 0:
                print(f"✓ Processed frame {frame_num}")
        
        print(f" Visualization saved to {out_dir}")
        return True

    def create_video(self, jump_number, fps=10):
        """Create video from visualization images"""
        jump_id = f"JP{jump_number:04d}"
        img_dir = self.root / f"annotations/{jump_id}/visualizations"
        out_vid = img_dir / f"output_video_{jump_id}.mp4"
        
        images = sorted(list(img_dir.glob("frame_*.jpg")))
        if not images:
            print("❌ No images found for video.")
            return False

        print(f"Creating video for {jump_id}...")
        frame0 = cv2.imread(str(images[0]))
        h, w, _ = frame0.shape
        
        writer = cv2.VideoWriter(
            str(out_vid), 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (w, h)
        )

        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                writer.write(frame)
        
        writer.release()
        print(f"✅ Video created: {out_vid}")
        return True