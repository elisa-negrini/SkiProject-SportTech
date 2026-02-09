import json
import re
import copy
import glob
import os
from pathlib import Path

class Interpolator:
    def __init__(self, dataset_root="dataset"):
        self.root = Path(dataset_root)

    def _extract_frame_num(self, name):
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else 0

    def _interpolate_list(self, list_a, list_b, t):
        return [a + t * (b - a) for a, b in zip(list_a, list_b)]

    def _normalize_kp(self, kp, bbox):
        x_b, y_b, w, h = bbox
        norm = []
        for i in range(0, len(kp), 3):
            x, y, v = kp[i:i+3]
            norm.extend([(x - x_b)/w if w else 0, (y - y_b)/h if h else 0, v])
        return norm

    def _denormalize_kp(self, kp_norm, bbox):
        x_b, y_b, w, h = bbox
        new_kp = []
        for i in range(0, len(kp_norm), 3):
            nx, ny, v = kp_norm[i:i+3]
            new_kp.extend([round(nx * w + x_b, 3), round(ny * h + y_b, 3), int(v)])
        return new_kp

    def _interp_norm_kp(self, kp_a, kp_b, t):
        res = []
        for i in range(0, len(kp_a), 3):
            xa, ya, va = kp_a[i:i+3]
            xb, yb, vb = kp_b[i:i+3]
            res.extend([xa + t * (xb - xa), ya + t * (yb - ya), va]) # Keep visibility
        return res

    def process(self, jump_number):
        jump_id = f"JP{jump_number:04d}"
        ann_dir = self.root / "annotations" / jump_id / "train"
        frames_dir = self.root / "frames" / jump_id
        
        input_json = ann_dir / f"annotations_jump{jump_number}.json"
        output_json = ann_dir / f"annotations_interpolated_jump{jump_number}.coco.json"
        boxes_file = frames_dir / "boxes_filtered.txt"

        if not input_json.exists() or not boxes_file.exists():
            print(f"❌ Missing JSON or boxes_filtered.txt for {jump_id}")
            return False

        with open(input_json, "r") as f:
            coco = json.load(f)
        
        bbox_list = []
        with open(boxes_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('['):
                    bbox_list.append([int(float(x)) for x in line.strip().split(',')])

        def get_idx(path):
            s = set()
            for p in glob.glob(str(path / "*.jpg")):
                try: s.add(self._extract_frame_num(os.path.basename(p)))
                except: pass
            return s

        idx_main = get_idx(frames_dir)
        idx_rem = get_idx(frames_dir / 'removed')
        idx_occ = get_idx(frames_dir / 'occluded')
        
        all_idx = idx_main | idx_rem | idx_occ
        to_remove = all_idx - idx_main
        kept_frames = sorted(list(all_idx - to_remove), key=int)

        if len(kept_frames) != len(bbox_list):
            print(f"❌ Sync Error: {len(kept_frames)} frames vs {len(bbox_list)} boxes.")
            return False

        bbox_map = {k: v for k, v in zip(kept_frames, bbox_list)}

        img_id_map = {}
        for img in coco["images"]:
            fname = img.get("extra", {}).get("name", img["file_name"])
            img_id_map[img["id"]] = self._extract_frame_num(fname)

        for ann in coco["annotations"]:
            fnum = img_id_map.get(ann["image_id"])
            if fnum in bbox_map:
                ann["bbox"] = bbox_map[fnum]

        ann_by_img = {a["image_id"]: a for a in coco["annotations"]}
        annotated_ids = set(ann_by_img.keys())
        
        frames_idx = []
        for img in coco["images"]:
            fnum = img_id_map[img["id"]]
            if img["id"] in annotated_ids:
                frames_idx.append((fnum, img))
        frames_idx.sort(key=lambda x: x[0])

        new_imgs = []
        new_anns = []
        next_img_id = max((i['id'] for i in coco['images']), default=0) + 1
        next_ann_id = max((a['id'] for a in coco['annotations']), default=0) + 1

        for (fa, img_a), (fb, img_b) in zip(frames_idx[:-1], frames_idx[1:]):
            if fb <= fa + 1: continue

            ann_a, ann_b = ann_by_img[img_a["id"]], ann_by_img[img_b["id"]]
            bbox_a, bbox_b = ann_a["bbox"], ann_b["bbox"]
            kp_a = self._normalize_kp(ann_a["keypoints"], bbox_a)
            kp_b = self._normalize_kp(ann_b["keypoints"], bbox_b)

            for cur in range(fa + 1, fb):
                t = (cur - fa) / (fb - fa)
                
                new_img = copy.deepcopy(img_a)
                new_img["id"] = next_img_id
                
                base_name = img_a["file_name"]
                prefix = re.match(r"(.*?)(\d+)", base_name).group(1)
                suffix = base_name.split('.')[-1]
                pad = len(re.search(r"(\d+)", base_name).group(1))
                new_name = f"{prefix}{cur:0{pad}d}.{suffix}"
                
                new_img["file_name"] = new_name
                if "extra" in new_img: new_img["extra"]["name"] = new_name
                
                new_imgs.append(new_img)
                next_img_id += 1

                new_ann = copy.deepcopy(ann_a)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = new_img["id"]
                next_ann_id += 1

                target_bbox = bbox_map.get(cur, self._interpolate_list(bbox_a, bbox_b, t))
                new_ann["bbox"] = target_bbox
                
                kp_interp = self._interp_norm_kp(kp_a, kp_b, t)
                new_ann["keypoints"] = self._denormalize_kp(kp_interp, target_bbox)
                
                new_anns.append(new_ann)

        coco["images"].extend(new_imgs)
        coco["annotations"].extend(new_anns)

        with open(output_json, "w") as f:
            json.dump(coco, f, indent=2)
        
        print(f" Interpolation done. Added {len(new_anns)} frames.")
        return True