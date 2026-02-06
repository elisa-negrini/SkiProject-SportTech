import json
import os
from pathlib import Path

class AnnotationManager:
    def __init__(self, raw_root='annotation_preprocessing/raw_annotations/', output_root='./dataset/annotations'):
        self.raw_root = Path(raw_root)
        self.output_root = Path(output_root)
        self.source_ann_path = self.raw_root / 'train' / '_annotations.coco.json'

    def extract_jump(self, jump_number):
        jump_id = f"JP{jump_number:04d}"
        tag_to_search = f"jump{jump_number}"
        
        target_dir = self.output_root / jump_id / 'train'
        target_ann_path = target_dir / f"annotations_jump{jump_number}.json"
        
        target_dir.mkdir(parents=True, exist_ok=True)

        if not self.source_ann_path.exists():
            print(f"❌ Error: Source file not found: {self.source_ann_path}")
            return False

        with open(self.source_ann_path, 'r') as f:
            coco_data = json.load(f)

        # Filter images by tag
        images_to_keep = []
        image_ids_to_keep = set()
        
        for image in coco_data['images']:
            user_tags = image.get('extra', {}).get('user_tags', [])
            if tag_to_search in user_tags:
                images_to_keep.append(image)
                image_ids_to_keep.add(image['id'])

        if not images_to_keep:
            print(f"❌ No images found for tag {tag_to_search}")
            return False

        # Filter annotations
        annotations_to_keep = [
            ann for ann in coco_data['annotations'] 
            if ann['image_id'] in image_ids_to_keep
        ]

        new_coco_data = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'categories': coco_data.get('categories', []),
            'images': images_to_keep,
            'annotations': annotations_to_keep
        }

        with open(target_ann_path, 'w') as f:
            json.dump(new_coco_data, f, indent=4)
        
        print(f"✅ Extracted {len(images_to_keep)} images to {target_ann_path}")
        return True