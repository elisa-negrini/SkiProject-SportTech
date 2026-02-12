import os
import json
import numpy as np
import pickle as pkl
import random
import glob
import domainadapt_flags
from absl import app, flags
import re

FLAGS = flags.FLAGS

def process_coco_json(json_path, num_joints):
    if not os.path.exists(json_path):
        print(f"Skipping {json_path}, not found.")
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)

    img_map = {img['id']: img['file_name'] for img in data.get('images', [])}
    base_dir = os.path.dirname(json_path) 
    
    annotations = data.get('annotations', [])
    extracted_poses = []
    image_paths = []

    USER_MAP_1_BASED = {
        1: 1, 2: 19, 3: 3, 4: 4, 5: 5,
        6: 2, 7: 6, 8: 7,
        9: 8, 10: 9, 11: 10, 12: 11,
        13: 17, 14: 18, 15: 21, 16: 20,
        17: 12, 18: 13, 19: 14,
        20: 15, 21: 16, 22: 22, 23: 23
    }
    KEYPOINT_MAP = {k-1: v for k, v in USER_MAP_1_BASED.items()}

    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in img_map:
            continue
        full_path = os.path.join(base_dir, img_map[image_id])
        raw_kps = np.array(ann['keypoints'])
        target_skel = np.zeros((num_joints, 3), dtype=np.float32)

        for t_idx, source_id in KEYPOINT_MAP.items():
            s_idx = (source_id - 1) * 3
            
            if s_idx + 2 < len(raw_kps):
                target_skel[t_idx, 0] = raw_kps[s_idx]    
                target_skel[t_idx, 1] = raw_kps[s_idx + 1] 
                target_skel[t_idx, 2] = raw_kps[s_idx + 2] 
        
        extracted_poses.append(target_skel)
        image_paths.append(full_path)
    
    return np.array(extracted_poses), image_paths

def main(argv):
    DATASET_ROOT = FLAGS.raw_dataset_dir
    OUTPUT_DIR = FLAGS.dataset_dir
    TRAIN_RATIO = FLAGS.train_split
    NUM_TARGET_JOINTS = FLAGS.n_joints
    
    print("Starting preprocessing...")
    print(f" Input:  {DATASET_ROOT}")
    print(f" Output: {OUTPUT_DIR}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    json_files = []
    json_files = glob.glob(os.path.join(DATASET_ROOT, "*.json"))
    if not json_files:
        json_files = glob.glob(os.path.join(DATASET_ROOT, "**", "*.json"), recursive=True)

    print(f"Found {len(json_files)} JSON files.")

    all_skeletons = []
    all_paths = []
    all_jumps = []

    for fpath in json_files:
        filename = os.path.basename(fpath)
        match = re.search(r'(jump\d+)', filename)
        if match:
            jump_id = match.group(1)
        else:
            jump_id = filename.split('.')[0] 
        poses, paths = process_coco_json(fpath, NUM_TARGET_JOINTS)
        if len(poses) > 0:
            all_skeletons.extend(poses)
            all_paths.extend(paths)
            all_jumps.extend([jump_id] * len(poses))
            print(f" -> {os.path.basename(fpath)}: {len(poses)} poses.")

    all_skeletons = np.array(all_skeletons)
    all_paths = np.array(all_paths)
    all_jumps = np.array(all_jumps)
    print(f"\nTotal extracted poses: {all_skeletons.shape[0]}")

    if len(all_skeletons) == 0:
        print("❌ No poses extracted. Check paths.")
        return

    indices = list(range(len(all_skeletons)))
    random.seed(42)
    random.shuffle(indices)

    train_split = int(len(all_skeletons) * TRAIN_RATIO)
    val_split = int(len(all_skeletons) * FLAGS.val_split)
    
    train_data = all_skeletons[indices[:train_split]]
    train_paths = all_paths[indices[:train_split]]

    val_data = all_skeletons[indices[train_split:val_split]]
    val_paths = all_paths[indices[train_split:val_split]]

    test_data = all_skeletons[indices[val_split:]]
    test_paths = all_paths[indices[val_split:]]

    jump_data = np.array(all_jumps)[indices]
    
    dict_train = {'openpose_2d': train_data, 'image_paths': train_paths, 'jump_ids': jump_data[:train_split]}
    dict_val = {'openpose_2d': val_data, 'image_paths': val_paths, 'jump_ids': jump_data[train_split:val_split]}
    dict_test = {'openpose_2d': test_data, 'image_paths': test_paths, 'jump_ids': jump_data[val_split:]}
    
    with open(os.path.join(OUTPUT_DIR, "train.pkl"), 'wb') as f:
        pkl.dump(dict_train, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(os.path.join(OUTPUT_DIR, "val.pkl"), 'wb') as f:
        pkl.dump(dict_val, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    with open(os.path.join(OUTPUT_DIR, "test.pkl"), 'wb') as f:
        pkl.dump(dict_test, f, protocol=pkl.HIGHEST_PROTOCOL)
        
    print(f" Files saved in {OUTPUT_DIR}")
    print(f" Train: {train_data.shape[0]} | Val: {val_data.shape[0]} | Test: {test_data.shape[0]}")

if __name__ == "__main__":
    app.run(main)