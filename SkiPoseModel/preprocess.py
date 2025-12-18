import os
import json
import numpy as np
import pickle as pkl
import glob
import random
import sys

# in questo script carichiamo i json e creiamo il dataset diviso in train e test in formato pkl
# dopodiche definiamo il nostro datamodule

DATASET_ROOT = "dataset/annotations"

OUTPUT_DIR = "dataset_processed" 
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_RATIO = 0.8

# Mapping (Target : Source)
# Target: The ID you want in the final pkl
# Source: The ID found inside the COCO JSON file
# NOTE: Python uses 0-based indexing. Below, I convert your map (which is 1-based)
# by subtracting 1 from everything.
USER_MAP_1_BASED = {
    1: 1, 2: 6, 3: 3, 4: 4, 5: 5, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11,
    11: 12, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21, 17: 13, 18: 14,
    19: 2, 20: 16, 21: 15, 22: 22, 23: 23
}

# Create 0-based map for array usage
KEYPOINT_MAP = {k-1: v-1 for k, v in USER_MAP_1_BASED.items()}

NUM_TARGET_JOINTS = 23
def process_coco_json(json_path):
    """
    Reads a COCO JSON file and extracts poses mapped according to KEYPOINT_MAP.
    Returns a list of numpy arrays (N_joints, 3) -> (x, y, confidence)
    """
    extracted_poses = []
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)

    # COCO files have an 'annotations' list
    annotations = data.get('annotations', [])
    
    for ann in annotations:
        # COCO keypoints is a flat list: [x1, y1, v1, x2, y2, v2, ...]
        raw_kps = np.array(ann['keypoints'])
        
        # Reshape to (N, 3) to have rows of (x, y, v)
        # N is calculated automatically (-1)
        raw_kps = raw_kps.reshape(-1, 3)
        
        # Create empty container for target skeleton (23 joints)
        # We use 3 channels: x, y, confidence (v)
        target_skel = np.zeros((NUM_TARGET_JOINTS, 3), dtype=np.float32)
        
        # Fill the new skeleton using the map
        for target_idx, source_idx in KEYPOINT_MAP.items():
            # Safety check: if the JSON has fewer joints than requested
            if source_idx < len(raw_kps):
                target_skel[target_idx] = raw_kps[source_idx]
            else:
                # If joint is missing in source, leave as 0,0,0
                pass
        
        extracted_poses.append(target_skel)

    return extracted_poses

def main():
    all_skeletons = []

    # 1. SEARCH FILES
    # Use glob to find all files matching the pattern
    # dataset/annotations/*/train/*.json
    # The double asterisk ** would be for recursive search, but given your fixed structure,
    # we build the specific pattern.
    
    search_pattern = os.path.join(DATASET_ROOT, "*", "train", "*.json")
    json_files = glob.glob(search_pattern)
    
    print(f"Found {len(json_files)} JSON files to process.")
    
    # 2. DATA EXTRACTION
    for fpath in json_files:
        print(f"Processing: {fpath}")
        poses = process_coco_json(fpath)
        all_skeletons.extend(poses)
        print(f" -> Extracted {len(poses)} poses.")

    # Convert to numpy array for convenience
    all_skeletons = np.array(all_skeletons) # Shape: (Total_Poses, 23, 3)
    
    print(f"\nTotal extracted poses: {all_skeletons.shape[0]}")

    # 3. TRAIN / TEST SPLIT (RANDOM)
    # Create a list of indices and shuffle them
    num_samples = len(all_skeletons)
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    split_point = int(num_samples * TRAIN_RATIO)
    
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    train_data = all_skeletons[train_indices]
    test_data = all_skeletons[test_indices]
    
    print(f"Train dimensions: {train_data.shape}")
    print(f"Test dimensions: {test_data.shape}")

    # 4. SAVE TO PKL
    # Create dictionaries as in the original file
    dict_train = {'openpose_2d': train_data}
    dict_test = {'openpose_2d': test_data}
    
    # Save Train
    train_path = os.path.join(OUTPUT_DIR, "train.pkl")
    with open(train_path, 'wb') as handle:
        pkl.dump(dict_train, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print(f"Train set saved to: {train_path}")

    # Save Test
    test_path = os.path.join(OUTPUT_DIR, "test.pkl")
    with open(test_path, 'wb') as handle:
        pkl.dump(dict_test, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print(f"Test set saved to: {test_path}")

if __name__ == "__main__":
    main()