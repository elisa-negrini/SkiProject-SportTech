import streamlit as st
import pandas as pd
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
project_dataset = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))
dashboard_dataset = os.path.join(BASE_DIR, "dataset")
if os.path.exists(project_dataset):
    DATASET_DIR = project_dataset
elif os.path.exists(dashboard_dataset):
    DATASET_DIR = dashboard_dataset
else:
    DATASET_DIR = project_dataset

def _extract_metric_from_name(name: str) -> str:
    base = os.path.splitext(name)[0]
    base = base.replace('-', '_')
    keywords = ["apertura", "sci", "opening", "angle", "knee", "hip", "torso", "head", "tilt"]
    low = base.lower()
    for k in keywords:
        if k in low:
            return k.capitalize()
    cleaned = re.sub(r"\_?\d+$", "", base)
    return cleaned

def _extract_frame_number(name: str):
    nums = re.findall(r"\d+", name)
    if not nums: return None
    longest = max(nums, key=len)
    try: return int(longest)
    except: return None

@st.cache_data
def load_jump_phases(csv_name: str = "dataset/jump_phases_SkiTB.csv"):
    search_paths = [
        os.path.abspath(os.path.join(BASE_DIR, "..", csv_name)),
        os.path.join(BASE_DIR, csv_name)
    ]
    csv_path = None
    for p in search_paths:
        if os.path.exists(p):
            csv_path = p
            break

    phases = {}
    if not csv_path:
        return phases

    try:
        df_ph = pd.read_csv(csv_path)
        for _, row in df_ph.iterrows():
            jump_raw = str(row.get('jump_id', '')).strip()
            if not jump_raw:
                continue
            nums = re.findall(r"\d+", jump_raw)
            if not nums:
                continue
            n = int(nums[0])
            jp = f"JP{n:04d}"
            take_off = row.get('take_off_frame')
            landing = row.get('landing')
            take_off = int(take_off) if pd.notna(take_off) and str(take_off).strip() else None
            landing = int(landing) if pd.notna(landing) and str(landing).strip() else None
            
            phases[jp] = {"take_off": take_off, "landing": landing}
    except Exception:
        pass
    return phases

def _determine_phase(jump_id: str, frame_num: int, phases_map: dict) -> str:
    """
    Determines the flight phase based on jump_id and frame number.
    
    Phases:
    1. Descending: from the first frame up to 6 before take_off_frame (frame_num < take_off - 6)
    2. Take off: from 5 before to 5 after take_off_frame (take_off - 5 <= frame_num <= take_off + 5)
    3. Flight: from 6 after take_off_frame up to 9 before landing (take_off + 6 <= frame_num <= landing - 10)
    4. Landing: from 9 before landing onwards (frame_num >= landing - 9)
    """
    if frame_num is None:
        return 'Unknown'
    info = phases_map.get(jump_id)
    if not info:
        return 'Unknown'
    take_off = info.get('take_off')
    landing = info.get('landing')
    if take_off is None or landing is None:
        return 'Unknown'
    if frame_num < take_off - 6:
        return 'Descending'
    if take_off - 5 <= frame_num <= take_off + 5:
        return 'Take Off'
    if take_off + 6 <= frame_num <= landing - 10:
        return 'Flight'
    if frame_num >= landing - 9:
        return 'Landing'
    return 'Unknown'

@st.cache_data
def load_jp_athletes(csv_name: str = "dataset/JP_data.csv"):
    search_paths = [
        os.path.abspath(os.path.join(BASE_DIR, "..", csv_name)),
        os.path.join(BASE_DIR, csv_name)
    ]
    csv_path = None
    for p in search_paths:
        if os.path.exists(p):
            csv_path = p
            break

    mapping = {}
    if not csv_path:
        return mapping

    try:
        df_j = pd.read_csv(csv_path, dtype=str)
        for _, r in df_j.iterrows():
            jp_raw = r.get('ID') or r.get('Idnum') or r.get('Id')
            if jp_raw is None:
                continue
            jp_raw = str(jp_raw).strip()
            if jp_raw.upper().startswith('JP'):
                jp_key = jp_raw.upper()
            else:
                nums = re.findall(r"\d+", jp_raw)
                if not nums:
                    continue
                jp_key = f"JP{int(nums[0]):04d}"
            first = str(r.get('AthleteName') or '').strip()
            last = str(r.get('AthleteSurname') or '').strip()
            name = f"{first} {last}".strip()
            mapping[jp_key] = name if name else 'Skier_Unknown'
    except Exception:
        pass
    return mapping

@st.cache_data(show_spinner="Loading dataset...")
def load_data_from_folders(limit=100):
    limit = min(int(limit), 50000)
    data = []
    annotations_path = os.path.join(DATASET_DIR, "annotations")
    frames_base = os.path.join(DATASET_DIR, "frames")

    if not os.path.exists(annotations_path):
        st.error(f"Folder not found: {annotations_path}")
        return []

    phases_map = load_jump_phases()
    athletes_map = load_jp_athletes()
    count = 0
    if os.path.exists(annotations_path):
        jump_folders = sorted([f for f in os.listdir(annotations_path) if os.path.isdir(os.path.join(annotations_path, f))])
    else:
        jump_folders = []

    for folder_name in jump_folders:
        if count >= limit: break
        nums = re.findall(r"\d+", folder_name)
        if not nums:
            continue
        jump_id = f"JP{int(nums[0]):04d}"
        frames_folder = os.path.join(frames_base, folder_name)
        annotations_jump_folder = os.path.join(annotations_path, folder_name)
        if os.path.exists(frames_folder):
            frame_files = sorted([f for f in os.listdir(frames_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for img_name in frame_files:
                if count >= limit: break
                frame_num = _extract_frame_number(img_name)
                phase = _determine_phase(jump_id, frame_num, phases_map)
                data.append({
                    "jump_id": jump_id,
                    "skier": athletes_map.get(jump_id, "Skier_Unknown"),
                    "metric_clean": "Raw Frame",
                    "image_path": os.path.join(frames_folder, img_name),
                    "has_skeleton": False,
                    "source": "frames",
                    "phase": phase
                })
                count += 1
        vis_folder = os.path.join(annotations_jump_folder, "visualizations")
        if os.path.exists(vis_folder):
            vis_files = sorted([f for f in os.listdir(vis_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for img_name in vis_files:
                if count >= limit: break
                frame_num = _extract_frame_number(img_name)
                phase = _determine_phase(jump_id, frame_num, phases_map)
                metric_name = _extract_metric_from_name(img_name)
                data.append({
                    "jump_id": jump_id,
                    "skier": athletes_map.get(jump_id, "Skier_Unknown"),
                    "metric_clean": metric_name,
                    "image_path": os.path.join(vis_folder, img_name),
                    "has_skeleton": True,
                    "source": "visualizations",
                    "phase": phase
                })
                count += 1
                
    return data