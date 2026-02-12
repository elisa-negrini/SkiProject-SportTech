import streamlit as st
import pandas as pd
from PIL import Image
import os
import sys

CURRENT_PAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_PAGE_DIR)

sys.path.append(PARENT_DIR)

from utils import load_data_from_folders

st.set_page_config(page_title="Ski Jump Dataset Gallery", page_icon="📂", layout="wide")
st.sidebar.title("Gallery Settings")
image_limit = st.sidebar.slider("Max images to load", min_value=50, max_value=50000, value=50000, step=50)

try:
    raw_data = load_data_from_folders(limit=image_limit)
    df = pd.DataFrame(raw_data)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

col_text, col_img = st.columns([5, 1], gap="medium", vertical_alignment="bottom")

with col_text:
    st.title("Ski Jump Dataset Gallery")
    st.markdown("""
    Here you can explore the results of our annotation project applied to the **[Ski-TB Dataset](https://machinelearning.uniud.it/datasets/skitb/)**, focused on competitive **ski jumping**.

    The visualization highlights our custom **23-keypoint skeleton** structure. Both original and annotated frames are shown, and you have the possibility to explore the dataset by filtering per jump, athlete, jump phase, and frame type.
    """)

with col_img:
    skeleton_path = os.path.join(PARENT_DIR, "skeleton.png")
    if os.path.exists(skeleton_path):
        st.image(skeleton_path, caption="23-Keypoint Skeleton", width='stretch')
    else:
        st.caption("Skeleton legend not found.")

st.divider()

if df.empty:
    st.warning("No images found. Please check that 'dataset' folder exists inside 'dashboard'.")
    st.stop()

st.subheader("Filter Gallery")
c1, c2, c3, c4 = st.columns(4)

with c1:
    vis_filter = st.selectbox("1. Visualization Type", ["All", "Original Frames (Raw)", "Annotated Skeletons"])
with c2:
    all_jumps = sorted(df['jump_id'].unique().tolist())
    jump_filter = st.multiselect("2. Filter by Jump ID", all_jumps)
with c3:
    all_athletes = sorted(df['skier'].unique().tolist())
    athlete_filter = st.multiselect("3. Filter by Athlete", all_athletes)
with c4:
    all_phases = sorted([str(p) for p in df['phase'].unique() if p is not None])
    phase_filter = st.selectbox("4. Filter by Phase", ["All"] + all_phases)

filtered_df = df.copy()

if vis_filter == "Original Frames (Raw)":
    filtered_df = filtered_df[filtered_df['source'] == 'frames']
elif vis_filter == "Annotated Skeletons":
    filtered_df = filtered_df[filtered_df['source'] == 'visualizations']

if jump_filter:
    filtered_df = filtered_df[filtered_df['jump_id'].isin(jump_filter)]

if athlete_filter:
    filtered_df = filtered_df[filtered_df['skier'].isin(athlete_filter)]

if phase_filter != "All":
    filtered_df = filtered_df[filtered_df['phase'] == phase_filter]

st.markdown(f"**Results:** {len(filtered_df)} images found")

if filtered_df.empty:
    st.info("No images match the selected filters.")
else:
    cols = st.columns(4)
    for idx, row in filtered_df.iterrows():
        col = cols[idx % 4]
        with col:
            img_path = row['image_path'] 
            metric_label = row['metric_clean']
            caption_text = f"{row['jump_id']} - {row['phase']} - {row['skier']}"
            if metric_label != "Raw Frame":
                caption_text += f" "
            if img_path and os.path.exists(img_path):
                try:
                    st.image(img_path, width='stretch')
                    st.caption(caption_text)
                except Exception as e:
                    st.warning(f"Could not display image: {e}")
            else:
                st.warning("File missing")