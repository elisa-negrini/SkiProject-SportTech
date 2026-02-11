import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys

sys.path.append("..")

st.set_page_config(page_title="Metric Analysis", page_icon="📊", layout="wide")

st.title("Detailed Metric Analysis")

viz_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "metrics", "metrics_visualizations", "frame_overlays"))
frame_metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "metrics", "core_metrics", "metrics_per_frame.csv"))
summary_metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "metrics", "timeseries_metrics", "timeseries_summary.csv"))

df_frame = pd.DataFrame()
if os.path.exists(frame_metrics_path):
    df_frame = pd.read_csv(frame_metrics_path)
else:
    st.error(f"❌ Frame metrics file not found: {frame_metrics_path}")

df_summary = pd.DataFrame()
if os.path.exists(summary_metrics_path):
    df_summary = pd.read_csv(summary_metrics_path)
    if 'jump_id' in df_summary.columns:
        def normalize_jid_simple(val):
            import re
            s = str(val)
            m = re.search(r'(\d+)', s)
            return f"JP{int(m.group(1)):04d}" if m else str(val)
        df_summary['jump_id'] = df_summary['jump_id'].apply(normalize_jid_simple)

metric_display_to_col = {
    "V-Style Angle (front)": "v_style_angle_front",
    "V-Style Angle (back)": "v_style_angle_back",
    "Take Off Knee Angle": "takeoff_knee_angle",
    "Body Ski Angle": "body_ski_angle",
    "Symmetry Index": "symmetry_index_back", 
    "Telemark Scissor": "telemark_scissor_ratio"
}

metric_display_to_folder = {
    "V-Style Angle (front)": "v_style_angle",
    "V-Style Angle (back)": "v_style_angle",
    "Take Off Knee Angle": "takeoff_knee_angle",
    "Body Ski Angle": "body_ski_angle",
    "Symmetry Index": "symmetry_index_back",
    "Telemark Scissor": "telemark_scissor"
}

st.divider()

col_sel, col_view, col_stats = st.columns([1, 1.5, 1.5], gap="medium")
with col_sel:
    st.subheader("Filters")
    available_metrics_display = list(metric_display_to_col.keys())
    selected_metric = st.selectbox("1. Select Metric", available_metrics_display)
    
    col_name = metric_display_to_col.get(selected_metric)
    valid_jumps = []

    if selected_metric == "Landing Knee Compression":
        # Cerca nel Summary CSV
        if not df_summary.empty and col_name in df_summary.columns:
            valid_jumps = df_summary[df_summary[col_name].notna()]['jump_id'].unique().tolist()
    else:
        if not df_frame.empty and col_name in df_frame.columns:
            valid_jumps = df_frame[df_frame[col_name].notna()]['jump_id'].unique().tolist()

    valid_jumps.sort()

    if not valid_jumps:
        st.warning(f"No jumps found with data for: {selected_metric}")
        st.stop()

    selected_jump = st.selectbox("2. Select Jump", valid_jumps)
    st.divider()
    st.subheader("Frame Details")
    session_key = f"idx_{selected_jump}_{selected_metric}"
    current_img_idx = st.session_state.get(session_key, 0)
    current_frame_idx = None
    folder_name = metric_display_to_folder.get(selected_metric)
    
    if folder_name:
        target_dir = os.path.join(viz_base_path, selected_jump, folder_name)
        if os.path.exists(target_dir):
            files = sorted([f for f in os.listdir(target_dir) if f.endswith(".jpg")])
            if files and current_img_idx < len(files):
                try:
                    frame_str = files[current_img_idx].split("_")[-1].split(".")[0]
                    current_frame_idx = int(frame_str)
                except: pass

    real_val = None
    
    if selected_metric == "Landing Knee Compression":
        if not df_summary.empty and col_name in df_summary.columns:
            row = df_summary[df_summary['jump_id'] == selected_jump]
            if not row.empty:
                val = row.iloc[0][col_name]
                if pd.notna(val): real_val = val
    else:
        if current_frame_idx is not None and not df_frame.empty and col_name in df_frame.columns:
            row = df_frame[
                (df_frame['jump_id'] == selected_jump) & 
                (df_frame['frame_idx'] == current_frame_idx)
            ]
            if not row.empty:
                val = row.iloc[0][col_name]
                if pd.notna(val): real_val = val

    if real_val is not None:
        lbl = "Compression (Global)" if selected_metric == "Landing Knee Compression" else f"Value at Frame {current_frame_idx}"
        
        fmt = "{:.3f}" if selected_metric == "Telemark Scissor" else "{:.2f}"
        st.metric(label=lbl, value=fmt.format(real_val))

        if selected_metric == "Telemark Scissor":
            st.caption(f"= {real_val * 100:.1f}% of leg height")

    else:
        st.metric(label=f"Frame {current_frame_idx or '?'}", value="--")
        st.caption("Numeric data not found for this frame.")

with col_view:
    st.write("#### Visualization")
    folder_name = metric_display_to_folder.get(selected_metric)
    
    if folder_name:
        target_dir = os.path.join(viz_base_path, selected_jump, folder_name)
        
        if os.path.exists(target_dir):
            files = sorted([f for f in os.listdir(target_dir) if f.endswith(".jpg")])
            
            if files:
                if session_key not in st.session_state: st.session_state[session_key] = 0
                if st.session_state[session_key] >= len(files): st.session_state[session_key] = 0

                c1, c2, c3 = st.columns([1, 2, 1])
                with c1:
                    if st.button("⬅️", key=f"p_{session_key}"):
                        st.session_state[session_key] = max(0, st.session_state[session_key] - 1)
                        st.rerun()
                with c3:
                    if st.button("➡️", key=f"n_{session_key}"):
                        st.session_state[session_key] = min(len(files)-1, st.session_state[session_key] + 1)
                        st.rerun()
                with c2:
                     st.markdown(f"<div style='text-align:center'><b>{st.session_state[session_key]+1} / {len(files)}</b></div>", unsafe_allow_html=True)

                img_path = os.path.join(target_dir, files[st.session_state[session_key]])
                st.image(img_path, width=450)
            else:
                st.info("No images generated for this jump/metric.")
                st.caption("Use 'Metrics Visualizer' (Mode 2) to generate them.")
        else:
            st.warning(f"Folder not found: {target_dir}")
            st.caption("Make sure you have run the visualizer.")

with col_stats:
    st.write("#### Global Comparison")
    
    df_agg = pd.DataFrame()
    col_name = metric_display_to_col.get(selected_metric)

    if selected_metric == "Landing Knee Compression":
        # Dal Summary
        if not df_summary.empty and col_name in df_summary.columns:
            df_agg = df_summary[['jump_id', col_name]].copy()
            df_agg.rename(columns={col_name: 'Average Value'}, inplace=True)
    else:
        if not df_frame.empty and col_name in df_frame.columns:
            df_agg = df_frame.groupby('jump_id')[col_name].mean().reset_index()
            df_agg.rename(columns={col_name: 'Average Value'}, inplace=True)

    if not df_agg.empty:
        df_agg = df_agg.dropna(subset=['Average Value'])
        df_agg['Status'] = df_agg['jump_id'].apply(lambda x: 'Current' if x == selected_jump else 'Others')
        
        import altair as alt
        chart = alt.Chart(df_agg).mark_bar().encode(
            x=alt.X('jump_id:N', sort='-y', title='Jumps'),
            y=alt.Y('Average Value:Q'),
            color=alt.Color('Status:N', scale=alt.Scale(range=['#d3d3d3', '#ff4b4b']), legend=None),
            tooltip=['jump_id', 'Average Value']
        ).properties(height=300).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        curr_val = df_agg[df_agg['jump_id'] == selected_jump]['Average Value'].iloc[0]
        avg_val = df_agg['Average Value'].mean()
        diff = curr_val - avg_val
        fmt = "{:.3f}" if selected_metric == "Telemark Scissor" else "{:.2f}"
        
        st.caption(f"Your Jump: **{fmt.format(curr_val)}** | Global Average: **{fmt.format(avg_val)}**")
        if diff > 0: st.success(f"+{fmt.format(diff)} above average")
        else: st.error(f"{fmt.format(diff)} below average")
    else:
        st.info("Insufficient data for comparison chart.")