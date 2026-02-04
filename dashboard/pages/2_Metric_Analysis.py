import streamlit as st
import pandas as pd
import numpy as np # For fake data simulation
from PIL import Image
import os
import sys

# Add parent directory to path to import utils
sys.path.append("..")
from utils import load_data_from_folders

st.set_page_config(page_title="Metric Analysis", page_icon="📊", layout="wide")

st.title("📊 Detailed Metric Analysis")

# --- LOAD DATA ---
# Load up to the maximum allowed images for analysis
raw_data = load_data_from_folders(limit=50000)
df = pd.DataFrame(raw_data)

# Filter out raw frames, we only want annotated metrics here
df_metrics = df[df['has_skeleton'] == True].copy()

if df_metrics.empty:
    st.warning("No annotated metric images found. Check dataset.")
    st.stop()

# --- MODE SELECTION ---
mode = st.radio("Select Analysis Mode:", ["By Jump", "By Metric"], horizontal=True)
st.divider()

# ==========================
# LOGIC A: BY JUMP
# ==========================
if mode == "By Jump":
    col_sel, col_view = st.columns([1, 2])

    # Ensure metrics CSV and mapping are available (loaded earlier in the file for BY METRIC branch)
    metrics_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "metrics", "metrics_summary_per_jump.csv"))
    try:
        metrics_df = pd.read_csv(metrics_csv)
    except Exception:
        metrics_df = pd.DataFrame()

    metric_display_to_col = {
        "V-Style Angle (front)": "avg_v_style_front",
        "V-Style Angle (back)": "avg_v_style_back",
        "Take Off Knee Angle": "takeoff_knee_angle",
        "Body Sky Angle": "avg_body_ski_angle",
        "Symmetry Index": "avg_symmetry_index_back",
        "Telemark Offset": "avg_telemark_offset_x",
    }

    with col_sel:
        # 1. Select Jump (prefer jumps present in metrics CSV if available, else use annotated jumps)
        if not metrics_df.empty:
            all_jumps = metrics_df['jump_id'].dropna().tolist()
            selected_jump = st.selectbox("1. Select Jump ID", all_jumps)

            # Determine available metrics for this jump (only those non-null in CSV for the jump)
            row = metrics_df[metrics_df['jump_id'] == selected_jump]
            available_metrics = [k for k, col in metric_display_to_col.items() if col in metrics_df.columns and not row.empty and pd.notna(row.iloc[0][col])]

            # Fallback: if CSV has no metrics for this jump, fallback to annotated metrics
            if not available_metrics:
                jump_subset = df_metrics[df_metrics['jump_id'] == selected_jump]
                available_metrics = sorted(jump_subset['metric_clean'].unique())
        else:
            # Fallback: use annotated metrics per jump
            all_jumps = sorted(df_metrics['jump_id'].unique())
            selected_jump = st.selectbox("1. Select Jump ID", all_jumps)
            jump_subset = df_metrics[df_metrics['jump_id'] == selected_jump]
            available_metrics = sorted(jump_subset['metric_clean'].unique())

        if not available_metrics:
            st.warning("No metrics found for this jump.")
            st.stop()

        # 2. Select Metric for the chosen jump
        selected_metric = st.selectbox("2. Select Metric", available_metrics)

    with col_view:
        st.subheader(f"{selected_jump} - {selected_metric}")

        # If CSV has the metric, show the numeric value
        if not metrics_df.empty and selected_metric in metric_display_to_col:
            col_name = metric_display_to_col[selected_metric]
            val_row = metrics_df[metrics_df['jump_id'] == selected_jump]
            if not val_row.empty and pd.notna(val_row.iloc[0].get(col_name)):
                val = val_row.iloc[0][col_name]
                st.metric(label=f"{selected_metric} ({selected_jump})", value=round(val, 2) if isinstance(val, (int, float)) else val)
            else:
                st.info("Metric value not available in CSV for this jump.")

        # Show an example annotated image for the jump (no phase/frame selection)
        examples = df_metrics[df_metrics['jump_id'] == selected_jump]
        if not examples.empty:
            example_row = examples.iloc[0]
            img_path = example_row['image_path']
            if os.path.exists(img_path):
                try:
                    st.image(img_path, caption=f"Jump: {selected_jump}", width='stretch')
                except Exception as e:
                    st.error(f"Could not display image: {e}")
            else:
                st.error("Example image file not found for this jump.")
        else:
            st.info("No annotated example images found for this jump.")

# ==========================
# LOGIC B: BY METRIC
# ==========================
elif mode == "By Metric":
    col_sel, col_view = st.columns([1, 2])

    # Metrics CSV (project-level metrics summary)
    metrics_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "metrics", "metrics_summary_per_jump.csv"))
    try:
        metrics_df = pd.read_csv(metrics_csv)
    except Exception:
        metrics_df = pd.DataFrame()

    # Map user-facing metric names to CSV columns
    metric_display_to_col = {
        "V-Style Angle (front)": "avg_v_style_front",
        "V-Style Angle (back)": "avg_v_style_back",
        "Take Off Knee Angle": "takeoff_knee_angle",
        "Body Sky Angle": "avg_body_ski_angle",
        "Symmetry Index": "avg_symmetry_index_back",
        "Telemark Offset": "avg_telemark_offset_x",
    }

    with col_sel:
        # 1. Select Metric and Jump with dynamic filtering
        if not metrics_df.empty:
            all_metric_displays = list(metric_display_to_col.keys())
            all_jumps = metrics_df['jump_id'].dropna().tolist()

            # First, ask for jump (can be 'All' meaning any)
            selected_jump = st.selectbox("1. Select Jump ID (or 'All')", ['All'] + all_jumps)

            # Determine available metrics for the selected jump
            if selected_jump != 'All':
                row = metrics_df[metrics_df['jump_id'] == selected_jump]
                available_metrics = [k for k, col in metric_display_to_col.items() if col in row.columns and pd.notna(row.iloc[0][col])]
            else:
                # if All jumps, metrics that have any non-null values
                available_metrics = [k for k, col in metric_display_to_col.items() if metrics_df[col].notna().any()]

            if not available_metrics:
                st.warning("No metrics available for this selection.")
                st.stop()

            # Ask for metric (filtered by selected_jump)
            selected_metric_display = st.selectbox("2. Select Metric Type", available_metrics)
            selected_col = metric_display_to_col[selected_metric_display]

            # Now refine available jumps for the selected metric
            jumps_with_metric = metrics_df[metrics_df[selected_col].notna()]['jump_id'].tolist()
            # If user previously chose a specific jump that doesn't have this metric, switch to first available
            if selected_jump != 'All' and selected_jump not in jumps_with_metric:
                selected_jump = jumps_with_metric[0]
        else:
            # Fallback to previous behavior when metrics summary is missing
            st.warning("Metrics summary CSV not found - falling back to image-based metric selection.")
            unique_metrics = sorted(df_metrics['metric_clean'].unique())

            # Allow selecting jump first or metric first
            all_jumps = sorted(df_metrics['jump_id'].unique().tolist())
            selected_jump = st.selectbox("1. Select Jump ID (or 'All')", ['All'] + all_jumps)

            if selected_jump != 'All':
                # metrics available for that jump
                jump_subset = df_metrics[df_metrics['jump_id'] == selected_jump]
                available_metrics = sorted(jump_subset['metric_clean'].unique())
            else:
                available_metrics = unique_metrics

            if not available_metrics:
                st.warning("No metrics found for this jump.")
                st.stop()

            selected_metric = st.selectbox("2. Select Metric Type", available_metrics)
            selected_metric_display = selected_metric
            selected_col = None

    with col_view:
        if not metrics_df.empty and selected_col is not None:
            st.subheader(f"Metric: {selected_metric_display}")

            # Show the metric table and simple chart
            metric_table = metrics_df[["jump_id", selected_col]].dropna().set_index('jump_id')
            st.dataframe(metric_table.sort_values(by=selected_col, ascending=False))

            try:
                st.bar_chart(metric_table[selected_col])
            except Exception:
                pass

            # Show value for selected jump
            val = metrics_df.loc[metrics_df['jump_id'] == selected_jump, selected_col].iloc[0]
            st.metric(label=f"{selected_metric_display} ({selected_jump})", value=round(val, 2))

            # Try to show an annotated example image for that jump
            row_subset = df_metrics[df_metrics['jump_id'] == selected_jump]
            if not row_subset.empty:
                final_row = row_subset.iloc[0]
                img_path = final_row['image_path']
                if os.path.exists(img_path):
                    try:
                        st.image(img_path, caption=f"Jump: {selected_jump} | Phase: {final_row['phase']}", width='stretch')
                    except Exception as e:
                        st.error(f"Could not display image: {e}")
                else:
                    st.error("Image file not found for this jump.")
            else:
                st.info("No annotated example image found for this jump in dataset.")
        else:
            # Fallback display when metrics CSV missing
            if selected_col is None:
                st.subheader(f"Comparison: {selected_metric}")
                row_subset = df_metrics[(df_metrics['jump_id'] == selected_jump) & (df_metrics['metric_clean'] == selected_metric)]
                if not row_subset.empty:
                    final_row = row_subset.iloc[0]
                    img_path = final_row['image_path']
                    if os.path.exists(img_path):
                        try:
                            st.image(img_path, caption=f"Jump: {selected_jump} | Phase: {final_row['phase']}", width='stretch')
                        except Exception as e:
                            st.error(f"Could not display image: {e}")
                        st.success("Metric verified")
                    else:
                        st.error("Image file not found.")
                else:
                    st.warning("No images found for this metric/jump.")