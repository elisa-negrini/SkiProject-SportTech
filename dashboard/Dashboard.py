import streamlit as st
import os

st.set_page_config(
    page_title="Ski-TB Analysis Dashboard",
    page_icon="⛷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

col_text, col_img = st.columns([4, 1], gap="medium", vertical_alignment="bottom")

with col_text:
    st.title("Ski Jump Performance Analysis Dashboard")
    st.markdown("""
    This dashboard presents our custom **23-keypoint skeleton annotations** applied to the **[Ski-TB Dataset](https://cvlab.epfl.ch/research/datasets/ski-tb/)**.

    **Dashboard Structure:**
    
    * **Gallery Explorer** Browse and filter the complete dataset, viewing both raw frames and skeleton visualizations across different jump phases.
      
    * **Metric Analysis** Inspect specific quantitative data and qualitative metrics computed on the annotated dataset.

    """)

with col_img:
    skeleton_path = os.path.join(CURRENT_DIR, "skeleton.png")
    if os.path.exists(skeleton_path):
        st.image(
            skeleton_path, 
            caption="23-Keypoint Skeleton Model",
            width='stretch'
        )
    else:
        st.warning(f"Skeleton image not found at: {skeleton_path}")

st.divider()
st.info("👈 Select a page from the sidebar to start exploring.")