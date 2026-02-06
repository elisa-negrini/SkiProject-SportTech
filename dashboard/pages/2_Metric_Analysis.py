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

# Mappa tra il nome visualizzato nella selectbox e il nome della cartella su disco
# Assicurati che questi valori a destra corrispondano esattamente alle cartelle in frame_overlays
metric_display_to_folder = {
    "V-Style Angle (front)": "v_style_angle",
    "V-Style Angle (back)": "v_style_angle",
    "Take Off Knee Angle": "takeoff_knee_angle",
    "Body Sky Angle": "body_ski_angle",
    "Symmetry Index": "symmetry_index_back",
    "Telemark Offset": "telemark_depth_back_ratio", # O il nome esatto della cartella telemark
}

# Definisci il percorso base per gli overlay
# Risale di 2 livelli da 'pages' -> root -> metrics -> visualizations -> frame_overlays
viz_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "metrics", "visualizations", "frame_overlays"))

# --- LOAD DATA ---
# Load up to the maximum allowed images for analysis
raw_data = load_data_from_folders(limit=50000)
df = pd.DataFrame(raw_data)

# Filter out raw frames, we only want annotated metrics here
df_metrics = df[df['has_skeleton'] == True].copy()

if df_metrics.empty:
    st.warning("No annotated metric images found. Check dataset.")
    st.stop()

st.divider()


# --- LAYOUT PRINCIPALE (3 Colonne) ---
col_sel, col_view, col_stats = st.columns([1, 1.5, 1.5], gap="medium")

# Caricamento CSV Metrics per Frame (se non lo hai già caricato sopra)
frame_metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "metrics", "core_metrics", "metrics_per_frame.csv"))
try:
    df_frame = pd.read_csv(frame_metrics_path)
except Exception:
    df_frame = pd.DataFrame()

# Mappe di configurazione (Metrica -> Colonna CSV / Cartella)
metric_display_to_frame_col = {
    "V-Style Angle (front)": "v_style_angle_front",
    "V-Style Angle (back)": "v_style_angle_back",
    "Take Off Knee Angle": "takeoff_knee_angle",
    "Body Sky Angle": "body_ski_angle",
    "Symmetry Index": "symmetry_index_back",
    "Telemark Offset": "telemark_depth_back_ratio", 
}

# Assicurati di avere anche questa mappa definita (per le cartelle delle immagini)
metric_display_to_folder = {
    "V-Style Angle (front)": "v_style_angle",
    "V-Style Angle (back)": "v_style_angle",
    "Take Off Knee Angle": "takeoff_knee_angle",
    "Body Sky Angle": "body_ski_angle",
    "Symmetry Index": "symmetry_index_back",
    "Telemark Offset": "telemark_depth_back_ratio", 
}

# --- COLONNA 1: FILTRI E VALORE PUNTUALE ---
with col_sel:
    st.subheader("Filtri")
    
    # 1. PRIMA LA METRICA
    available_metrics_display = list(metric_display_to_frame_col.keys())
    selected_metric = st.selectbox("1. Seleziona Metrica", available_metrics_display)
    
    # --- LOGICA DI FILTRO SALTI ---
    valid_jumps = []
    col_csv = metric_display_to_frame_col.get(selected_metric)
    
    if not df_frame.empty and col_csv in df_frame.columns:
        # Filtra: prendi i Jump ID dove la colonna non è vuota
        valid_jumps = df_frame[df_frame[col_csv].notna()]['jump_id'].unique().tolist()
        valid_jumps.sort()
    else:
        valid_jumps = sorted(df_metrics['jump_id'].unique())

    if not valid_jumps:
        st.warning(f"Nessun salto trovato per {selected_metric}")
        st.stop()

    # 2. POI IL SALTO
    selected_jump = st.selectbox("2. Seleziona Salto", valid_jumps)
    
    st.divider()

    # --- VALORE PUNTUALE DEL FRAME ---
    st.subheader("Dettaglio Frame")
    
    # Recuperiamo l'indice dell'immagine che l'utente sta guardando nella colonna centrale
    session_key = f"idx_{selected_jump}_{selected_metric}"
    current_img_idx = st.session_state.get(session_key, 0)
    
    # Dobbiamo capire a quale frame corrisponde l'immagine n-esima
    current_frame_idx = None
    real_val = None
    
    folder_name = metric_display_to_folder.get(selected_metric)
    
    if folder_name:
        target_dir = os.path.join(viz_base_path, selected_jump, folder_name)
        if os.path.exists(target_dir):
            files = sorted([f for f in os.listdir(target_dir) if f.endswith(".jpg")])
            
            # Se l'indice è valido, prendiamo il file
            if files and current_img_idx < len(files):
                file_name = files[current_img_idx]
                try:
                    # Esempio nome file: viz_00344.jpg -> estraiamo 344
                    frame_str = file_name.split("_")[-1].split(".")[0]
                    current_frame_idx = int(frame_str)
                except:
                    pass
    
    # Se abbiamo il numero del frame, cerchiamo il valore nel CSV
    if current_frame_idx is not None and not df_frame.empty:
        if col_csv and col_csv in df_frame.columns:
            # Query precisa: Salto + Frame
            row = df_frame[
                (df_frame['jump_id'] == selected_jump) & 
                (df_frame['frame_idx'] == current_frame_idx)
            ]
            if not row.empty:
                val = row.iloc[0][col_csv]
                if pd.notna(val):
                    real_val = val

    # Visualizzazione
    if real_val is not None:
        st.metric(label=f"Valore al Frame {current_frame_idx}", value=f"{real_val:.2f}")
    else:
        label_txt = f"Frame {current_frame_idx}" if current_frame_idx else "Frame"
        st.metric(label=label_txt, value="--")
        st.caption("Dato non disponibile nel CSV per questo frame preciso.")

# --- COLONNA 2: IMMAGINE E NAVIGAZIONE ---
with col_view:
    st.write("#### Visualizzazione")
    folder_name = metric_display_to_folder.get(selected_metric)
    
    if folder_name:
        target_dir = os.path.join(viz_base_path, selected_jump, folder_name)
        if os.path.exists(target_dir):
            files = sorted([f for f in os.listdir(target_dir) if f.endswith(".jpg")])
            
            if files:
                # Inizializza sessione se necessario
                if session_key not in st.session_state: st.session_state[session_key] = 0
                
                # Pulsanti freccia
                c_prev, c_txt, c_next = st.columns([1, 2, 1])
                with c_prev:
                    if st.button("⬅️", key=f"prev_{session_key}"):
                        st.session_state[session_key] = max(0, st.session_state[session_key] - 1)
                        st.rerun()
                with c_next:
                    if st.button("➡️", key=f"next_{session_key}"):
                        st.session_state[session_key] = min(len(files)-1, st.session_state[session_key] + 1)
                        st.rerun()
                
                # Info testo
                curr_idx = st.session_state[session_key]
                with c_txt:
                    st.markdown(f"<div style='text-align: center; margin-top: 5px;'><b>Img {curr_idx+1}/{len(files)}</b></div>", unsafe_allow_html=True)

                # Mostra immagine
                file_path = os.path.join(target_dir, files[curr_idx])
                st.image(file_path, width=430)
            else:
                st.info("Nessuna immagine trovata.")
        else:
            st.info("Cartella non trovata.")

# --- COLONNA 3: CONFRONTO CON TUTTI I SALTI (INTERATTIVO) ---
with col_stats:
    st.write("#### Confronto Globale")
    
    col_name = metric_display_to_frame_col.get(selected_metric)
    
    if not df_frame.empty and col_name in df_frame.columns:
        # 1. Aggreghiamo i dati: Calcola la media della metrica per OGNI salto
        # Raggruppa per jump_id e fai la media, ignorando i NaN
        df_agg = df_frame.groupby('jump_id')[col_name].mean().reset_index()
        
        # Rinomina la colonna per chiarezza nel grafico
        df_agg.rename(columns={col_name: 'Valore Medio'}, inplace=True)
        
        # 2. Aggiungi una colonna per colorare diversamente il salto selezionato
        df_agg['Stato'] = df_agg['jump_id'].apply(
            lambda x: 'Salto Attuale' if x == selected_jump else 'Altri Salti'
        )
        
        if not df_agg.empty:
            import altair as alt
            
            # 3. Creazione Grafico Interattivo con Altair
            # Ordiniamo i salti dal valore più alto al più basso
            chart = alt.Chart(df_agg).mark_bar().encode(
                x=alt.X('jump_id:N', sort='-y', title='Salti (ordinati per valore)'), # N = Nominale
                y=alt.Y('Valore Medio:Q', title=f'Media {selected_metric}'),         # Q = Quantitativo
                color=alt.Color('Stato:N', 
                                scale=alt.Scale(domain=['Salto Attuale', 'Altri Salti'], 
                                                range=['#ff2b2b', '#d3d3d3']), # Rosso acceso vs Grigio spento
                                legend=None),
                tooltip=['jump_id', alt.Tooltip('Valore Medio', format='.2f')]
            ).properties(
                title=f"Posizionamento {selected_jump} rispetto agli altri",
                height=300
            ).interactive() # Rende il grafico zoomabile e navigabile
            
            st.altair_chart(chart, use_container_width=True)
            
            # 4. Statistiche testuali rapide
            curr_avg = df_agg[df_agg['jump_id'] == selected_jump]['Valore Medio'].iloc[0] if not df_agg[df_agg['jump_id'] == selected_jump].empty else 0
            global_avg = df_agg['Valore Medio'].mean()
            diff = curr_avg - global_avg
            
            st.caption(f"Media di {selected_jump}: **{curr_avg:.2f}**")
            st.caption(f"Media Globale (tutti i salti): **{global_avg:.2f}**")
            
            if diff > 0:
                st.success(f"Questo salto è sopra la media di +{diff:.2f}")
            else:
                st.error(f"Questo salto è sotto la media di {diff:.2f}")

        else:
            st.warning("Dati insufficienti per il confronto.")
    else:
        st.info("Dati metrici non disponibili nel CSV.")
# ==========================
# LOGIC B: BY METRIC
# ==========================
# elif mode == "By Metric":
#     col_sel, col_view = st.columns([1, 2], gap="large")

#     # Metrics CSV (project-level metrics summary)
#     metrics_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "metrics", "metrics_summary_per_jump.csv"))
#     try:
#         metrics_df = pd.read_csv(metrics_csv)
#     except Exception:
#         metrics_df = pd.DataFrame()

#     # Map user-facing metric names to CSV columns
#     metric_display_to_col = {
#         "V-Style Angle (front)": "avg_v_style_front",
#         "V-Style Angle (back)": "avg_v_style_back",
#         "Take Off Knee Angle": "takeoff_knee_angle",
#         "Body Sky Angle": "avg_body_ski_angle",
#         "Symmetry Index": "avg_symmetry_index_back",
#         "Telemark Offset": "avg_telemark_offset_x",
#     }

#     with col_sel:
#         # 1. Select Metric and Jump with dynamic filtering
#         if not metrics_df.empty:
#             all_metric_displays = list(metric_display_to_col.keys())
#             all_jumps = metrics_df['jump_id'].dropna().tolist()

#             # First, ask for jump (can be 'All' meaning any)
#             selected_jump = st.selectbox("1. Select Jump ID (or 'All')", ['All'] + all_jumps)

#             # Determine available metrics for the selected jump
#             if selected_jump != 'All':
#                 row = metrics_df[metrics_df['jump_id'] == selected_jump]
#                 available_metrics = [k for k, col in metric_display_to_col.items() if col in row.columns and pd.notna(row.iloc[0][col])]
#             else:
#                 # if All jumps, metrics that have any non-null values
#                 available_metrics = [k for k, col in metric_display_to_col.items() if metrics_df[col].notna().any()]

#             if not available_metrics:
#                 st.warning("No metrics available for this selection.")
#                 st.stop()

#             # Ask for metric (filtered by selected_jump)
#             selected_metric_display = st.selectbox("2. Select Metric Type", available_metrics)
#             selected_col = metric_display_to_col[selected_metric_display]

#             # Now refine available jumps for the selected metric
#             jumps_with_metric = metrics_df[metrics_df[selected_col].notna()]['jump_id'].tolist()
#             # If user previously chose a specific jump that doesn't have this metric, switch to first available
#             if selected_jump != 'All' and selected_jump not in jumps_with_metric:
#                 selected_jump = jumps_with_metric[0]
#         else:
#             # Fallback to previous behavior when metrics summary is missing
#             st.warning("Metrics summary CSV not found - falling back to image-based metric selection.")
#             unique_metrics = sorted(df_metrics['metric_clean'].unique())

#             # Allow selecting jump first or metric first
#             all_jumps = sorted(df_metrics['jump_id'].unique().tolist())
#             selected_jump = st.selectbox("1. Select Jump ID (or 'All')", ['All'] + all_jumps)

#             if selected_jump != 'All':
#                 # metrics available for that jump
#                 jump_subset = df_metrics[df_metrics['jump_id'] == selected_jump]
#                 available_metrics = sorted(jump_subset['metric_clean'].unique())
#             else:
#                 available_metrics = unique_metrics

#             if not available_metrics:
#                 st.warning("No metrics found for this jump.")
#                 st.stop()

#             selected_metric = st.selectbox("2. Select Metric Type", available_metrics)
#             selected_metric_display = selected_metric
#             selected_col = None

#     with col_view:
#         if not metrics_df.empty and selected_col is not None:
#             st.subheader(f"Metric: {selected_metric_display}")

#             # Show the metric table and simple chart
#             metric_table = metrics_df[["jump_id", selected_col]].dropna().set_index('jump_id')
#             st.dataframe(metric_table.sort_values(by=selected_col, ascending=False))

#             try:
#                 st.bar_chart(metric_table[selected_col])
#             except Exception:
#                 pass

#             # Show value for selected jump
#             val = metrics_df.loc[metrics_df['jump_id'] == selected_jump, selected_col].iloc[0]
#             st.metric(label=f"{selected_metric_display} ({selected_jump})", value=round(val, 2))

#             # --- MODIFICA: Caricamento Immagine da frame_overlays ---
#             st.write("---")
            
#             # Recupera il nome della cartella
#             folder_name = metric_display_to_folder.get(selected_metric_display)
            
#             if folder_name:
#                 target_dir = os.path.join(viz_base_path, selected_jump, folder_name)
                
#                 found_image = None
#                 if os.path.exists(target_dir):
#                     files = sorted([f for f in os.listdir(target_dir) if f.endswith(".jpg")])
#                     if files:
#                         found_image = os.path.join(target_dir, files[0])
                
#                 if found_image:
#                     st.image(found_image, caption=f"Visualization: {selected_jump} | {selected_metric_display}", width=None, use_container_width=True)
#                 else:
#                     st.info(f"Visualizzazione non trovata per {selected_jump} (Path: {target_dir})")
#             else:
#                 st.warning("Mapping della cartella non definito per questa metrica.")
#         else:
#             # Fallback display when metrics CSV missing
#             if selected_col is None:
#                 st.subheader(f"Comparison: {selected_metric}")
#                 row_subset = df_metrics[(df_metrics['jump_id'] == selected_jump) & (df_metrics['metric_clean'] == selected_metric)]
#                 if not row_subset.empty:
#                     final_row = row_subset.iloc[0]
#                     img_path = final_row['image_path']
#                     if os.path.exists(img_path):
#                         try:
#                             st.image(img_path, caption=f"Jump: {selected_jump} | Phase: {final_row['phase']}", width='stretch')
#                         except Exception as e:
#                             st.error(f"Could not display image: {e}")
#                         st.success("Metric verified")
#                     else:
#                         st.error("Image file not found.")
#                 else:
#                     st.warning("No images found for this metric/jump.")