import pandas as pd
import numpy as np
import os
from pathlib import Path

class MetricsCalculator:
    """
    Calculates biomechanical metrics from normalized keypoints:
    - Angles (V-Style, Body-Ski, Takeoff knee angle, etc.)
    - Symmetry (V-Style)
    - Stability (standard deviations during flight)
    - Telemark (offset, projection, leg angle)
    
    Output:
    - metrics/metrics_per_frame.csv: metrics per frame
    - metrics/metrics_summary_per_jump.csv: aggregated metrics per jump
    """
    
    def __init__(self):
        # --- PATHS ---
        self.keypoints_file = 'keypoints_dataset.csv'
        self.phases_file = 'jump_phases_SkiTB.csv'
        
        # Output directory
        self.metrics_dir = Path('metrics')
        self.output_detailed = self.metrics_dir / 'metrics_per_frame.csv'
        self.output_summary = self.metrics_dir / 'metrics_summary_per_jump.csv'
        
        # --- KEYPOINT MAPPING ---
        # Maps keypoint names -> ID in CSV (without 'kpt_' prefix)
        # These IDs correspond to the Roboflow annotation system
        self.kpt_map = {
            # Body
            'head': '1',
            'neck': '2',
            'center_pelvis': '9',
            'r_shoulder': '3',  'l_shoulder': '6',
            'r_hip': '17',      'l_hip': '10',
            'r_knee': '18',     'l_knee': '11',
            'r_ankle': '19',    'l_ankle': '12',
            
            # Skis
            'r_ski_tip': '23',  'r_ski_tail': '22',
            'l_ski_tip': '16',  'l_ski_tail': '15'
        }
        
        # Create output directory if it doesn't exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """
        Loads spatial data (normalized keypoints) and temporal data (jump phases).
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        if not os.path.exists(self.keypoints_file):
            print(f"❌ Error: {self.keypoints_file} not found.")
            return False
        if not os.path.exists(self.phases_file):
            print(f"❌ Error: {self.phases_file} not found.")
            return False

        self.df_kpts = pd.read_csv(self.keypoints_file)
        self.df_phases = pd.read_csv(self.phases_file)
        
        # --- JUMP ID NORMALIZATION ---
        # Converts "jump5" -> "JP0005" to match keypoints dataset
        def normalize_jid(val):
            # Extract numbers from string (e.g. "jump5" -> "5")
            digits = ''.join(filter(str.isdigit, str(val)))
            if digits:
                # Format as JPxxxx (e.g. "JP0005")
                return f"JP{int(digits):04d}"
            return str(val)

        # Apply normalization to both datasets
        self.df_phases['jump_id'] = self.df_phases['jump_id'].apply(normalize_jid)
        self.df_kpts['jump_id'] = self.df_kpts['jump_id'].apply(normalize_jid)

        # --- PARSING FRAME NUMBERS ---
        # Converts file name (e.g. "frame_00324.jpg") -> integer number (324)
        try:
            self.df_kpts['frame_idx'] = self.df_kpts['frame_name'].apply(
                lambda x: int(''.join(filter(str.isdigit, str(x))))
            )
        except Exception as e:
            print(f"⚠️ Warning: Could not parse frame numbers. Check format. {e}")
            return False

        print(f"✅ Loaded Data: {len(self.df_kpts)} frames, {len(self.df_phases)} jumps defined.")
        return True

    def get_point(self, row, name):
        """
        Extracts (x, y) coordinates of a keypoint from a dataframe row.
        
        Args:
            row: keypoints DataFrame row
            name: keypoint name (e.g. 'neck', 'r_knee')
        
        Returns:
            np.array([x, y]) if keypoint is visible, None otherwise
        """
        try:
            kid = self.kpt_map[name]
            x = row[f'kpt_{kid}_x']
            y = row[f'kpt_{kid}_y']
            v = row[f'kpt_{kid}_v']
            if v == 0: return None  # Invisible
            return np.array([x, y])
        except KeyError:
            return None

    def calculate_vector_angle(self, v1, v2):
        """
        Calculates angle in degrees between two vectors.
        
        Args:
            v1, v2: np.array 2D vectors
        
        Returns:
            float: angle in degrees [0-180], np.nan if calculation not possible
        """
        # Normalize vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return np.nan
        
        # Calculate dot product
        dot_product = np.dot(v1, v2)
        
        # Clip to avoid numerical errors beyond [-1, 1]
        cosine_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def detect_view_type(self, row):
        """
        Heuristic to detect view type: Side vs Frontal.
        
        Based on shoulder width:
        - Side view: overlapping shoulders (width ≈ 0)
        - Frontal view: distant shoulders (width > 0.04)
        
        Args:
            row: keypoints DataFrame row
        
        Returns:
            str: 'SIDE', 'FRONT' or 'UNKNOWN'
        """
        p_r_sh = self.get_point(row, 'r_shoulder')
        p_l_sh = self.get_point(row, 'l_shoulder')
        
        if p_r_sh is None or p_l_sh is None:
            return 'UNKNOWN'
        
        # Normalized shoulder width
        width = np.linalg.norm(p_r_sh - p_l_sh)
        
        # Threshold: in side view shoulders overlap (width ~ 0)
        # In frontal view width is > 0.04-0.05 (on 0-1 scale)
        if width < 0.04: 
            return 'SIDE'
        else:
            return 'FRONT'
        

    def get_best_fit_vector(self, points):
        """
        Calculates direction vector using linear regression on N points.
        points: list of np.array [x, y]
        Returns: np.array [dx, dy] normalized
        """
        if len(points) < 2: return None
        
        # Separate X and Y
        data = np.array(points)
        x = data[:, 0]
        y = data[:, 1]
        
        # Linear Regression (Fit line y = mx + q)
        # Note: if the athlete is perfectly vertical, polyfit may have issues.
        # During flight they are horizontal/diagonal, so it's fine.
        try:
            slope, intercept = np.polyfit(x, y, 1)
            
            # Construct direction vector from slope
            # If x increases by 1, y increases by 'slope'
            # Does the vector go from head to feet?
            # Check direction based on first and last point (Shoulder -> Ankle)
            vec = np.array([1, slope])
            vec = vec / np.linalg.norm(vec) # Normalize
            
            # Correct direction (must point towards feet)
            # Raw vector shoulder->ankle
            vec_ref = points[-1] - points[0] 
            if np.dot(vec, vec_ref) < 0:
                vec = -vec
                
            return vec
        except:
            return None

    def process(self):
        """
        Main loop: iterates over jumps and frames to calculate all metrics.
        
        For each jump:
        1. Reads temporal windows of phases (takeoff, flight, landing)
        2. Calculates metrics frame-by-frame
        3. Aggregates metrics per jump
        4. Saves detailed and summary CSV files
        
        Returns:
            bool: True if process successful
        """
        
        detailed_results = []
        
        # Iterate over each jump defined in phases file
        for _, phase_row in self.df_phases.iterrows():
            jump_id = phase_row['jump_id']
            print(f"   -> Calculating metrics for {jump_id}...")

            # Telemark flag (forward 'f' or lateral 'l')
            telemark_flag = phase_row.get('telemark_f_l', 'unknown') 
            
            # Get all frames for this jump
            jump_frames = self.df_kpts[self.df_kpts['jump_id'] == jump_id].copy()
            if jump_frames.empty:
                print(f"      ⚠️ No frames found for {jump_id} in dataset.")
                continue

            # --- TEMPORAL WINDOWS (SEPARATE FRONT/BACK) ---
            
            # V-Style FRONT Window
            v_front_window = None
            if pd.notna(phase_row.get('v_style_front_start')):
                v_front_window = (phase_row['v_style_front_start'], phase_row['v_style_front_end'])

            # V-Style BACK Window
            v_back_window = None
            if pd.notna(phase_row.get('v_style_back_start')):
                v_back_window = (phase_row['v_style_back_start'], phase_row['v_style_back_end'])

            # Body-Ski Angle (body-ski angle during flight)
            bsa_window = None
            if phase_row.get('body_ski_measurable') == 1.0:
                bsa_window = (phase_row['bsa_start'], phase_row['bsa_end'])
            
            # Telemark (landing)
            tele_window = None
            if phase_row.get('telemark_measurable') == 1.0:
                tele_window = (phase_row['telemark_start'], phase_row['telemark_end'])

            # --- PER-FRAME CALCULATION ---
            # For each frame of the jump, calculate applicable metrics
            for _, frame_row in jump_frames.iterrows():
                f_idx = frame_row['frame_idx']
                
                # Detect view type (heuristic)
                view_type = self.detect_view_type(frame_row)
                
                # Initialize results for this frame
                res = {
                    'jump_id': jump_id,
                    'frame_idx': f_idx,
                    'view_detected': view_type,
                    'telemark_flag': telemark_flag,
                    'takeoff_knee_angle': np.nan,
                    'v_style_angle_front': np.nan,
                    'v_style_angle_back': np.nan,
                    'body_ski_angle': np.nan,
                    'symmetry_index_back': np.nan,
                    'telemark_offset_x_raw': np.nan,
                    'telemark_proj_ski_raw': np.nan,
                    'telemark_depth_back_ratio': np.nan,
                    'telemark_leg_angle': np.nan,
                    'is_flight_phase': 0,
                    'is_landing_phase': 0
                }

                # ========================================
                # METRIC 1: V-STYLE (Split Front/Back)
                # ========================================
                is_front = v_front_window and (v_front_window[0] <= f_idx <= v_front_window[1])
                is_back = v_back_window and (v_back_window[0] <= f_idx <= v_back_window[1])
                
                if is_front or is_back:
                    res['is_flight_phase'] = 1
                    
                    p_tip_r = self.get_point(frame_row, 'r_ski_tip')
                    p_tail_r = self.get_point(frame_row, 'r_ski_tail')
                    p_tip_l = self.get_point(frame_row, 'l_ski_tip')
                    p_tail_l = self.get_point(frame_row, 'l_ski_tail')
                    
                    if all(x is not None for x in [p_tip_r, p_tail_r, p_tip_l, p_tail_l]):
                        vec_ski_r = p_tip_r - p_tail_r
                        vec_ski_l = p_tip_l - p_tail_l
                        
                        angle_val = self.calculate_vector_angle(vec_ski_r, vec_ski_l)
                        
                        # Assign to correct column based on window
                        if is_back:
                            res['v_style_angle_back'] = angle_val

                            # --- ROBUST SYMMETRY CALCULATION ---
                            # Use neck and pelvis for axis (more stable than head)
                            p_neck = self.get_point(frame_row, 'neck')
                            p_pelvis = self.get_point(frame_row, 'center_pelvis')
                            
                            if p_neck is not None and p_pelvis is not None:
                                # Vettore Asse Corporeo: Dal Bacino al Collo (verso l'alto)
                                vec_axis = p_pelvis - p_neck
                                
                                # Compute angles: right ski vs axis, left ski vs axis
                                # calculate_vector_angle returns positive angle in [0,180]
                                angle_r = self.calculate_vector_angle(vec_ski_r, vec_axis)
                                angle_l = self.calculate_vector_angle(vec_ski_l, vec_axis)
                                
                                # Optional robustness check (cross product) for ski crossing
                                # (Currently rely on absolute angle as main indicator)
                                if not np.isnan(angle_r) and not np.isnan(angle_l):
                                    res['symmetry_index_back'] = abs(angle_r - angle_l)
                        

                if bsa_window and bsa_window[0] <= f_idx <= bsa_window[1]:
                    res['is_flight_phase'] = 1
                    
                    # 1. Calculate Body Vectors (Regression)
                    # Right side
                    pts_body_r = [self.get_point(frame_row, k) for k in ['r_shoulder', 'r_hip', 'r_knee', 'r_ankle']]
                    pts_body_r = [p for p in pts_body_r if p is not None] # Filter None
                    vec_body_r = self.get_best_fit_vector(pts_body_r)
                    
                    # Left side
                    pts_body_l = [self.get_point(frame_row, k) for k in ['l_shoulder', 'l_hip', 'l_knee', 'l_ankle']]
                    pts_body_l = [p for p in pts_body_l if p is not None]
                    vec_body_l = self.get_best_fit_vector(pts_body_l)
                    
                    # 2. Calculate Ski Vectors
                    # Right
                    p_tr, p_tailr = self.get_point(frame_row, 'r_ski_tip'), self.get_point(frame_row, 'r_ski_tail')
                    vec_ski_r = (p_tr - p_tailr) if (p_tr is not None and p_tailr is not None) else None
                    
                    # Left
                    p_tl, p_taill = self.get_point(frame_row, 'l_ski_tip'), self.get_point(frame_row, 'l_ski_tail')
                    vec_ski_l = (p_tl - p_taill) if (p_tl is not None and p_taill is not None) else None
                    
                    # 3. Averages
                    # Body average
                    final_body_vec = None
                    if vec_body_r is not None and vec_body_l is not None:
                        final_body_vec = (vec_body_r + vec_body_l) / 2
                    elif vec_body_r is not None: final_body_vec = vec_body_r
                    elif vec_body_l is not None: final_body_vec = vec_body_l
                    
                    # Ski average
                    final_ski_vec = None
                    # Normalize before averaging for safety
                    if vec_ski_r is not None: vec_ski_r = vec_ski_r / np.linalg.norm(vec_ski_r)
                    if vec_ski_l is not None: vec_ski_l = vec_ski_l / np.linalg.norm(vec_ski_l)
                    
                    if vec_ski_r is not None and vec_ski_l is not None:
                        final_ski_vec = (vec_ski_r + vec_ski_l) / 2
                    elif vec_ski_r is not None: final_ski_vec = vec_ski_r
                    elif vec_ski_l is not None: final_ski_vec = vec_ski_l
                    
                    # 4. Calculate Final Angle
                    if final_body_vec is not None and final_ski_vec is not None:
                        res['body_ski_angle'] = self.calculate_vector_angle(final_body_vec, final_ski_vec)
                
                # ========================================
                # METRIC 3: TELEMARK (Landing)
                # ========================================
                # Measures leg offset at landing
                # 3 complementary approaches for different views
                if tele_window and tele_window[0] <= f_idx <= tele_window[1]:
                    res['is_landing_phase'] = 1
                    
                    # Get ankles
                    p_ank_r = self.get_point(frame_row, 'r_ankle')
                    p_ank_l = self.get_point(frame_row, 'l_ankle')
                    
                    # 1. X OFFSET (Raw Coordinates)
                    # Horizontal difference between ankles
                    # Works well in frontal view
                    if p_ank_r is not None and p_ank_l is not None:
                        res['telemark_offset_x_raw'] = abs(p_ank_r[0] - p_ank_l[0])

                    # 2. PROJECTION ON SKI VECTOR
                    # Projects ankle distance onto ski direction
                    # Great for side/diagonal views
                    p_tip_r, p_tail_r = self.get_point(frame_row, 'r_ski_tip'), self.get_point(frame_row, 'r_ski_tail')
                    p_tip_l, p_tail_l = self.get_point(frame_row, 'l_ski_tip'), self.get_point(frame_row, 'l_ski_tail')
                    
                    ski_dirs = []
                    if p_tip_r is not None and p_tail_r is not None: 
                        ski_dirs.append(p_tip_r - p_tail_r)
                    if p_tip_l is not None and p_tail_l is not None: 
                        ski_dirs.append(p_tip_l - p_tail_l)
                    
                    if ski_dirs and p_ank_r is not None and p_ank_l is not None:
                        # Calculate average ski direction
                        avg_ski_vec = np.mean(ski_dirs, axis=0)
                        norm = np.linalg.norm(avg_ski_vec)
                        if norm > 0:
                            unit_ski_vec = avg_ski_vec / norm
                            vec_ankles = p_ank_r - p_ank_l
                            # Dot product: projection of ankle distance on ski axis
                            res['telemark_proj_ski_raw'] = abs(np.dot(vec_ankles, unit_ski_vec))

                        p_neck = self.get_point(frame_row, 'neck')
                        p_pelvis = self.get_point(frame_row, 'center_pelvis')
                        
                        if p_neck is not None and p_pelvis is not None:
                            # Compute back length in px for current frame
                            back_len_px = np.linalg.norm(p_neck - p_pelvis)
                            
                            # Compute ratio (if back length > 0)
                            res['telemark_depth_back_ratio'] = res['telemark_proj_ski_raw'] / back_len_px

                    # 3. LEG OPENING ANGLE
                    # Angle between femurs (hip->knee)
                    # Invariant to zoom, works in all views
                    p_hip_r, p_knee_r = self.get_point(frame_row, 'r_hip'), self.get_point(frame_row, 'r_knee')
                    p_hip_l, p_knee_l = self.get_point(frame_row, 'l_hip'), self.get_point(frame_row, 'l_knee')
                    
                    if all(x is not None for x in [p_hip_r, p_knee_r, p_hip_l, p_knee_l]):
                        vec_femur_r = p_knee_r - p_hip_r  # Right femur
                        vec_femur_l = p_knee_l - p_hip_l  # Left femur
                        res['telemark_leg_angle'] = self.calculate_vector_angle(vec_femur_r, vec_femur_l)

                # ========================================
                # METRIC 4: TAKEOFF KNEE ANGLE
                # ========================================
                # Measures leg flexion at exact takeoff frame
                # Important for explosive power
                takeoff_frame = phase_row.get('take_off_frame') 
                
                # Calculate ONLY at exact takeoff frame
                if pd.notna(takeoff_frame) and f_idx == int(takeoff_frame):
                    
                    # Calculate angle for both legs and take average
                    angles = []
                    
                    # RIGHT LEG: Hip -> Knee -> Ankle
                    p_hip_r = self.get_point(frame_row, 'r_hip')
                    p_knee_r = self.get_point(frame_row, 'r_knee')
                    p_ank_r = self.get_point(frame_row, 'r_ankle')
                    
                    if all(x is not None for x in [p_hip_r, p_knee_r, p_ank_r]):
                        # Vectors from knee up and down
                        vec_femur = p_hip_r - p_knee_r   # Knee->Hip (femur)
                        vec_tibia = p_ank_r - p_knee_r   # Knee->Ankle (tibia)
                        angles.append(self.calculate_vector_angle(vec_femur, vec_tibia))

                    # LEFT LEG
                    p_hip_l = self.get_point(frame_row, 'l_hip')
                    p_knee_l = self.get_point(frame_row, 'l_knee')
                    p_ank_l = self.get_point(frame_row, 'l_ankle')
                    
                    if all(x is not None for x in [p_hip_l, p_knee_l, p_ank_l]):
                        vec_femur = p_hip_l - p_knee_l
                        vec_tibia = p_ank_l - p_knee_l
                        angles.append(self.calculate_vector_angle(vec_femur, vec_tibia))
                    
                    # Average of angles (typically ~140-170°)
                    if angles:
                        res['takeoff_knee_angle'] = np.mean(angles)

                # --- FRAME SAVING ---
                # Save only if we calculated at least one metric
                metrics_found = [
                    res['v_style_angle_front'], 
                    res['v_style_angle_back'],
                    res['body_ski_angle'], 
                    res.get('telemark_offset_x_raw'),
                    res.get('telemark_proj_ski_raw'),
                    res.get('telemark_depth_back_ratio'),
                    res.get('takeoff_knee_angle'),
                    res.get('symmetry_index_back')
                ]
                
                # Save only if there's at least one valid metric
                if any(pd.notna(x) for x in metrics_found):
                    detailed_results.append(res)
        
        # --- DETAILED CSV SAVING ---
        df_detailed = pd.DataFrame(detailed_results)
        df_detailed.to_csv(self.output_detailed, index=False)
        print(f"\n✅ Detailed metrics saved: {self.output_detailed}")
        
        # --- AGGREGATION PER JUMP ---
        self.aggregate_metrics(df_detailed)
        
        return True

    def aggregate_metrics(self, df_det):
        if df_det.empty:
            print("⚠️ No metrics calculated to aggregate.")
            return

        summary_rows = []
        
        for jump_id, group in df_det.groupby('jump_id'):
            row = {'jump_id': jump_id}
            
            # 1. Separate V-Style averages
            row['avg_v_style_front'] = group['v_style_angle_front'].mean()
            row['avg_v_style_back'] = group['v_style_angle_back'].mean()
            
            # Other averages
            row['takeoff_knee_angle'] = group['takeoff_knee_angle'].mean()
            row['avg_body_ski_angle'] = group['body_ski_angle'].mean()
            row['avg_symmetry_index_back'] = group['symmetry_index_back'].mean()
            
            # Telemark
            row['avg_telemark_offset_x'] = group['telemark_offset_x_raw'].mean()
            row['avg_telemark_proj_ski'] = group['telemark_proj_ski_raw'].mean()
            row['avg_telemark_depth_ratio'] = group['telemark_depth_back_ratio'].mean()
            row['avg_telemark_leg_angle'] = group['telemark_leg_angle'].mean()
            
            # Stability (uses whichever is found: front or back)
            # Combine the two columns in a temporary series to calculate total standard deviation
            v_combined = group['v_style_angle_front'].fillna(group['v_style_angle_back'])
            std_v = v_combined.std()
            std_bsa = group['body_ski_angle'].std()
            row['flight_stability_std'] = np.nanmean([std_v, std_bsa])
            
            summary_rows.append(row)
            
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(self.output_summary, index=False)
        print(f"✅ Summary metrics saved: {self.output_summary}")
