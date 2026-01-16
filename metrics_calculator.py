import pandas as pd
import numpy as np
import os
import glob

class MetricsCalculator:
    def __init__(self):
        # --- CONFIGURATION ---
        self.keypoints_file = 'keypoints_dataset.csv'
        self.phases_file = 'jump_phases_SkiTB.csv'
        self.output_detailed = 'metrics_per_frame.csv'
        self.output_summary = 'metrics_summary_per_jump.csv'
        
        # Keypoint Mapping (User Defined)
        # Format: 'name': 'id_in_csv' (without 'kpt_' prefix)
        self.kpt_map = {
            # Body
            'neck': '2',
            'r_shoulder': '3',  'l_shoulder': '6',
            'r_hip': '17',      'l_hip': '10',
            'r_knee': '18',     'l_knee': '11',
            'r_ankle': '19',    'l_ankle': '12',
            
            # Skis
            'r_ski_tip': '23',  'r_ski_tail': '22',
            'l_ski_tip': '16',  'l_ski_tail': '15'
        }

    def load_data(self):
        """Loads the spatial dataset and the temporal phases."""
        if not os.path.exists(self.keypoints_file):
            print(f"❌ Error: {self.keypoints_file} not found.")
            return False
        if not os.path.exists(self.phases_file):
            print(f"❌ Error: {self.phases_file} not found.")
            return False

        self.df_kpts = pd.read_csv(self.keypoints_file)
        self.df_phases = pd.read_csv(self.phases_file)
        
        # Convert "jump5" -> "JP0005" to match the keypoints dataset
        def normalize_jid(val):
            # Extract numbers from string (e.g. "jump5" -> "5")
            digits = ''.join(filter(str.isdigit, str(val)))
            if digits:
                # Format as JPxxxx (e.g. "JP0005")
                return f"JP{int(digits):04d}"
            return str(val)

        # Apply to phases file
        self.df_phases['jump_id'] = self.df_phases['jump_id'].apply(normalize_jid)
        
        # Apply to keypoints file (just in case)
        self.df_kpts['jump_id'] = self.df_kpts['jump_id'].apply(normalize_jid)
        # ---------------------------------

        # Ensure we can match frames (string filename -> int number)
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
        """Extracts (x, y) for a keypoint from a dataframe row."""
        try:
            kid = self.kpt_map[name]
            x = row[f'kpt_{kid}_x']
            y = row[f'kpt_{kid}_y']
            v = row[f'kpt_{kid}_v']
            if v == 0: return None # Invisible
            return np.array([x, y])
        except KeyError:
            return None

    def calculate_vector_angle(self, v1, v2):
        """Calculates angle in degrees between two vectors."""
        # Normalize vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return np.nan
        
        dot_product = np.dot(v1, v2)
        # Clip to avoid numerical errors going slightly beyond [-1, 1]
        cosine_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def detect_view_type(self, row):
        """Heuristic to detect view: Side vs Frontal based on shoulder width."""
        p_r_sh = self.get_point(row, 'r_shoulder')
        p_l_sh = self.get_point(row, 'l_shoulder')
        
        if p_r_sh is None or p_l_sh is None:
            return 'UNKNOWN'
        
        # Normalized width
        width = np.linalg.norm(p_r_sh - p_l_sh)
        
        # Threshold: In side view shoulders overlap (width ~ 0)
        # In front view width is usually > 0.05 or 0.1 (in 0-1 scale)
        if width < 0.04: 
            return 'SIDE'
        else:
            return 'FRONT'

    def run_calculations(self):
        """Main loop: Iterates over jumps and frames to calculate metrics."""
        
        detailed_results = []
        
        # Iterate over each jump in the phases file
        for _, phase_row in self.df_phases.iterrows():
            jump_id = phase_row['jump_id']
            print(f"Processing {jump_id}...")
            
            # Get all frames for this jump
            jump_frames = self.df_kpts[self.df_kpts['jump_id'] == jump_id].copy()
            if jump_frames.empty:
                print(f"   ⚠️ No frames found for {jump_id} in dataset.")
                continue

            # --- PREPARE TIME WINDOWS ---
            # V-Style (Front + Back)
            v_windows = []
            if phase_row.get('v_style_measurable') == 1.0:
                if pd.notna(phase_row.get('v_style_front_start')):
                    v_windows.append((phase_row['v_style_front_start'], phase_row['v_style_front_end']))
                if pd.notna(phase_row.get('v_style_back_start')):
                    v_windows.append((phase_row['v_style_back_start'], phase_row['v_style_back_end']))
            
            # Body-Ski Angle
            bsa_window = None
            if phase_row.get('body_ski_measurable') == 1.0:
                bsa_window = (phase_row['bsa_start'], phase_row['bsa_end'])
            
            # Telemark (Landing)
            tele_window = None
            if phase_row.get('telemark_measurable') == 1.0:
                tele_window = (phase_row['telemark_start'], phase_row['telemark_end'])

            # --- PER-FRAME CALCULATION ---
            for _, frame_row in jump_frames.iterrows():
                f_idx = frame_row['frame_idx']
                
                # Heuristic View Detection
                view_type = self.detect_view_type(frame_row)
                
                # Initialize row result
                res = {
                    'jump_id': jump_id,
                    'frame_idx': f_idx,
                    'view_detected': view_type,
                    'v_style_angle': np.nan,
                    'body_ski_angle': np.nan,
                    'symmetry_index': np.nan,
                    'telemark_offset_x': np.nan,
                    'is_flight_phase': 0,
                    'is_landing_phase': 0
                }

                # 1. METRIC: V-Style & Symmetry (Frontal)
                in_v_window = any(s <= f_idx <= e for s, e in v_windows)
                if in_v_window:
                    res['is_flight_phase'] = 1
                    
                    # Points
                    p_tip_r = self.get_point(frame_row, 'r_ski_tip')
                    p_tail_r = self.get_point(frame_row, 'r_ski_tail')
                    p_tip_l = self.get_point(frame_row, 'l_ski_tip')
                    p_tail_l = self.get_point(frame_row, 'l_ski_tail')
                    
                    p_neck = self.get_point(frame_row, 'neck')
                    p_r_hip = self.get_point(frame_row, 'r_hip')
                    p_l_hip = self.get_point(frame_row, 'l_hip')

                    # Calculate Vectors
                    if all(x is not None for x in [p_tip_r, p_tail_r, p_tip_l, p_tail_l]):
                        vec_ski_r = p_tip_r - p_tail_r
                        vec_ski_l = p_tip_l - p_tail_l
                        
                        # V-Style: Angle between skis
                        res['v_style_angle'] = self.calculate_vector_angle(vec_ski_r, vec_ski_l)

                        # Symmetry: Angle relative to Body Axis (Neck -> MidHip)
                        if p_neck is not None and p_r_hip is not None and p_l_hip is not None:
                            mid_hip = (p_r_hip + p_l_hip) / 2
                            vec_body = mid_hip - p_neck # Downward vector
                            
                            angle_r = self.calculate_vector_angle(vec_ski_r, vec_body)
                            angle_l = self.calculate_vector_angle(vec_ski_l, vec_body)
                            
                            res['symmetry_index'] = abs(angle_r - angle_l)

                # 2. METRIC: Body-Ski Angle (Lateral)
                if bsa_window and bsa_window[0] <= f_idx <= bsa_window[1]:
                    res['is_flight_phase'] = 1
                    
                    # We calculate for both sides and take the valid one or average
                    angles = []
                    
                    # SIDE R
                    p_sh_r = self.get_point(frame_row, 'r_shoulder')
                    p_ank_r = self.get_point(frame_row, 'r_ankle')
                    p_tip_r = self.get_point(frame_row, 'r_ski_tip')
                    p_tail_r = self.get_point(frame_row, 'r_ski_tail') # Or use ankle as tail? Standard is Tail->Tip
                    
                    if all(x is not None for x in [p_sh_r, p_ank_r, p_tip_r, p_tail_r]):
                        vec_body_r = p_sh_r - p_ank_r # Shoulder to Ankle
                        vec_ski_r = p_tip_r - p_tail_r
                        angles.append(self.calculate_vector_angle(vec_body_r, vec_ski_r))
                    
                    # SIDE L
                    p_sh_l = self.get_point(frame_row, 'l_shoulder')
                    p_ank_l = self.get_point(frame_row, 'l_ankle')
                    p_tip_l = self.get_point(frame_row, 'l_ski_tip')
                    p_tail_l = self.get_point(frame_row, 'l_ski_tail')

                    if all(x is not None for x in [p_sh_l, p_ank_l, p_tip_l, p_tail_l]):
                        vec_body_l = p_sh_l - p_ank_l
                        vec_ski_l = p_tip_l - p_tail_l
                        angles.append(self.calculate_vector_angle(vec_body_l, vec_ski_l))
                    
                    if angles:
                        res['body_ski_angle'] = np.mean(angles)

                # 3. METRIC: Telemark Offset (Landing)
                if tele_window and tele_window[0] <= f_idx <= tele_window[1]:
                    res['is_landing_phase'] = 1
                    
                    p_ank_r = self.get_point(frame_row, 'r_ankle')
                    p_ank_l = self.get_point(frame_row, 'l_ankle')
                    
                    if p_ank_r is not None and p_ank_l is not None:
                        # Absolute X distance
                        res['telemark_offset_x'] = abs(p_ank_r[0] - p_ank_l[0])

                # Only add row if we calculated at least one metric
                if res['v_style_angle'] is not np.nan or \
                   res['body_ski_angle'] is not np.nan or \
                   res['telemark_offset_x'] is not np.nan:
                    detailed_results.append(res)
        
        # Save Detailed CSV
        df_detailed = pd.DataFrame(detailed_results)
        df_detailed.to_csv(self.output_detailed, index=False)
        print(f"\n✅ Detailed metrics saved to {self.output_detailed}")
        
        self.aggregate_metrics(df_detailed)

    def aggregate_metrics(self, df_det):
        """Aggregates detailed frames into one row per jump."""
        if df_det.empty:
            print("⚠️ No metrics calculated to aggregate.")
            return

        summary_rows = []
        grouped = df_det.groupby('jump_id')
        
        for jump_id, group in grouped:
            row = {'jump_id': jump_id}
            
            # Averages (ignoring NaNs automatically)
            row['avg_v_style_angle'] = group['v_style_angle'].mean()
            row['avg_body_ski_angle'] = group['body_ski_angle'].mean()
            row['avg_symmetry_index'] = group['symmetry_index'].mean()
            row['telemark_score_raw'] = group['telemark_offset_x'].mean()
            
            # Stability (Standard Deviation of Angles in flight)
            # We combine V-style and BSA std dev as a proxy for jitter
            # If front view, use V-style jitter. If side, use BSA jitter.
            std_v = group['v_style_angle'].std()
            std_bsa = group['body_ski_angle'].std()
            
            # Pick the valid one (fillna 0 if not enough data for std)
            row['flight_stability_std'] = np.nanmean([std_v, std_bsa])
            
            summary_rows.append(row)
            
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(self.output_summary, index=False)
        print(f"✅ Summary metrics saved to {self.output_summary}")

def main():
    calc = MetricsCalculator()
    if calc.load_data():
        calc.run_calculations()

if __name__ == "__main__":
    main()