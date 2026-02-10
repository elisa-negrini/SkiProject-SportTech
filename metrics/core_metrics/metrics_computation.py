import pandas as pd
import numpy as np
import warnings
import re
from pathlib import Path
from scipy.signal import savgol_filter
from typing import Optional, Dict, Tuple

warnings.filterwarnings('ignore')


class MetricsComputation:
   
    def __init__(self, keypoints_file: str = 'dataset/keypoints_dataset.csv',
                 phases_file: str = 'dataset/jump_phases_SkiTB.csv'):
        """Initialize the calculator with paths and constants."""
        
        self.base_path = Path(__file__).parent.parent.parent
        self.keypoints_file = self.base_path / keypoints_file
        self.phases_file = self.base_path / phases_file
        
        self.output_dir = self.base_path / 'metrics' / 'core_metrics'
        self.output_detailed = self.output_dir / 'metrics_per_frame.csv'
        self.output_summary = self.output_dir / 'metrics_summary_per_jump.csv'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps = 30  # Frame rate
        self.dt = 1.0 / self.fps
        
        self.kpt_map = {
            'head': '1',
            'neck': '2',
            'center_pelvis': '9',
            'r_shoulder': '3',    'l_shoulder': '6',
            'r_hip': '17',        'l_hip': '10',
            'r_knee': '18',       'l_knee': '11',
            'r_ankle': '19',      'l_ankle': '12',
            'r_ski_tip': '23',    'r_ski_tail': '22',
            'l_ski_tip': '16',    'l_ski_tail': '15'
        }
        
        self.validity_ranges = {
            'v_style_angle': (10.0, 60.0),
            'body_ski_inclination': (0.0, 40.0),
            'knee_angle': (100.0, 180.0),
            'symmetry_index': (0.0, 30.0),
            'telemark_leg_angle': (0.0, 90.0),
            'knee_velocity': (50.0, 800.0),
            'telemark_scissor': (0.0, 0.30),
            'landing_absorption': (-2.0, 2.0),
        }
        
        self.smooth_window = 5
        self.smooth_poly = 2

    def apply_validity_range(self, value: float, metric_type: str) -> float:
        """Apply validity range filter. Returns NaN if outside range."""
        if pd.isna(value):
            return np.nan
        if metric_type not in self.validity_ranges:
            return value
        min_val, max_val = self.validity_ranges[metric_type]
        return value if min_val <= value <= max_val else np.nan
    
    def _normalize_jid(self, val) -> str:
        """Convert jump ID to standard format (JP####)."""
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s.lower().startswith('jump'):
            digits = ''.join(filter(str.isdigit, s))
            if digits:
                return f"JP{int(digits):04d}"
        if isinstance(val, (int, float)) and not pd.isna(val):
            try:
                return f"JP{int(val):04d}"
            except (ValueError, TypeError):
                pass
        return s
    
    def _extract_frame_num(self, fname: str) -> int:
        """Extract frame number from filename."""
        match = re.match(r'^(\d+)', str(fname))
        return int(match.group(1)) if match else -1
    
    def get_point(self, row, name: str) -> Optional[np.ndarray]:
        """Extract (x, y) coordinates of a keypoint."""
        try:
            kid = self.kpt_map[name]
            x = row[f'kpt_{kid}_x']
            y = row[f'kpt_{kid}_y']
            v = row.get(f'kpt_{kid}_v', 2)
            
            if pd.isna(x) or pd.isna(y) or v == 0:
                return None
            return np.array([x, y], dtype=float)
        except (KeyError, TypeError):
            return None
    
    def calculate_vector_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle in degrees between two 2D vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return np.nan
        
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def convert_bsa_to_inclination(self, bsa_angle: float) -> float:
        """Convert Body-Ski Angle (BSA) to inclination (0-40 deg)."""
        if pd.isna(bsa_angle):
            return np.nan
        inclination = 180.0 - bsa_angle
        return self.apply_validity_range(inclination, 'body_ski_inclination')
    
    def detect_view_type(self, row) -> str:
        """Detect view type (SIDE vs FRONT) based on shoulder width."""
        p_r_sh = self.get_point(row, 'r_shoulder')
        p_l_sh = self.get_point(row, 'l_shoulder')
        
        if p_r_sh is None or p_l_sh is None:
            return 'UNKNOWN'
        
        width = np.linalg.norm(p_r_sh - p_l_sh)
        return 'SIDE' if width < 0.04 else 'FRONT'
    
    def get_best_fit_vector(self, points):
        """Calculate direction vector using linear regression on points."""
        if len(points) < 2:
            return None
        
        data = np.array(points)
        x = data[:, 0]
        y = data[:, 1]
        
        try:
            slope, intercept = np.polyfit(x, y, 1)
            vec = np.array([1, slope])
            vec = vec / np.linalg.norm(vec)
            
            # Correct direction (point towards feet)
            vec_ref = points[-1] - points[0]
            if np.dot(vec, vec_ref) < 0:
                vec = -vec
            
            return vec
        except:
            return None

    def load_data(self) -> bool:
        """Load keypoints and jump phases data."""
        if not self.keypoints_file.exists():
            print(f"❌ Error: {self.keypoints_file} not found.")
            return False
        if not self.phases_file.exists():
            print(f"❌ Error: {self.phases_file} not found.")
            return False
        
        self.df_kpts = pd.read_csv(self.keypoints_file)
        self.df_phases = pd.read_csv(self.phases_file)
        
        self.df_phases['jump_id'] = self.df_phases['jump_id'].apply(self._normalize_jid)
        self.df_kpts['jump_id'] = self.df_kpts['jump_id'].apply(self._normalize_jid)
        
        self.df_kpts['frame_idx'] = self.df_kpts['frame_name'].apply(self._extract_frame_num)
        
        self.df_kpts = self.df_kpts.drop_duplicates(
            subset=['jump_id', 'frame_idx'], keep='first'
        )
        
        print(f" Loaded: {len(self.df_kpts)} keypoint frames, {len(self.df_phases)} jump phases")
        return True
    
    def compute_v_style_angle(self, frame_row) -> float:
        """Calculate V-style angle. Returns single angle value."""
        p_tip_r = self.get_point(frame_row, 'r_ski_tip')
        p_tail_r = self.get_point(frame_row, 'r_ski_tail')
        p_tip_l = self.get_point(frame_row, 'l_ski_tip')
        p_tail_l = self.get_point(frame_row, 'l_ski_tail')
        
        if not all(x is not None for x in [p_tip_r, p_tail_r, p_tip_l, p_tail_l]):
            return np.nan
        
        vec_ski_r = p_tip_r - p_tail_r
        vec_ski_l = p_tip_l - p_tail_l
        
        angle = self.calculate_vector_angle(vec_ski_r, vec_ski_l)
        return self.apply_validity_range(angle, 'v_style_angle')
    
    def compute_symmetry_index(self, frame_row, v_style_angle: float) -> float:
        """Calculate symmetry index for back view. Requires neck and pelvis."""
        if np.isnan(v_style_angle):
            return np.nan
        
        p_neck = self.get_point(frame_row, 'neck')
        p_pelvis = self.get_point(frame_row, 'center_pelvis')
        p_knee_r = self.get_point(frame_row, 'r_knee')
        p_knee_l = self.get_point(frame_row, 'l_knee')
        
        if not all(x is not None for x in [p_neck, p_pelvis, p_knee_r, p_knee_l]):
            return np.nan
        
        p_tip_r = self.get_point(frame_row, 'r_ski_tip')
        p_tail_r = self.get_point(frame_row, 'r_ski_tail')
        p_tip_l = self.get_point(frame_row, 'l_ski_tip')
        p_tail_l = self.get_point(frame_row, 'l_ski_tail')
        
        if not all(x is not None for x in [p_tip_r, p_tail_r, p_tip_l, p_tail_l]):
            return np.nan
        
        vec_ski_r = p_tip_r - p_tail_r
        vec_ski_l = p_tip_l - p_tail_l
        
        p_mid_knee = (p_knee_r + p_knee_l) / 2
        vec_axis = p_mid_knee - p_neck
        
        angle_r = self.calculate_vector_angle(vec_ski_r, vec_axis)
        angle_l = self.calculate_vector_angle(vec_ski_l, vec_axis)
        
        if np.isnan(angle_r) or np.isnan(angle_l):
            return np.nan
        
        sym_idx = abs(angle_r - angle_l)
        return self.apply_validity_range(sym_idx, 'symmetry_index')
    
    def compute_body_ski_angle(self, frame_row) -> float:
        """Calculate body-ski inclination angle."""
        pts_body_r = [self.get_point(frame_row, k) for k in 
                     ['r_shoulder', 'r_hip', 'r_knee', 'r_ankle']]
        pts_body_r = [p for p in pts_body_r if p is not None]
        vec_body_r = self.get_best_fit_vector(pts_body_r)
        
        pts_body_l = [self.get_point(frame_row, k) for k in 
                     ['l_shoulder', 'l_hip', 'l_knee', 'l_ankle']]
        pts_body_l = [p for p in pts_body_l if p is not None]
        vec_body_l = self.get_best_fit_vector(pts_body_l)
        
        p_tr = self.get_point(frame_row, 'r_ski_tip')
        p_tailr = self.get_point(frame_row, 'r_ski_tail')
        vec_ski_r = (p_tr - p_tailr) if (p_tr is not None and p_tailr is not None) else None
        
        p_tl = self.get_point(frame_row, 'l_ski_tip')
        p_taill = self.get_point(frame_row, 'l_ski_tail')
        vec_ski_l = (p_tl - p_taill) if (p_tl is not None and p_taill is not None) else None
        
        final_body_vec = None
        if vec_body_r is not None and vec_body_l is not None:
            final_body_vec = (vec_body_r + vec_body_l) / 2
        elif vec_body_r is not None:
            final_body_vec = vec_body_r
        elif vec_body_l is not None:
            final_body_vec = vec_body_l
        
        final_ski_vec = None
        if vec_ski_r is not None:
            vec_ski_r = vec_ski_r / np.linalg.norm(vec_ski_r)
        if vec_ski_l is not None:
            vec_ski_l = vec_ski_l / np.linalg.norm(vec_ski_l)
        
        if vec_ski_r is not None and vec_ski_l is not None:
            final_ski_vec = (vec_ski_r + vec_ski_l) / 2
        elif vec_ski_r is not None:
            final_ski_vec = vec_ski_r
        elif vec_ski_l is not None:
            final_ski_vec = vec_ski_l
        
        if final_body_vec is not None and final_ski_vec is not None:
            raw_bsa = self.calculate_vector_angle(final_body_vec, final_ski_vec)
            return self.convert_bsa_to_inclination(raw_bsa)
        
        return np.nan
    
    def compute_telemark_metrics(self, frame_row) -> Tuple[float, float, float, float]:
        """Calculate telemark landing metrics."""
        p_ank_r = self.get_point(frame_row, 'r_ankle')
        p_ank_l = self.get_point(frame_row, 'l_ankle')
        
        offset_x = np.nan
        proj_ski = np.nan
        depth_ratio = np.nan
        leg_angle = np.nan
        
        if p_ank_r is not None and p_ank_l is not None:
            offset_x = abs(p_ank_r[0] - p_ank_l[0])
        
        p_tip_r = self.get_point(frame_row, 'r_ski_tip')
        p_tail_r = self.get_point(frame_row, 'r_ski_tail')
        p_tip_l = self.get_point(frame_row, 'l_ski_tip')
        p_tail_l = self.get_point(frame_row, 'l_ski_tail')
        
        ski_dirs = []
        if p_tip_r is not None and p_tail_r is not None:
            ski_dirs.append(p_tip_r - p_tail_r)
        if p_tip_l is not None and p_tail_l is not None:
            ski_dirs.append(p_tip_l - p_tail_l)
        
        if ski_dirs and p_ank_r is not None and p_ank_l is not None:
            avg_ski_vec = np.mean(ski_dirs, axis=0)
            norm = np.linalg.norm(avg_ski_vec)
            if norm > 0:
                unit_ski_vec = avg_ski_vec / norm
                vec_ankles = p_ank_r - p_ank_l
                proj_ski = abs(np.dot(vec_ankles, unit_ski_vec))
                
                p_neck = self.get_point(frame_row, 'neck')
                p_pelvis = self.get_point(frame_row, 'center_pelvis')
                
                if p_neck is not None and p_pelvis is not None:
                    back_len_px = np.linalg.norm(p_neck - p_pelvis)
                    if back_len_px > 0:
                        depth_ratio = proj_ski / back_len_px
        
        p_hip_r = self.get_point(frame_row, 'r_hip')
        p_knee_r = self.get_point(frame_row, 'r_knee')
        p_hip_l = self.get_point(frame_row, 'l_hip')
        p_knee_l = self.get_point(frame_row, 'l_knee')
        
        if all(x is not None for x in [p_hip_r, p_knee_r, p_hip_l, p_knee_l]):
            vec_femur_r = p_knee_r - p_hip_r
            vec_femur_l = p_knee_l - p_hip_l
            leg_angle = self.calculate_vector_angle(vec_femur_r, vec_femur_l)
            leg_angle = self.apply_validity_range(leg_angle, 'telemark_leg_angle')
        
        return offset_x, proj_ski, depth_ratio, leg_angle
    
    def compute_takeoff_knee_angle(self, frame_row) -> float:
        """Calculate knee angle at takeoff."""
        angles = []
        
        p_hip_r = self.get_point(frame_row, 'r_hip')
        p_knee_r = self.get_point(frame_row, 'r_knee')
        p_ank_r = self.get_point(frame_row, 'r_ankle')
        
        if all(x is not None for x in [p_hip_r, p_knee_r, p_ank_r]):
            vec_femur = p_hip_r - p_knee_r
            vec_tibia = p_ank_r - p_knee_r
            angles.append(self.calculate_vector_angle(vec_femur, vec_tibia))
        
        p_hip_l = self.get_point(frame_row, 'l_hip')
        p_knee_l = self.get_point(frame_row, 'l_knee')
        p_ank_l = self.get_point(frame_row, 'l_ankle')
        
        if all(x is not None for x in [p_hip_l, p_knee_l, p_ank_l]):
            vec_femur = p_hip_l - p_knee_l
            vec_tibia = p_ank_l - p_knee_l
            angles.append(self.calculate_vector_angle(vec_femur, vec_tibia))
        
        if angles:
            avg_knee = np.mean(angles)
            return self.apply_validity_range(avg_knee, 'knee_angle')
        
        return np.nan

    def compute_takeoff_dynamics(self, jump_df: pd.DataFrame,
                                takeoff_frame: int,
                                window: int = 10) -> Dict:
        """Analyze takeoff timing and explosiveness."""
        knee_angles = {}
        
        start_f = takeoff_frame - window
        end_f = takeoff_frame + window
        
        for _, row in jump_df.iterrows():
            frame_idx = row['frame_idx']
            
            if frame_idx < start_f or frame_idx > end_f:
                continue
            
            for side in ['r', 'l']:
                hip = self.get_point(row, f'{side}_hip')
                knee = self.get_point(row, f'{side}_knee')
                ankle = self.get_point(row, f'{side}_ankle')
                
                if hip is not None and knee is not None and ankle is not None:
                    v1 = hip - knee
                    v2 = ankle - knee
                    angle = self.calculate_vector_angle(v1, v2)
                    
                    if not np.isnan(angle):
                        knee_angles[frame_idx] = angle
                        break
        
        if len(knee_angles) < 5:
            return {
                'takeoff_timing_offset': np.nan,
                'takeoff_peak_velocity': np.nan
            }
        
        angles_series = pd.Series(knee_angles).sort_index()
        angles_array = angles_series.values
        frames_array = angles_series.index.values
        
        if len(angles_array) >= self.smooth_window:
            try:
                smoothed = savgol_filter(angles_array, self.smooth_window, self.smooth_poly)
            except:
                smoothed = angles_array
        else:
            smoothed = angles_array
        
        velocity = np.gradient(smoothed)
        
        peak_idx = np.argmax(velocity)
        peak_frame = frames_array[peak_idx]
        timing_offset = peak_frame - takeoff_frame
        
        peak_velocity = velocity[peak_idx] * self.fps
        peak_velocity = self.apply_validity_range(peak_velocity, 'knee_velocity')
        
        return {
            'takeoff_timing_offset': timing_offset,
            'takeoff_peak_velocity': peak_velocity
        }
    
    def compute_telemark_quality(self, jump_df: pd.DataFrame,
                                landing_frame: int,
                                window: int = 15) -> Dict:
        """Compute landing (Telemark) quality metrics."""
        telemark_data = []
        
        start_f = landing_frame
        end_f = landing_frame + window
        
        for _, row in jump_df.iterrows():
            frame_idx = row['frame_idx']
            
            if frame_idx < start_f or frame_idx > end_f:
                continue
            
            r_ankle = self.get_point(row, 'r_ankle')
            l_ankle = self.get_point(row, 'l_ankle')
            r_hip = self.get_point(row, 'r_hip')
            l_hip = self.get_point(row, 'l_hip')
            pelvis = self.get_point(row, 'center_pelvis')
            
            if r_ankle is not None and l_ankle is not None:
                ankle_y_diff = abs(r_ankle[1] - l_ankle[1])
                
                hip_height = None
                if r_hip is not None and l_hip is not None:
                    avg_hip_y = (r_hip[1] + l_hip[1]) / 2
                    avg_ankle_y = (r_ankle[1] + l_ankle[1]) / 2
                    hip_height = abs(avg_hip_y - avg_ankle_y)
                
                if hip_height and hip_height > 0.01:
                    normalized_scissor = ankle_y_diff / hip_height
                else:
                    normalized_scissor = ankle_y_diff
                
                telemark_data.append({
                    'frame': frame_idx,
                    'scissor': normalized_scissor,
                    'pelvis_y': pelvis[1] if pelvis is not None else np.nan
                })
        
        if len(telemark_data) < 3:
            return {
                'telemark_scissor_mean': np.nan,
                'telemark_stability': np.nan,
                'landing_absorption_rate': np.nan
            }
        
        df_tele = pd.DataFrame(telemark_data)
        
        scissor_mean = df_tele['scissor'].mean()
        scissor_mean = self.apply_validity_range(scissor_mean, 'telemark_scissor')
        
        scissor_std = df_tele['scissor'].std()
        stability = 1.0 / (scissor_std + 0.001) if scissor_std < 1 else 0.0
        
        pelvis_y = df_tele['pelvis_y'].dropna()
        if len(pelvis_y) >= 3:
            absorption_rate = (pelvis_y.iloc[-1] - pelvis_y.iloc[0]) / len(pelvis_y) * self.fps
            absorption_rate = self.apply_validity_range(absorption_rate, 'landing_absorption')
        else:
            absorption_rate = np.nan
        
        return {
            'telemark_scissor_mean': scissor_mean,
            'telemark_stability': stability,
            'landing_absorption_rate': absorption_rate
        }

    def process(self) -> bool:
        """Main processing loop for all jumps and frames."""
        
        if not self.load_data():
            return False
        
        print("\n" + "=" * 80)
        print("UNIFIED METRICS COMPUTATION")
        print("Ski Jumping Biomechanical Analysis (Core + Advanced Metrics)")
        print("=" * 80)
        
        detailed_results = []
        summary_results = []
        
        for _, phase_row in self.df_phases.iterrows():
            jump_id = phase_row['jump_id']
            if pd.isna(jump_id) or jump_id == '':
                continue
            
            print(f"\n📊 Processing {jump_id}...")
            
            jump_frames = self.df_kpts[self.df_kpts['jump_id'] == jump_id].copy()
            if jump_frames.empty:
                print(f"   ⚠️  No frames found for {jump_id}")
                continue
            
            telemark_flag = phase_row.get('telemark_f_l', 'unknown')
            jump_summary = {'jump_id': jump_id}
            
            v_front_window = None
            if pd.notna(phase_row.get('v_style_front_start')):
                v_front_window = (phase_row['v_style_front_start'], phase_row['v_style_front_end'])
            
            v_back_window = None
            if pd.notna(phase_row.get('v_style_back_start')):
                v_back_window = (phase_row['v_style_back_start'], phase_row['v_style_back_end'])
            
            bsa_window = None
            if phase_row.get('body_ski_measurable') == 1.0:
                bsa_window = (phase_row['bsa_start'], phase_row['bsa_end'])
            
            tele_window = None
            if phase_row.get('telemark_measurable') == 1.0:
                tele_window = (phase_row['telemark_start'], phase_row['telemark_end'])
            
            for _, frame_row in jump_frames.iterrows():
                f_idx = frame_row['frame_idx']
                view_type = self.detect_view_type(frame_row)
                
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
                
                is_front = v_front_window and (v_front_window[0] <= f_idx <= v_front_window[1])
                is_back = v_back_window and (v_back_window[0] <= f_idx <= v_back_window[1])
                
                if is_front or is_back:
                    res['is_flight_phase'] = 1
                    v_angle = self.compute_v_style_angle(frame_row)
                    
                    if is_front:
                        res['v_style_angle_front'] = v_angle
                    if is_back:
                        res['v_style_angle_back'] = v_angle
                        if not np.isnan(v_angle):
                            res['symmetry_index_back'] = self.compute_symmetry_index(frame_row, v_angle)
                
                if bsa_window and bsa_window[0] <= f_idx <= bsa_window[1]:
                    res['is_flight_phase'] = 1
                    res['body_ski_angle'] = self.compute_body_ski_angle(frame_row)
                
                if tele_window and tele_window[0] <= f_idx <= tele_window[1]:
                    res['is_landing_phase'] = 1
                    offset_x, proj_ski, depth_ratio, leg_angle = self.compute_telemark_metrics(frame_row)
                    res['telemark_offset_x_raw'] = offset_x
                    res['telemark_proj_ski_raw'] = proj_ski
                    res['telemark_depth_back_ratio'] = depth_ratio
                    res['telemark_leg_angle'] = leg_angle
                
                takeoff_frame = phase_row.get('take_off_frame')
                if pd.notna(takeoff_frame) and f_idx == int(takeoff_frame):
                    res['takeoff_knee_angle'] = self.compute_takeoff_knee_angle(frame_row)
                
                metrics_found = [
                    res['v_style_angle_front'],
                    res['v_style_angle_back'],
                    res['body_ski_angle'],
                    res.get('telemark_offset_x_raw'),
                    res.get('telemark_proj_ski_raw'),
                    res.get('telemark_depth_back_ratio'),
                    res.get('takeoff_knee_angle'),
                    res.get('symmetry_index_back'),
                    res.get('telemark_leg_angle')
                ]
                
                if any(pd.notna(x) for x in metrics_found):
                    detailed_results.append(res)
            
            advanced_metrics = {'jump_id': jump_id}
            
            takeoff_frame = phase_row.get('take_off_frame')
            has_takeoff = pd.notna(takeoff_frame) and phase_row.get('take_off_measurable') == 1
            if has_takeoff:
                takeoff_dynamics = self.compute_takeoff_dynamics(jump_frames, int(takeoff_frame))
                advanced_metrics.update(takeoff_dynamics)
            else:
                advanced_metrics['takeoff_timing_offset'] = np.nan
                advanced_metrics['takeoff_peak_velocity'] = np.nan
            
            landing_frame = phase_row.get('landing')
            has_landing = pd.notna(landing_frame)
            if has_landing:
                telemark_quality = self.compute_telemark_quality(jump_frames, int(landing_frame))
                advanced_metrics.update(telemark_quality)
            else:
                advanced_metrics['telemark_scissor_mean'] = np.nan
                advanced_metrics['telemark_stability'] = np.nan
                advanced_metrics['landing_absorption_rate'] = np.nan
            
            summary_results.append(advanced_metrics)
        
        df_detailed = pd.DataFrame(detailed_results)
        df_detailed.to_csv(self.output_detailed, index=False)
        print(f"\n Detailed metrics saved: {self.output_detailed}")
        
        df_aggregated = self.aggregate_core_metrics(df_detailed)
        
        df_advanced = pd.DataFrame(summary_results)
        df_summary = df_aggregated.merge(df_advanced, on='jump_id', how='left')
        
        df_summary.to_csv(self.output_summary, index=False)
        print(f" Summary metrics saved: {self.output_summary}")
        
        print("\n" + "=" * 80)
        print(f" Processing complete: {len(df_summary)} jumps")
        print("=" * 80 + "\n")
        
        return True
    
    def aggregate_core_metrics(self, df_detailed: pd.DataFrame) -> pd.DataFrame:
        """Aggregate core metrics per jump."""
        if df_detailed.empty:
            return pd.DataFrame()
        
        summary_rows = []
        
        for jump_id, group in df_detailed.groupby('jump_id'):
            row = {'jump_id': jump_id}
            
            row['avg_v_style_front'] = group['v_style_angle_front'].mean()
            row['avg_v_style_back'] = group['v_style_angle_back'].mean()
            
            row['takeoff_knee_angle'] = group['takeoff_knee_angle'].mean()
            row['avg_body_ski_angle'] = group['body_ski_angle'].mean()
            row['avg_symmetry_index_back'] = group['symmetry_index_back'].mean()
            
            row['avg_telemark_proj_ski'] = group['telemark_proj_ski_raw'].mean()
            row['avg_telemark_depth_ratio'] = group['telemark_depth_back_ratio'].mean()
            row['avg_telemark_leg_angle'] = group['telemark_leg_angle'].mean()
            
            summary_rows.append(row)
        
        return pd.DataFrame(summary_rows)

def main():
    """Main entry point."""
    calculator = MetricsComputation()
    success = calculator.process()
    
    if not success:
        print("\n❌ Processing failed!")
        return 1
    
    print(" All metrics computed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())