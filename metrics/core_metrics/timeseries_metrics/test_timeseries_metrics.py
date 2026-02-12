import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesMetricsCalculator:

    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent.parent
        self.keypoints_file = self.base_path / 'dataset'/ 'keypoints_dataset.csv'
        self.phases_file = self.base_path / 'dataset'/ 'jump_phases_SkiTB.csv'
        self.jp_data_file = self.base_path / 'dataset'/ 'JP_data.csv'
        
        # Output 
        self.output_dir = self.base_path / 'metrics' / 'core_metrics' / 'timeseries_metrics'
        self.output_per_frame = self.output_dir / 'timeseries_per_frame.csv'
        self.output_summary = self.output_dir / 'additional_timeseries_metrics.csv'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Frame rate (30 fps)
        self.fps = 30
        self.dt = 1.0 / self.fps  
        
        self.kpt_map = {
            'head': '1',
            'neck': '2',
            'center_pelvis': '9',
            'r_shoulder': '3',  'l_shoulder': '6',
            'r_hip': '17',      'l_hip': '10',
            'r_knee': '18',     'l_knee': '11',
            'r_ankle': '19',    'l_ankle': '12',
            'r_ski_tip': '23',  'r_ski_tail': '22',
            'l_ski_tip': '16',  'l_ski_tail': '15'
        }
        
        self.smooth_window = 5 
        self.smooth_poly = 2
        
    def load_data(self) -> bool:
        
        if not self.keypoints_file.exists():
            print(f"❌ Keypoints file not found: {self.keypoints_file}")
            return False
        if not self.phases_file.exists():
            print(f"❌ Phases file not found: {self.phases_file}")
            return False
            
        self.df_kpts = pd.read_csv(self.keypoints_file)
        self.df_phases = pd.read_csv(self.phases_file)
        
        if self.jp_data_file.exists():
            self.df_jp = pd.read_csv(self.jp_data_file)
        else:
            self.df_jp = None
            print("⚠️ JP_data.csv not found - scores won't be available")
        
        self.df_phases['jump_id'] = self.df_phases['jump_id'].apply(self._normalize_jid)
        self.df_kpts['jump_id'] = self.df_kpts['jump_id'].apply(self._normalize_jid)
        
        def extract_frame_num(fname):
            import re
            match = re.match(r'^(\d+)', str(fname))
            if match:
                return int(match.group(1))
            return -1
        
        self.df_kpts['frame_idx'] = self.df_kpts['frame_name'].apply(extract_frame_num)
        
        self.df_kpts = self.df_kpts.drop_duplicates(subset=['jump_id', 'frame_idx'], keep='first')
        
        print(f" Loaded: {len(self.df_kpts)} keypoint frames, {len(self.df_phases)} jump phases")
        return True
    
    def _normalize_jid(self, val) -> str:
        """Converts 'jump5' -> 'JP0005' format."""
        s = str(val).strip()
        if s.lower().startswith('jump'):
            num = int(s[4:])
            return f"JP{num:04d}"
        return s
    
    def get_point(self, row, name: str) -> Optional[np.ndarray]:
        """Extract (x, y) coordinates of a keypoint."""
        try:
            kpt_id = self.kpt_map[name]
            x = row[f'kpt_{kpt_id}_x']
            y = row[f'kpt_{kpt_id}_y']
            v = row.get(f'kpt_{kpt_id}_v', 2)
            
            if pd.isna(x) or pd.isna(y) or v == 0:
                return None
            return np.array([x, y])
        except KeyError:
            return None
    
    def calculate_angle_3points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3.
        Returns angle in degrees.
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return np.nan
        
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def calculate_vector_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return np.nan
        
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    # METRIC 1: KNEE VELOCITY (Explosiveness at Take-off)
    def compute_knee_angle_series(self, jump_df: pd.DataFrame, side: str = 'r') -> pd.Series:
        """
        Compute knee angle for each frame.
        Knee angle = angle at knee between hip-knee-ankle.
        """
        angles = {}
        
        for _, row in jump_df.iterrows():
            frame_idx = row['frame_idx']
            
            hip = self.get_point(row, f'{side}_hip')
            knee = self.get_point(row, f'{side}_knee')
            ankle = self.get_point(row, f'{side}_ankle')
            
            if hip is not None and knee is not None and ankle is not None:
                angle = self.calculate_angle_3points(hip, knee, ankle)
                angles[frame_idx] = angle
            else:
                angles[frame_idx] = np.nan
        
        return pd.Series(angles).sort_index()
    
    def compute_knee_velocity(self, knee_angles: pd.Series, takeoff_frame: int, 
                               window: int = 5) -> Dict:
        """
        Compute angular velocity of knee extension around take-off.
        """
        start_frame = takeoff_frame - window
        end_frame = takeoff_frame + window
        
        mask = (knee_angles.index >= start_frame) & (knee_angles.index <= end_frame)
        window_angles = knee_angles[mask].dropna()
        
        if len(window_angles) < 3:
            return {
                'knee_peak_velocity': np.nan
            }
        
        angles_array = window_angles.values
        frames_array = window_angles.index.values
        
        if len(angles_array) >= self.smooth_window:
            try:
                smoothed = savgol_filter(angles_array, self.smooth_window, self.smooth_poly)
                velocity = np.gradient(smoothed)
            except:
                velocity = np.gradient(angles_array)
        else:
            velocity = np.gradient(angles_array)
        
        velocity_per_sec = velocity * self.fps
        
        peak_velocity = np.nanmax(velocity_per_sec)
        
        return {
            'knee_peak_velocity': peak_velocity
        }
    
    # METRIC 2: FLIGHT FROZENNESS (Stability during flight)   
    def compute_body_ski_angle_series(self, jump_df: pd.DataFrame) -> pd.Series:
        """
        Compute body-ski angle for each frame.
        Body axis: neck -> center_pelvis
        Ski axis: avg of left and right ski vectors
        """
        angles = {}
        
        for _, row in jump_df.iterrows():
            frame_idx = row['frame_idx']
            
            neck = self.get_point(row, 'neck')
            pelvis = self.get_point(row, 'center_pelvis')
            
            r_tip = self.get_point(row, 'r_ski_tip')
            r_tail = self.get_point(row, 'r_ski_tail')
            l_tip = self.get_point(row, 'l_ski_tip')
            l_tail = self.get_point(row, 'l_ski_tail')
            
            if neck is None or pelvis is None:
                angles[frame_idx] = np.nan
                continue
            
            body_vec = neck - pelvis
            
            ski_vecs = []
            if r_tip is not None and r_tail is not None:
                ski_vecs.append(r_tip - r_tail)
            if l_tip is not None and l_tail is not None:
                ski_vecs.append(l_tip - l_tail)
            
            if len(ski_vecs) == 0:
                angles[frame_idx] = np.nan
                continue
            
            avg_ski_vec = np.mean(ski_vecs, axis=0)
            
            angle = self.calculate_vector_angle(body_vec, avg_ski_vec)
            angles[frame_idx] = angle
        
        return pd.Series(angles).sort_index()
    
    def compute_flight_frozenness(self, bsa_series: pd.Series, 
                                   bsa_start: int, bsa_end: int) -> Dict:
        """
        Compute stability metrics during flight phase.
        Low std = "frozen" position = good
        High std = corrections/instability = bad
        """
        mask = (bsa_series.index >= bsa_start) & (bsa_series.index <= bsa_end)
        flight_angles = bsa_series[mask].dropna()
        
        if len(flight_angles) < 3:
            return {
                'flight_std': np.nan,
                'flight_mean_bsa': np.nan,
                'flight_jitter': np.nan
            }
        
        flight_std = flight_angles.std()
        
        flight_mean = flight_angles.mean()
        
        angles = flight_angles.values
        diffs = np.diff(angles)
        jitter = np.std(diffs) if len(diffs) > 1 else np.nan

        return {
            'flight_std': flight_std,
            'flight_mean_bsa': flight_mean,
            'flight_jitter': jitter
        }
    
    # METRIC 3: LANDING SMOOTHNESS (Absorption quality)    
    def compute_hip_height_series(self, jump_df: pd.DataFrame) -> pd.Series:
        """
        Compute hip height (y-coordinate of center_pelvis) for each frame.
        Note: In image coordinates, Y increases downward, so lower Y = higher position.
        """
        heights = {}
        
        for _, row in jump_df.iterrows():
            frame_idx = row['frame_idx']
            pelvis = self.get_point(row, 'center_pelvis')
            
            if pelvis is not None:
                heights[frame_idx] = 1.0 - pelvis[1]
            else:
                heights[frame_idx] = np.nan
        
        return pd.Series(heights).sort_index()
    
    def compute_landing_smoothness(self, hip_series: pd.Series, knee_series: pd.Series,
                                    landing_frame: int, window: int = 15) -> Dict:
        """
        Compute landing absorption metrics.
        Smooth landing: gradual hip descent
        Hard landing: sudden drop
        """
        start_frame = landing_frame
        end_frame = landing_frame + window
        
        mask_hip = (hip_series.index >= start_frame) & (hip_series.index <= end_frame)
        landing_hip = hip_series[mask_hip].dropna()
        
        mask_knee = (knee_series.index >= start_frame) & (knee_series.index <= end_frame)
        landing_knee = knee_series[mask_knee].dropna()
        
        result = {
            'landing_hip_velocity': np.nan,
            'landing_knee_compression': np.nan
        }
        
        if len(landing_hip) >= 3:
            hip_velocity = np.gradient(landing_hip.values) * self.fps
            result['landing_hip_velocity'] = np.abs(hip_velocity).mean()
        
        if len(landing_knee) >= 3:
            result['landing_knee_compression'] = landing_knee.iloc[0] - landing_knee.min()
        
        return result
    
    # MAIN PROCESSING    
    def process_jump(self, jump_id: str, phase_row: pd.Series) -> Dict:
        """Process a single jump and compute all time-series metrics."""
        
        jump_df = self.df_kpts[self.df_kpts['jump_id'] == jump_id].copy()
        
        if jump_df.empty:
            print(f"  ⚠️ No keypoints for {jump_id}")
            return None
        
        result = {'jump_id': jump_id}
        
        takeoff_frame = phase_row.get('take_off_frame')
        if pd.notna(takeoff_frame) and phase_row.get('take_off_measurable') == 1:
            takeoff_frame = int(takeoff_frame)
            
            knee_r = self.compute_knee_angle_series(jump_df, 'r')
            knee_l = self.compute_knee_angle_series(jump_df, 'l')
            
            knee_series = knee_r if knee_r.notna().sum() >= knee_l.notna().sum() else knee_l
            
            knee_metrics = self.compute_knee_velocity(knee_series, takeoff_frame)
            result.update(knee_metrics)
        else:
            result.update({
                'knee_peak_velocity': np.nan
            })
        
        bsa_start = phase_row.get('bsa_start')
        bsa_end = phase_row.get('bsa_end')
        
        if pd.notna(bsa_start) and pd.notna(bsa_end):
            bsa_start = int(bsa_start)
            bsa_end = int(bsa_end)
            
            bsa_series = self.compute_body_ski_angle_series(jump_df)
            
            landing_frame = phase_row.get('landing')
            if pd.notna(landing_frame):
                extended_end = int(landing_frame)
            else:
                extended_end = bsa_end + 50
            
            frozenness_metrics = self.compute_flight_frozenness(bsa_series, bsa_start, extended_end)
            result.update(frozenness_metrics)
        else:
            result.update({
                'flight_std': np.nan,
                'flight_mean_bsa': np.nan,
                'flight_jitter': np.nan
            })
        
        landing_frame = phase_row.get('landing')
        
        if pd.notna(landing_frame):
            landing_frame = int(landing_frame)
            
            hip_series = self.compute_hip_height_series(jump_df)
            knee_series = self.compute_knee_angle_series(jump_df, 'r')  # Use right knee
            
            landing_metrics = self.compute_landing_smoothness(hip_series, knee_series, landing_frame)
            result.update(landing_metrics)
        else:
            result.update({
                'landing_hip_velocity': np.nan,
                'landing_knee_compression': np.nan
            })
        
        return result
    
    def process_all(self) -> bool:
        
        if not self.load_data():
            return False
        
        print("COMPUTING TIME-SERIES METRICS")
        
        results = []
        per_frame_results = []
        
        for _, phase_row in self.df_phases.iterrows():
            jump_id = phase_row['jump_id']
            
            if pd.isna(jump_id) or jump_id == '':
                continue
            
            jump_id = self._normalize_jid(jump_id)
            print(f"\n Processing {jump_id}...")
            
            jump_result = self.process_jump(jump_id, phase_row)
            
            if jump_result:
                results.append(jump_result)
                
                jump_df = self.df_kpts[self.df_kpts['jump_id'] == jump_id].copy()
                if not jump_df.empty:
                    bsa_series = self.compute_body_ski_angle_series(jump_df)
                    knee_series = self.compute_knee_angle_series(jump_df, 'r')
                    hip_series = self.compute_hip_height_series(jump_df)
                    
                    for frame_idx in jump_df['frame_idx'].unique():
                        per_frame_results.append({
                            'jump_id': jump_id,
                            'frame_idx': frame_idx,
                            'body_ski_angle': bsa_series.get(frame_idx, np.nan),
                            'knee_angle_r': knee_series.get(frame_idx, np.nan),
                            'hip_height': hip_series.get(frame_idx, np.nan)
                        })
        
        df_summary = pd.DataFrame(results)
        df_summary.to_csv(self.output_summary, index=False)
        print(f"\n Summary saved: {self.output_summary}")
        print(f"   Jumps processed: {len(df_summary)}")
        
        df_per_frame = pd.DataFrame(per_frame_results)
        df_per_frame.to_csv(self.output_per_frame, index=False)
        print(f" Per-frame saved: {self.output_per_frame}")
        
        print("SUMMARY STATISTICS")
        
        metrics_to_show = ['knee_peak_velocity', 'flight_std', 'flight_jitter', 'landing_hip_velocity']
        for metric in metrics_to_show:
            if metric in df_summary.columns:
                valid = df_summary[metric].dropna()
                if len(valid) > 0:
                    print(f"\n{metric}:")
                    print(f"  Mean: {valid.mean():.3f}")
                    print(f"  Std:  {valid.std():.3f}")
                    print(f"  Min:  {valid.min():.3f}")
                    print(f"  Max:  {valid.max():.3f}")
                    print(f"  N:    {len(valid)}")
        
        return True

if __name__ == "__main__":
    calculator = TimeSeriesMetricsCalculator()
    calculator.process_all()