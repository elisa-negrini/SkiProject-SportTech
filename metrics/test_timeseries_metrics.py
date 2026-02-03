"""
TEST: Time-Series Metrics for Ski Jumping Analysis
===================================================
Computes "perspective-robust" metrics that focus on TEMPORAL DYNAMICS
rather than absolute angles (which are distorted by diagonal camera view).

Metrics computed:
1. Knee Velocity (Explosiveness): Peak angular velocity at take-off
2. Flight Frozenness: Stability (std) of body-ski angle during flight
3. Landing Smoothness: Hip descent rate after landing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesMetricsCalculator:
    """
    Calculates time-series based metrics that are robust to perspective distortion.
    Focus on DERIVATIVES and VARIABILITY rather than absolute values.
    """
    
    def __init__(self):
        # --- PATHS ---
        self.base_path = Path(__file__).parent.parent
        self.keypoints_file = self.base_path / 'keypoints_dataset.csv'
        self.phases_file = self.base_path / 'jump_phases_SkiTB.csv'
        self.jp_data_file = self.base_path / 'JP_data.csv'
        
        # Output (in subfolder)
        self.output_dir = self.base_path / 'metrics' / 'timeseries_metrics'
        self.output_per_frame = self.output_dir / 'timeseries_per_frame.csv'
        self.output_summary = self.output_dir / 'timeseries_summary.csv'
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Frame rate (30 fps)
        self.fps = 30
        self.dt = 1.0 / self.fps  # seconds per frame
        
        # Keypoint mapping (same as original MetricsCalculator)
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
        
        # Smoothing parameters for derivatives
        self.smooth_window = 5  # frames for Savitzky-Golay filter
        self.smooth_poly = 2
        
    def load_data(self) -> bool:
        """Load keypoints, phases, and JP data."""
        
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
        
        # Normalize jump IDs
        self.df_phases['jump_id'] = self.df_phases['jump_id'].apply(self._normalize_jid)
        self.df_kpts['jump_id'] = self.df_kpts['jump_id'].apply(self._normalize_jid)
        
        # Parse frame numbers from filenames
        # Handles formats like: "00324.jpg", "00177_jpg.rf.xxx", etc.
        def extract_frame_num(fname):
            import re
            # Try to extract leading digits
            match = re.match(r'^(\d+)', str(fname))
            if match:
                return int(match.group(1))
            return -1
        
        self.df_kpts['frame_idx'] = self.df_kpts['frame_name'].apply(extract_frame_num)
        
        # Remove duplicates (keep first occurrence)
        self.df_kpts = self.df_kpts.drop_duplicates(subset=['jump_id', 'frame_idx'], keep='first')
        
        print(f"✅ Loaded: {len(self.df_kpts)} keypoint frames, {len(self.df_phases)} jump phases")
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
        """Calculate angle between two vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return np.nan
        
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    # =========================================================================
    # METRIC 1: KNEE VELOCITY (Explosiveness at Take-off)
    # =========================================================================
    
    def compute_knee_angle_series(self, jump_df: pd.DataFrame, side: str = 'r') -> pd.Series:
        """
        Compute knee angle for each frame.
        Knee angle = angle at knee between hip-knee-ankle.
        
        Args:
            jump_df: DataFrame with keypoints for a single jump
            side: 'r' for right, 'l' for left
        
        Returns:
            Series with knee angles indexed by frame_idx
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
        
        Args:
            knee_angles: Series of knee angles by frame
            takeoff_frame: Frame index of take-off
            window: Frames before/after take-off to analyze
        
        Returns:
            Dict with peak velocity, mean velocity, etc.
        """
        # Get window around take-off
        start_frame = takeoff_frame - window
        end_frame = takeoff_frame + window
        
        # Filter to window
        mask = (knee_angles.index >= start_frame) & (knee_angles.index <= end_frame)
        window_angles = knee_angles[mask].dropna()
        
        if len(window_angles) < 3:
            return {
                'knee_peak_velocity': np.nan,
                'knee_mean_velocity': np.nan,
                'knee_angle_at_takeoff': np.nan,
                'knee_extension_range': np.nan
            }
        
        # Compute derivative (angular velocity in deg/frame)
        angles_array = window_angles.values
        frames_array = window_angles.index.values
        
        # Numerical derivative
        if len(angles_array) >= self.smooth_window:
            # Smooth first, then differentiate
            try:
                smoothed = savgol_filter(angles_array, self.smooth_window, self.smooth_poly)
                velocity = np.gradient(smoothed)
            except:
                velocity = np.gradient(angles_array)
        else:
            velocity = np.gradient(angles_array)
        
        # Convert to deg/second
        velocity_per_sec = velocity * self.fps
        
        # Peak velocity (positive = extension)
        peak_velocity = np.nanmax(velocity_per_sec)
        mean_velocity = np.nanmean(velocity_per_sec[velocity_per_sec > 0]) if np.any(velocity_per_sec > 0) else np.nan
        
        # Angle at take-off
        takeoff_angle = np.nan
        if takeoff_frame in window_angles.index:
            takeoff_angle = window_angles[takeoff_frame]
        
        # Extension range (max - min in window)
        extension_range = np.nanmax(angles_array) - np.nanmin(angles_array)
        
        return {
            'knee_peak_velocity': peak_velocity,
            'knee_mean_velocity': mean_velocity,
            'knee_angle_at_takeoff': takeoff_angle,
            'knee_extension_range': extension_range
        }
    
    # =========================================================================
    # METRIC 2: FLIGHT FROZENNESS (Stability during flight)
    # =========================================================================
    
    def compute_body_ski_angle_series(self, jump_df: pd.DataFrame) -> pd.Series:
        """
        Compute body-ski angle for each frame.
        Body axis: neck -> center_pelvis
        Ski axis: avg of left and right ski vectors
        
        Returns:
            Series with BSA indexed by frame_idx
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
            
            # Body vector (pointing from pelvis to neck)
            body_vec = neck - pelvis
            
            # Ski vectors (try both, use available)
            ski_vecs = []
            if r_tip is not None and r_tail is not None:
                ski_vecs.append(r_tip - r_tail)
            if l_tip is not None and l_tail is not None:
                ski_vecs.append(l_tip - l_tail)
            
            if len(ski_vecs) == 0:
                angles[frame_idx] = np.nan
                continue
            
            # Average ski vector
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
        
        Args:
            bsa_series: Series of body-ski angles
            bsa_start: Start frame of BSA measurement window
            bsa_end: End frame of BSA measurement window
        
        Returns:
            Dict with frozenness metrics
        """
        # Filter to flight window
        mask = (bsa_series.index >= bsa_start) & (bsa_series.index <= bsa_end)
        flight_angles = bsa_series[mask].dropna()
        
        if len(flight_angles) < 3:
            return {
                'flight_std': np.nan,
                'flight_range': np.nan,
                'flight_mean_bsa': np.nan,
                'flight_trend': np.nan,
                'flight_jitter': np.nan
            }
        
        # Standard deviation (main frozenness metric)
        flight_std = flight_angles.std()
        
        # Range (max - min)
        flight_range = flight_angles.max() - flight_angles.min()
        
        # Mean BSA
        flight_mean = flight_angles.mean()
        
        # Trend (slope of linear fit - is athlete opening or closing?)
        frames = flight_angles.index.values
        angles = flight_angles.values
        
        try:
            slope, _ = np.polyfit(frames, angles, 1)
            # Convert to deg/frame then to deg/sec
            trend = slope * self.fps
        except:
            trend = np.nan
        
        # Jitter: std of frame-to-frame changes (high frequency instability)
        diffs = np.diff(angles)
        jitter = np.std(diffs) if len(diffs) > 1 else np.nan
        
        return {
            'flight_std': flight_std,
            'flight_range': flight_range,
            'flight_mean_bsa': flight_mean,
            'flight_trend': trend,
            'flight_jitter': jitter
        }
    
    # =========================================================================
    # METRIC 3: LANDING SMOOTHNESS (Absorption quality)
    # =========================================================================
    
    def compute_hip_height_series(self, jump_df: pd.DataFrame) -> pd.Series:
        """
        Compute hip height (y-coordinate of center_pelvis) for each frame.
        Note: In image coordinates, Y increases downward, so lower Y = higher position.
        
        Returns:
            Series with hip height indexed by frame_idx
        """
        heights = {}
        
        for _, row in jump_df.iterrows():
            frame_idx = row['frame_idx']
            pelvis = self.get_point(row, 'center_pelvis')
            
            if pelvis is not None:
                # Invert Y so that higher = larger value
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
        
        Args:
            hip_series: Series of hip heights
            knee_series: Series of knee angles
            landing_frame: Frame of landing
            window: Frames after landing to analyze
        
        Returns:
            Dict with smoothness metrics
        """
        start_frame = landing_frame
        end_frame = landing_frame + window
        
        # Hip descent analysis
        mask_hip = (hip_series.index >= start_frame) & (hip_series.index <= end_frame)
        landing_hip = hip_series[mask_hip].dropna()
        
        # Knee compression analysis
        mask_knee = (knee_series.index >= start_frame) & (knee_series.index <= end_frame)
        landing_knee = knee_series[mask_knee].dropna()
        
        result = {
            'landing_hip_drop': np.nan,
            'landing_hip_velocity': np.nan,
            'landing_knee_compression': np.nan,
            'landing_smoothness_score': np.nan
        }
        
        if len(landing_hip) >= 3:
            # Total hip drop in window
            result['landing_hip_drop'] = landing_hip.iloc[0] - landing_hip.min()
            
            # Velocity of descent (derivative)
            hip_velocity = np.gradient(landing_hip.values) * self.fps
            result['landing_hip_velocity'] = np.abs(hip_velocity).mean()
        
        if len(landing_knee) >= 3:
            # Knee compression (how much knee bends after landing)
            result['landing_knee_compression'] = landing_knee.iloc[0] - landing_knee.min()
        
        # Smoothness score: combination metric
        # Lower velocity + more knee bend = smoother
        if not np.isnan(result['landing_hip_velocity']) and not np.isnan(result['landing_knee_compression']):
            # Normalize and combine (higher = smoother)
            # Knee compression is good, velocity is bad
            result['landing_smoothness_score'] = result['landing_knee_compression'] / (result['landing_hip_velocity'] + 1)
        
        return result
    
    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================
    
    def process_jump(self, jump_id: str, phase_row: pd.Series) -> Dict:
        """Process a single jump and compute all time-series metrics."""
        
        # Get keypoints for this jump
        jump_df = self.df_kpts[self.df_kpts['jump_id'] == jump_id].copy()
        
        if jump_df.empty:
            print(f"  ⚠️ No keypoints for {jump_id}")
            return None
        
        result = {'jump_id': jump_id}
        
        # --- METRIC 1: Knee Velocity (if take_off available) ---
        takeoff_frame = phase_row.get('take_off_frame')
        if pd.notna(takeoff_frame) and phase_row.get('take_off_measurable') == 1:
            takeoff_frame = int(takeoff_frame)
            
            # Try right knee, then left knee
            knee_r = self.compute_knee_angle_series(jump_df, 'r')
            knee_l = self.compute_knee_angle_series(jump_df, 'l')
            
            # Use whichever has more data around take-off
            knee_series = knee_r if knee_r.notna().sum() >= knee_l.notna().sum() else knee_l
            
            knee_metrics = self.compute_knee_velocity(knee_series, takeoff_frame)
            result.update(knee_metrics)
        else:
            result.update({
                'knee_peak_velocity': np.nan,
                'knee_mean_velocity': np.nan,
                'knee_angle_at_takeoff': np.nan,
                'knee_extension_range': np.nan
            })
        
        # --- METRIC 2: Flight Frozenness (if BSA window available) ---
        bsa_start = phase_row.get('bsa_start')
        bsa_end = phase_row.get('bsa_end')
        
        if pd.notna(bsa_start) and pd.notna(bsa_end):
            bsa_start = int(bsa_start)
            bsa_end = int(bsa_end)
            
            bsa_series = self.compute_body_ski_angle_series(jump_df)
            
            # Extended window for full flight analysis
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
                'flight_range': np.nan,
                'flight_mean_bsa': np.nan,
                'flight_trend': np.nan,
                'flight_jitter': np.nan
            })
        
        # --- METRIC 3: Landing Smoothness (if landing available) ---
        landing_frame = phase_row.get('landing')
        
        if pd.notna(landing_frame):
            landing_frame = int(landing_frame)
            
            hip_series = self.compute_hip_height_series(jump_df)
            knee_series = self.compute_knee_angle_series(jump_df, 'r')  # Use right knee
            
            landing_metrics = self.compute_landing_smoothness(hip_series, knee_series, landing_frame)
            result.update(landing_metrics)
        else:
            result.update({
                'landing_hip_drop': np.nan,
                'landing_hip_velocity': np.nan,
                'landing_knee_compression': np.nan,
                'landing_smoothness_score': np.nan
            })
        
        return result
    
    def process_all(self) -> bool:
        """Process all jumps and save results."""
        
        if not self.load_data():
            return False
        
        print("\n" + "="*60)
        print("COMPUTING TIME-SERIES METRICS")
        print("="*60)
        
        results = []
        per_frame_results = []
        
        for _, phase_row in self.df_phases.iterrows():
            jump_id = phase_row['jump_id']
            
            if pd.isna(jump_id) or jump_id == '':
                continue
            
            jump_id = self._normalize_jid(jump_id)
            print(f"\n📊 Processing {jump_id}...")
            
            # Get summary metrics
            jump_result = self.process_jump(jump_id, phase_row)
            
            if jump_result:
                results.append(jump_result)
                
                # Also save per-frame data for time-series visualization
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
        
        # Save summary
        df_summary = pd.DataFrame(results)
        df_summary.to_csv(self.output_summary, index=False)
        print(f"\n✅ Summary saved: {self.output_summary}")
        print(f"   Jumps processed: {len(df_summary)}")
        
        # Save per-frame
        df_per_frame = pd.DataFrame(per_frame_results)
        df_per_frame.to_csv(self.output_per_frame, index=False)
        print(f"✅ Per-frame saved: {self.output_per_frame}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        metrics_to_show = ['knee_peak_velocity', 'flight_std', 'flight_jitter', 'landing_smoothness_score']
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
