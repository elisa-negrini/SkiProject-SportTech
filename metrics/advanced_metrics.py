"""
Advanced Metrics for Ski Jumping Analysis (Cleaned Version)
============================================================

This module computes advanced biomechanical metrics that complement the core metrics.
Metrics are designed to be robust to perspective distortion from diagonal camera views.

METRICS IMPLEMENTED:
--------------------
1. Takeoff Dynamics: Peak velocity, acceleration, smoothness at take-off
2. Takeoff Timing: Timing precision relative to ramp edge
3. Telemark Quality: Scissor distance, stability, absorption rate

REMOVED METRICS (unreliable with diagonal camera):
--------------------------------------------------
- Ski Jitter: Too sensitive to perspective
- Body Rotation: False positives from camera movement
- Arm Stability: Unreliable keypoints for hands
- V-Style Dynamics: Redundant with core V-style
- Body Compactness: Ambiguous definition

Output:
- metrics/advanced_metrics/advanced_metrics_summary.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class AdvancedMetricsCalculator:
    """
    Calculates advanced biomechanical metrics from keypoint data.
    
    Focus on:
    - Takeoff dynamics (velocities, accelerations)
    - Landing quality (telemark metrics)
    
    All metrics designed to be robust to diagonal camera perspective.
    """
    
    def __init__(self):
        """Initialize the calculator with paths and constants."""
        
        # Path configuration
        self.base_path = Path(__file__).parent.parent
        self.keypoints_file = self.base_path / 'keypoints_dataset.csv'
        self.phases_file = self.base_path / 'jump_phases_SkiTB.csv'
        
        # Output files (in subfolder)
        self.output_dir = self.base_path / 'metrics' / 'advanced_metrics'
        self.output_summary = self.output_dir / 'advanced_metrics_summary.csv'
        
        # Video constants
        self.fps = 30  # Frame rate
        self.dt = 1.0 / self.fps
        
        # Keypoint mapping
        self.kpt_map = {
            'head': '1', 'neck': '2', 'center_pelvis': '9',
            'r_shoulder': '3', 'l_shoulder': '6',
            'r_hip': '17', 'l_hip': '10',
            'r_knee': '18', 'l_knee': '11',
            'r_ankle': '19', 'l_ankle': '12',
        }
        
        # Signal processing parameters
        self.smooth_window = 5
        self.smooth_poly = 2
        
        # Validity ranges
        self.validity_ranges = {
            'knee_velocity': (50.0, 800.0),      # deg/sec
            'telemark_scissor': (0.0, 0.15),     # normalized units
            'landing_absorption': (-2.0, 2.0),   # normalized units/sec
        }
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def apply_validity_range(self, value: float, metric_type: str) -> float:
        """Apply validity range filter. Returns NaN if outside range."""
        if pd.isna(value):
            return np.nan
        if metric_type not in self.validity_ranges:
            return value
        min_val, max_val = self.validity_ranges[metric_type]
        return value if min_val <= value <= max_val else np.nan
    
    def load_data(self) -> bool:
        """Load all required data files."""
        if not self.keypoints_file.exists():
            print(f"❌ Keypoints file not found: {self.keypoints_file}")
            return False
        if not self.phases_file.exists():
            print(f"❌ Phases file not found: {self.phases_file}")
            return False
        
        self.df_kpts = pd.read_csv(self.keypoints_file)
        self.df_phases = pd.read_csv(self.phases_file)
        
        # Normalize jump IDs
        self.df_phases['jump_id'] = self.df_phases['jump_id'].apply(self._normalize_jid)
        self.df_kpts['jump_id'] = self.df_kpts['jump_id'].apply(self._normalize_jid)
        
        # Parse frame numbers
        self.df_kpts['frame_idx'] = self.df_kpts['frame_name'].apply(self._extract_frame_num)
        
        # Remove duplicates
        self.df_kpts = self.df_kpts.drop_duplicates(
            subset=['jump_id', 'frame_idx'], keep='first'
        )
        
        print(f"✅ Loaded: {len(self.df_kpts)} keypoint frames, {len(self.df_phases)} jump phases")
        return True
    
    def _normalize_jid(self, val) -> str:
        """Convert jump ID to standard format (JP####)."""
        s = str(val).strip()
        if s.lower().startswith('jump'):
            num = int(s[4:])
            return f"JP{num:04d}"
        return s
    
    def _extract_frame_num(self, fname: str) -> int:
        """Extract frame number from filename."""
        import re
        match = re.match(r'^(\d+)', str(fname))
        return int(match.group(1)) if match else -1
    
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
    
    def calculate_angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle in degrees between two 2D vectors."""
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return np.nan
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    # =========================================================================
    # METRIC 1: TAKEOFF DYNAMICS
    # =========================================================================
    
    def compute_takeoff_dynamics(self, jump_df: pd.DataFrame,
                                  takeoff_frame: int,
                                  window: int = 10) -> Dict:
        """
        Analyze take-off timing and explosiveness.
        
        Computes:
        - Peak extension velocity (how fast the leg extends)
        - Peak acceleration
        - Timing offset (when peak velocity occurs vs annotated takeoff)
        - Smoothness (consistency of acceleration)
        """
        knee_angles = {}
        
        start_f = takeoff_frame - window
        end_f = takeoff_frame + window
        
        for _, row in jump_df.iterrows():
            frame_idx = row['frame_idx']
            
            if frame_idx < start_f or frame_idx > end_f:
                continue
            
            # Try right knee, then left knee
            for side in ['r', 'l']:
                hip = self.get_point(row, f'{side}_hip')
                knee = self.get_point(row, f'{side}_knee')
                ankle = self.get_point(row, f'{side}_ankle')
                
                if hip is not None and knee is not None and ankle is not None:
                    v1 = hip - knee
                    v2 = ankle - knee
                    angle = self.calculate_angle_between_vectors(v1, v2)
                    
                    if not np.isnan(angle):
                        knee_angles[frame_idx] = angle
                        break
        
        if len(knee_angles) < 5:
            return {
                'takeoff_timing_offset': np.nan,
                'takeoff_peak_velocity': np.nan,
                'takeoff_acceleration_peak': np.nan,
                'takeoff_smoothness': np.nan
            }
        
        # Convert to sorted series
        angles_series = pd.Series(knee_angles).sort_index()
        angles_array = angles_series.values
        frames_array = angles_series.index.values
        
        # Smooth and compute derivatives
        if len(angles_array) >= self.smooth_window:
            try:
                smoothed = savgol_filter(angles_array, self.smooth_window, self.smooth_poly)
            except:
                smoothed = angles_array
        else:
            smoothed = angles_array
        
        # Velocity (first derivative) and acceleration (second derivative)
        velocity = np.gradient(smoothed)
        acceleration = np.gradient(velocity)
        
        # Peak extension velocity (positive = opening)
        peak_idx = np.argmax(velocity)
        peak_frame = frames_array[peak_idx]
        
        # Timing offset: positive = late, negative = early
        timing_offset = peak_frame - takeoff_frame
        
        # Peak velocity in degrees/second with validity check
        peak_velocity = velocity[peak_idx] * self.fps
        peak_velocity = self.apply_validity_range(peak_velocity, 'knee_velocity')
        
        # Peak acceleration
        acceleration_peak = np.max(np.abs(acceleration)) * self.fps * self.fps
        
        # Smoothness: inverse of acceleration variability
        smoothness = 1.0 / (np.std(acceleration) + 0.001)
        
        return {
            'takeoff_timing_offset': timing_offset,
            'takeoff_peak_velocity': peak_velocity,
            'takeoff_acceleration_peak': acceleration_peak,
            'takeoff_smoothness': smoothness
        }
    
    # =========================================================================
    # METRIC 2: TELEMARK QUALITY
    # =========================================================================
    
    def compute_telemark_quality(self, jump_df: pd.DataFrame,
                                  landing_frame: int,
                                  window: int = 15) -> Dict:
        """
        Compute landing (Telemark) quality metrics.
        
        Computes:
        - Scissor distance (normalized by hip height)
        - Stability (how constant the scissor is maintained)
        - Absorption rate (how fast pelvis descends - smooth is good)
        """
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
                # Vertical distance between ankles (telemark scissor)
                ankle_y_diff = abs(r_ankle[1] - l_ankle[1])
                
                # Normalize by hip height if available
                hip_height = None
                if r_hip is not None and l_hip is not None:
                    avg_hip_y = (r_hip[1] + l_hip[1]) / 2
                    avg_ankle_y = (r_ankle[1] + l_ankle[1]) / 2
                    hip_height = abs(avg_hip_y - avg_ankle_y)
                
                # Normalized scissor
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
        
        # Mean scissor distance (main Telemark metric)
        scissor_mean = df_tele['scissor'].mean()
        scissor_mean = self.apply_validity_range(scissor_mean, 'telemark_scissor')
        
        # Stability: inverse of std (low variation = stable)
        scissor_std = df_tele['scissor'].std()
        stability = 1.0 / (scissor_std + 0.001) if scissor_std < 1 else 0.0
        
        # Absorption rate: pelvis descent rate
        pelvis_y = df_tele['pelvis_y'].dropna()
        if len(pelvis_y) >= 3:
            # Pelvis descent rate (positive Y = lower in image)
            absorption_rate = (pelvis_y.iloc[-1] - pelvis_y.iloc[0]) / len(pelvis_y) * self.fps
            absorption_rate = self.apply_validity_range(absorption_rate, 'landing_absorption')
        else:
            absorption_rate = np.nan
        
        return {
            'telemark_scissor_mean': scissor_mean,
            'telemark_stability': stability,
            'landing_absorption_rate': absorption_rate
        }
    
    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================
    
    def process_jump(self, jump_id: str, phase_row: pd.Series) -> Optional[Dict]:
        """Process a single jump and compute all advanced metrics."""
        
        jump_df = self.df_kpts[self.df_kpts['jump_id'] == jump_id].copy()
        
        if jump_df.empty:
            print(f"  ⚠️ No keypoints for {jump_id}")
            return None
        
        result = {'jump_id': jump_id}
        
        # Parse phase information
        takeoff_frame = phase_row.get('take_off_frame')
        has_takeoff = pd.notna(takeoff_frame) and phase_row.get('take_off_measurable') == 1
        if has_takeoff:
            takeoff_frame = int(takeoff_frame)
        
        landing_frame = phase_row.get('landing')
        has_landing = pd.notna(landing_frame)
        if has_landing:
            landing_frame = int(landing_frame)
        
        # Compute metrics
        
        # Takeoff Dynamics
        if has_takeoff:
            takeoff_metrics = self.compute_takeoff_dynamics(jump_df, takeoff_frame)
        else:
            takeoff_metrics = {k: np.nan for k in [
                'takeoff_timing_offset', 'takeoff_peak_velocity',
                'takeoff_acceleration_peak', 'takeoff_smoothness'
            ]}
        result.update(takeoff_metrics)
        
        # Telemark Quality
        if has_landing:
            telemark_metrics = self.compute_telemark_quality(jump_df, landing_frame)
        else:
            telemark_metrics = {k: np.nan for k in [
                'telemark_scissor_mean', 'telemark_stability', 'landing_absorption_rate'
            ]}
        result.update(telemark_metrics)
        
        return result
    
    def process_all(self) -> bool:
        """Process all jumps and save results."""
        
        if not self.load_data():
            return False
        
        print("\n" + "=" * 70)
        print("COMPUTING ADVANCED METRICS (Cleaned Version)")
        print("=" * 70)
        
        results = []
        
        for _, phase_row in self.df_phases.iterrows():
            jump_id = phase_row['jump_id']
            
            if pd.isna(jump_id) or jump_id == '':
                continue
            
            jump_id = self._normalize_jid(jump_id)
            print(f"  📊 Processing {jump_id}...")
            
            jump_result = self.process_jump(jump_id, phase_row)
            
            if jump_result:
                results.append(jump_result)
        
        # Save results
        df_summary = pd.DataFrame(results)
        df_summary.to_csv(self.output_summary, index=False)
        
        print("\n" + "=" * 70)
        print(f"✅ Advanced metrics saved: {self.output_summary}")
        print(f"   Jumps processed: {len(df_summary)}")
        
        # Print summary statistics
        print("\n📈 SUMMARY STATISTICS")
        print("-" * 50)
        
        for col in ['takeoff_peak_velocity', 'takeoff_timing_offset', 
                    'telemark_scissor_mean', 'telemark_stability']:
            if col in df_summary.columns:
                valid = df_summary[col].dropna()
                if len(valid) > 0:
                    print(f"  {col:30s}: {valid.mean():8.3f} ± {valid.std():.3f} (n={len(valid)})")
        
        print("=" * 70)
        return True


if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED METRICS CALCULATOR (Cleaned Version)")
    print("Ski Jumping Biomechanical Analysis")
    print("=" * 70)
    
    calculator = AdvancedMetricsCalculator()
    success = calculator.process_all()
    
    if success:
        print("\n✅ Processing complete!")
    else:
        print("\n❌ Processing failed!")
