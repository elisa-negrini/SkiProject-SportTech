"""
Phase-by-Phase Analysis for Ski Jumping
========================================

This module performs detailed analysis of each jump phase separately,
allowing identification of phase-specific performance factors.

JUMP PHASES ANALYZED:
---------------------
1. In-Run: Approach before take-off
2. Take-Off: The explosive jump moment
3. Early Flight: Initial flight stabilization
4. Mid Flight: Stable flight (V-style position)
5. Late Flight: Preparation for landing
6. Landing: Telemark and absorption

ANALYSIS PER PHASE:
-------------------
- Phase duration (frames and time)
- Key metrics specific to each phase
- Transition quality between phases
- Phase-score correlations

ATHLETE FINGERPRINTING:
-----------------------
Creates a biomechanical profile for each athlete showing their
strengths and weaknesses across different phases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


@dataclass
class JumpPhase:
    """
    Represents a single phase of a ski jump.
    
    Attributes:
        name: Phase identifier
        start_frame: First frame of phase
        end_frame: Last frame of phase
        duration_frames: Number of frames
        duration_seconds: Duration in seconds
    """
    name: str
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float


class PhaseAnalyzer:
    """
    Analyzes ski jumps on a phase-by-phase basis.
    
    This class provides:
    - Automatic phase segmentation from annotations
    - Phase-specific metric computation
    - Cross-phase transition analysis
    - Athlete profiling
    """
    
    def __init__(self):
        """Initialize the analyzer with paths and configurations."""
        
        # =====================================================================
        # PATH CONFIGURATION
        # =====================================================================
        self.base_path = Path(__file__).parent.parent
        self.keypoints_file = self.base_path / 'keypoints_dataset.csv'
        self.phases_file = self.base_path / 'jump_phases_SkiTB.csv'
        self.jp_data_file = self.base_path / 'JP_data.csv'
        
        # Output
        self.output_dir = self.base_path / 'metrics' / 'phase_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # =====================================================================
        # CONSTANTS
        # =====================================================================
        self.fps = 30  # Frame rate
        
        # Phase colors for visualization
        self.phase_colors = {
            'in_run': '#3498db',      # Blue
            'take_off': '#e74c3c',    # Red
            'early_flight': '#f39c12', # Orange
            'mid_flight': '#2ecc71',   # Green
            'late_flight': '#9b59b6',  # Purple
            'landing': '#1abc9c'       # Teal
        }
        
        # Keypoint mapping
        self.kpt_map = {
            'head': '1', 'neck': '2', 'center_pelvis': '9',
            'r_shoulder': '3', 'l_shoulder': '6',
            'r_hip': '17', 'l_hip': '10',
            'r_knee': '18', 'l_knee': '11',
            'r_ankle': '19', 'l_ankle': '12',
            'r_ski_tip': '23', 'r_ski_tail': '22',
            'l_ski_tip': '16', 'l_ski_tail': '15'
        }
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_data(self) -> bool:
        """
        Load all required data files.
        
        Returns:
            bool: True if successful
        """
        # Load keypoints
        if not self.keypoints_file.exists():
            print(f"❌ Keypoints not found: {self.keypoints_file}")
            return False
        
        self.df_kpts = pd.read_csv(self.keypoints_file)
        self.df_kpts['jump_id'] = self.df_kpts['jump_id'].apply(self._normalize_jid)
        self.df_kpts['frame_idx'] = self.df_kpts['frame_name'].apply(self._extract_frame_num)
        self.df_kpts = self.df_kpts.drop_duplicates(subset=['jump_id', 'frame_idx'], keep='first')
        
        # Load phases
        if not self.phases_file.exists():
            print(f"❌ Phases not found: {self.phases_file}")
            return False
        
        self.df_phases = pd.read_csv(self.phases_file)
        self.df_phases['jump_id'] = self.df_phases['jump_id'].apply(self._normalize_jid)
        
        # Load JP data for scores
        if self.jp_data_file.exists():
            self.df_jp = pd.read_csv(self.jp_data_file)
            self._compute_scores()
        else:
            self.df_jp = None
            self.df_scores = None
        
        print(f"✅ Loaded {len(self.df_kpts)} keypoints, {len(self.df_phases)} phase definitions")
        return True
    
    def _normalize_jid(self, val) -> str:
        """Normalize jump ID to JP#### format."""
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
    
    def _compute_scores(self):
        """Compute Physical_Score and Style_Score."""
        scores = []
        for _, row in self.df_jp.iterrows():
            judges = [row.get(f'AthleteJdg{x}', np.nan) for x in 'ABCDE']
            valid = [s for s in judges if pd.notna(s)]
            
            if len(valid) >= 5:
                style = sum(sorted(valid)[1:4])
            elif len(valid) >= 3:
                style = sum(sorted(valid)[:3])
            else:
                style = np.nan
            
            athlete_score = row.get('AthleteScore', np.nan)
            physical = athlete_score - style if pd.notna(athlete_score) and pd.notna(style) else np.nan
            
            scores.append({
                'jump_id': row['ID'],
                'Style_Score': style,
                'Physical_Score': physical,
                'AthleteDistance': row.get('AthleteDistance', np.nan),
                'AthleteName': f"{row.get('AthleteName', '')} {row.get('AthleteSurname', '')}"
            })
        
        self.df_scores = pd.DataFrame(scores)
    
    # =========================================================================
    # KEYPOINT UTILITIES
    # =========================================================================
    
    def get_point(self, row, name: str) -> Optional[np.ndarray]:
        """Extract (x, y) coordinates of a keypoint."""
        try:
            kpt_id = self.kpt_map[name]
            x = row[f'kpt_{kpt_id}_x']
            y = row[f'kpt_{kpt_id}_y']
            if pd.isna(x) or pd.isna(y):
                return None
            return np.array([x, y])
        except KeyError:
            return None
    
    # =========================================================================
    # PHASE SEGMENTATION
    # =========================================================================
    
    def segment_jump_phases(self, phase_row: pd.Series) -> Dict[str, JumpPhase]:
        """
        Segment a jump into distinct phases based on annotations.
        
        PHASE DEFINITIONS:
        ------------------
        - in_run: From first keypoint to take_off_frame
        - take_off: Window around take_off_frame (±5 frames)
        - early_flight: From take_off to bsa_start
        - mid_flight: From bsa_start to bsa_end (stable V-style)
        - late_flight: From bsa_end to landing
        - landing: From landing to telemark_end
        
        Args:
            phase_row: Row from phases dataframe
            
        Returns:
            Dict mapping phase name to JumpPhase object
        """
        phases = {}
        
        # Extract key frames
        takeoff = phase_row.get('take_off_frame')
        bsa_start = phase_row.get('bsa_start')
        bsa_end = phase_row.get('bsa_end')
        landing = phase_row.get('landing')
        tele_start = phase_row.get('telemark_start')
        tele_end = phase_row.get('telemark_end')
        
        # Convert to int where valid
        def to_int(val):
            return int(val) if pd.notna(val) else None
        
        takeoff = to_int(takeoff)
        bsa_start = to_int(bsa_start)
        bsa_end = to_int(bsa_end)
        landing = to_int(landing)
        tele_start = to_int(tele_start)
        tele_end = to_int(tele_end)
        
        # Define phases based on available data
        
        # TAKE-OFF PHASE (window around take-off)
        if takeoff:
            phases['take_off'] = JumpPhase(
                name='take_off',
                start_frame=max(takeoff - 10, 0),
                end_frame=takeoff + 5,
                duration_frames=15,
                duration_seconds=15 / self.fps
            )
        
        # EARLY FLIGHT (take-off to BSA start)
        if takeoff and bsa_start:
            phases['early_flight'] = JumpPhase(
                name='early_flight',
                start_frame=takeoff + 5,
                end_frame=bsa_start,
                duration_frames=bsa_start - takeoff - 5,
                duration_seconds=(bsa_start - takeoff - 5) / self.fps
            )
        
        # MID FLIGHT (BSA window - stable V-style)
        if bsa_start and bsa_end:
            phases['mid_flight'] = JumpPhase(
                name='mid_flight',
                start_frame=bsa_start,
                end_frame=bsa_end,
                duration_frames=bsa_end - bsa_start,
                duration_seconds=(bsa_end - bsa_start) / self.fps
            )
        
        # LATE FLIGHT (BSA end to landing)
        if bsa_end and landing:
            phases['late_flight'] = JumpPhase(
                name='late_flight',
                start_frame=bsa_end,
                end_frame=landing,
                duration_frames=landing - bsa_end,
                duration_seconds=(landing - bsa_end) / self.fps
            )
        
        # LANDING (landing to telemark end)
        if landing and tele_end:
            phases['landing'] = JumpPhase(
                name='landing',
                start_frame=landing,
                end_frame=tele_end,
                duration_frames=tele_end - landing,
                duration_seconds=(tele_end - landing) / self.fps
            )
        elif landing:
            # No telemark annotation, use landing + window
            phases['landing'] = JumpPhase(
                name='landing',
                start_frame=landing,
                end_frame=landing + 15,
                duration_frames=15,
                duration_seconds=15 / self.fps
            )
        
        return phases
    
    # =========================================================================
    # PHASE-SPECIFIC METRICS
    # =========================================================================
    
    def compute_phase_metrics(self, jump_df: pd.DataFrame, 
                               phase: JumpPhase) -> Dict[str, float]:
        """
        Compute metrics specific to a phase.
        
        Different phases have different relevant metrics:
        - take_off: Knee velocity, hip acceleration
        - early_flight: V-style opening rate
        - mid_flight: Stability (std of angles)
        - late_flight: Position maintenance
        - landing: Absorption quality
        
        Args:
            jump_df: Keypoint data for this jump
            phase: Phase definition
            
        Returns:
            Dict with phase-specific metrics
        """
        # Filter to phase frames
        mask = (jump_df['frame_idx'] >= phase.start_frame) & (jump_df['frame_idx'] <= phase.end_frame)
        phase_df = jump_df[mask].sort_values('frame_idx')
        
        if len(phase_df) < 2:
            return {}
        
        metrics = {
            'n_frames': len(phase_df),
            'duration_sec': phase.duration_seconds
        }
        
        # =====================================================================
        # COMMON METRICS (all phases)
        # =====================================================================
        
        # Hip height trajectory
        hip_heights = []
        for _, row in phase_df.iterrows():
            pelvis = self.get_point(row, 'center_pelvis')
            if pelvis is not None:
                hip_heights.append(1.0 - pelvis[1])  # Invert Y
        
        if hip_heights:
            metrics['hip_height_mean'] = np.mean(hip_heights)
            metrics['hip_height_std'] = np.std(hip_heights)
        
        # =====================================================================
        # PHASE-SPECIFIC METRICS
        # =====================================================================
        
        if phase.name == 'take_off':
            metrics.update(self._compute_takeoff_metrics(phase_df))
        
        elif phase.name == 'early_flight':
            metrics.update(self._compute_early_flight_metrics(phase_df))
        
        elif phase.name == 'mid_flight':
            metrics.update(self._compute_mid_flight_metrics(phase_df))
        
        elif phase.name == 'late_flight':
            metrics.update(self._compute_late_flight_metrics(phase_df))
        
        elif phase.name == 'landing':
            metrics.update(self._compute_landing_metrics(phase_df))
        
        return metrics
    
    def _compute_takeoff_metrics(self, phase_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute take-off specific metrics.
        
        Focus on:
        - Knee extension velocity
        - Hip acceleration
        - Body angle at take-off
        """
        metrics = {}
        
        knee_angles = []
        for _, row in phase_df.iterrows():
            for side in ['r', 'l']:
                hip = self.get_point(row, f'{side}_hip')
                knee = self.get_point(row, f'{side}_knee')
                ankle = self.get_point(row, f'{side}_ankle')
                
                if hip is not None and knee is not None and ankle is not None:
                    v1 = hip - knee
                    v2 = ankle - knee
                    dot = np.dot(v1, v2)
                    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                    if norms > 0:
                        angle = np.degrees(np.arccos(np.clip(dot / norms, -1, 1)))
                        knee_angles.append(angle)
                        break
        
        if len(knee_angles) >= 3:
            velocity = np.diff(knee_angles) * self.fps
            metrics['takeoff_knee_velocity_max'] = np.max(velocity)
            metrics['takeoff_knee_velocity_mean'] = np.mean(velocity[velocity > 0]) if np.any(velocity > 0) else 0
            metrics['takeoff_knee_angle_final'] = knee_angles[-1]
        
        return metrics
    
    def _compute_early_flight_metrics(self, phase_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute early flight metrics.
        
        Focus on:
        - V-style opening rate
        - Initial stabilization
        """
        metrics = {}
        
        ski_angles = []
        for _, row in phase_df.iterrows():
            r_tip = self.get_point(row, 'r_ski_tip')
            r_tail = self.get_point(row, 'r_ski_tail')
            l_tip = self.get_point(row, 'l_ski_tip')
            l_tail = self.get_point(row, 'l_ski_tail')
            
            if all(p is not None for p in [r_tip, r_tail, l_tip, l_tail]):
                r_vec = r_tip - r_tail
                l_vec = l_tip - l_tail
                
                dot = np.dot(r_vec, l_vec)
                norms = np.linalg.norm(r_vec) * np.linalg.norm(l_vec)
                if norms > 0:
                    angle = np.degrees(np.arccos(np.clip(dot / norms, -1, 1)))
                    ski_angles.append(angle)
        
        if len(ski_angles) >= 3:
            opening_rate = np.diff(ski_angles) * self.fps
            metrics['early_vstyle_opening_rate'] = np.mean(opening_rate[opening_rate > 0]) if np.any(opening_rate > 0) else 0
            metrics['early_vstyle_final'] = ski_angles[-1]
        
        return metrics
    
    def _compute_mid_flight_metrics(self, phase_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute mid-flight (stable V-style) metrics.
        
        Focus on:
        - Body-ski angle consistency
        - Stability (low std)
        """
        metrics = {}
        
        bsa_angles = []
        for _, row in phase_df.iterrows():
            neck = self.get_point(row, 'neck')
            pelvis = self.get_point(row, 'center_pelvis')
            r_tip = self.get_point(row, 'r_ski_tip')
            r_tail = self.get_point(row, 'r_ski_tail')
            
            if all(p is not None for p in [neck, pelvis, r_tip, r_tail]):
                body_vec = neck - pelvis
                ski_vec = r_tip - r_tail
                
                dot = np.dot(body_vec, ski_vec)
                norms = np.linalg.norm(body_vec) * np.linalg.norm(ski_vec)
                if norms > 0:
                    angle = np.degrees(np.arccos(np.clip(dot / norms, -1, 1)))
                    bsa_angles.append(angle)
        
        if len(bsa_angles) >= 3:
            metrics['mid_bsa_mean'] = np.mean(bsa_angles)
            metrics['mid_bsa_std'] = np.std(bsa_angles)  # Lower = more stable
            metrics['mid_bsa_range'] = np.max(bsa_angles) - np.min(bsa_angles)
        
        return metrics
    
    def _compute_late_flight_metrics(self, phase_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute late flight metrics.
        
        Focus on:
        - Position maintenance before landing
        - Preparation quality
        """
        metrics = {}
        
        # Use same BSA computation as mid-flight
        # Check if angle is drifting (trend)
        bsa_angles = []
        frames = []
        
        for _, row in phase_df.iterrows():
            neck = self.get_point(row, 'neck')
            pelvis = self.get_point(row, 'center_pelvis')
            r_tip = self.get_point(row, 'r_ski_tip')
            r_tail = self.get_point(row, 'r_ski_tail')
            
            if all(p is not None for p in [neck, pelvis, r_tip, r_tail]):
                body_vec = neck - pelvis
                ski_vec = r_tip - r_tail
                
                dot = np.dot(body_vec, ski_vec)
                norms = np.linalg.norm(body_vec) * np.linalg.norm(ski_vec)
                if norms > 0:
                    angle = np.degrees(np.arccos(np.clip(dot / norms, -1, 1)))
                    bsa_angles.append(angle)
                    frames.append(row['frame_idx'])
        
        if len(bsa_angles) >= 3:
            metrics['late_bsa_std'] = np.std(bsa_angles)
            
            # Trend (positive = opening, negative = closing)
            try:
                slope, _ = np.polyfit(frames, bsa_angles, 1)
                metrics['late_bsa_trend'] = slope * self.fps
            except:
                pass
        
        return metrics
    
    def _compute_landing_metrics(self, phase_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute landing metrics.
        
        Focus on:
        - Telemark quality
        - Absorption (hip descent)
        """
        metrics = {}
        
        # Telemark scissor distance
        ankle_diffs = []
        hip_heights = []
        
        for _, row in phase_df.iterrows():
            r_ankle = self.get_point(row, 'r_ankle')
            l_ankle = self.get_point(row, 'l_ankle')
            pelvis = self.get_point(row, 'center_pelvis')
            
            if r_ankle is not None and l_ankle is not None:
                ankle_diffs.append(abs(r_ankle[1] - l_ankle[1]))
            
            if pelvis is not None:
                hip_heights.append(pelvis[1])
        
        if ankle_diffs:
            metrics['landing_scissor_mean'] = np.mean(ankle_diffs)
            metrics['landing_scissor_max'] = np.max(ankle_diffs)
        
        if len(hip_heights) >= 3:
            # Absorption rate (how fast hip descends)
            metrics['landing_absorption'] = (hip_heights[-1] - hip_heights[0]) * self.fps / len(hip_heights)
        
        return metrics
    
    # =========================================================================
    # FULL JUMP ANALYSIS
    # =========================================================================
    
    def analyze_jump(self, jump_id: str) -> Optional[Dict]:
        """
        Perform complete phase-by-phase analysis of a single jump.
        
        Args:
            jump_id: Jump identifier
            
        Returns:
            Dict with all phase metrics and overall summary
        """
        # Get phase row
        phase_row = self.df_phases[self.df_phases['jump_id'] == jump_id]
        if phase_row.empty:
            return None
        phase_row = phase_row.iloc[0]
        
        # Get keypoints
        jump_df = self.df_kpts[self.df_kpts['jump_id'] == jump_id]
        if jump_df.empty:
            return None
        
        # Segment into phases
        phases = self.segment_jump_phases(phase_row)
        
        if not phases:
            return None
        
        # Compute metrics for each phase
        result = {'jump_id': jump_id}
        
        for phase_name, phase in phases.items():
            phase_metrics = self.compute_phase_metrics(jump_df, phase)
            
            # Prefix metrics with phase name
            for metric_name, value in phase_metrics.items():
                result[f'{phase_name}_{metric_name}'] = value
        
        return result
    
    # =========================================================================
    # ATHLETE PROFILING
    # =========================================================================
    
    def create_athlete_profile(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create athlete profiles based on phase performance.
        
        This identifies:
        - Phase-specific strengths (above average)
        - Phase-specific weaknesses (below average)
        - Overall technique fingerprint
        
        Args:
            results_df: DataFrame with all jump analyses
            
        Returns:
            DataFrame with athlete profiles
        """
        if self.df_scores is None:
            print("⚠️ Score data not available for profiling")
            return pd.DataFrame()
        
        # Merge with athlete names
        df = results_df.merge(self.df_scores[['jump_id', 'AthleteName']], on='jump_id', how='left')
        
        # Group by athlete and compute means
        # Select only numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        profile_df = df.groupby('AthleteName')[numeric_cols].mean()
        
        # Compute z-scores relative to all athletes
        zscore_df = (profile_df - profile_df.mean()) / profile_df.std()
        
        return zscore_df
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def plot_phase_timeline(self, jump_id: str, save_path: Optional[Path] = None):
        """
        Plot visual timeline of jump phases.
        
        Args:
            jump_id: Jump to visualize
            save_path: Path to save figure
        """
        phase_row = self.df_phases[self.df_phases['jump_id'] == jump_id]
        if phase_row.empty:
            print(f"⚠️ No phase data for {jump_id}")
            return
        
        phases = self.segment_jump_phases(phase_row.iloc[0])
        
        if not phases:
            return
        
        fig, ax = plt.subplots(figsize=(14, 4))
        
        # Find overall range
        all_starts = [p.start_frame for p in phases.values()]
        all_ends = [p.end_frame for p in phases.values()]
        min_frame = min(all_starts) - 20
        max_frame = max(all_ends) + 20
        
        # Plot each phase as a colored bar
        y_pos = 0.5
        for phase_name, phase in phases.items():
            color = self.phase_colors.get(phase_name, 'gray')
            width = phase.end_frame - phase.start_frame
            
            ax.barh(y_pos, width, left=phase.start_frame, height=0.3, 
                    color=color, alpha=0.8, label=phase_name)
            
            # Add phase name label
            mid_x = phase.start_frame + width / 2
            ax.text(mid_x, y_pos, phase_name.replace('_', '\n'), 
                    ha='center', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlim(min_frame, max_frame)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Frame Number')
        ax.set_title(f'Jump Phases Timeline: {jump_id}')
        ax.set_yticks([])
        
        # Legend
        patches = [Patch(color=self.phase_colors[p], label=p.replace('_', ' ').title()) 
                   for p in phases.keys()]
        ax.legend(handles=patches, loc='upper right', ncol=3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Timeline saved: {save_path}")
        
        plt.close()
    
    def plot_phase_comparison(self, results_df: pd.DataFrame, 
                               metric_prefix: str,
                               save_path: Optional[Path] = None):
        """
        Compare a metric across different phases.
        
        Args:
            results_df: DataFrame with phase analysis results
            metric_prefix: Metric to compare (e.g., 'bsa_std')
            save_path: Path to save figure
        """
        # Find columns matching the metric
        matching_cols = [c for c in results_df.columns if metric_prefix in c]
        
        if len(matching_cols) < 2:
            print(f"⚠️ Not enough data for {metric_prefix}")
            return
        
        # Prepare data for plotting
        plot_data = []
        for col in matching_cols:
            # Extract phase name (handle multi-word phases like 'early_flight')
            parts = col.split('_')
            if len(parts) >= 2 and parts[0] in ['take', 'early', 'mid', 'late', 'landing']:
                if parts[0] == 'landing':
                    phase = 'landing'
                elif len(parts) >= 2:
                    phase = f"{parts[0]}_{parts[1]}" if parts[1] == 'off' or parts[1] == 'flight' else parts[0]
                else:
                    phase = parts[0]
            else:
                phase = parts[0]
            
            values = results_df[col].dropna()
            for v in values:
                plot_data.append({'phase': phase, 'value': v})
        
        if not plot_data:
            print(f"⚠️ No data for {metric_prefix}")
            return
        
        df_plot = pd.DataFrame(plot_data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        phases_order = ['take_off', 'early_flight', 'mid_flight', 'late_flight', 'landing']
        phases_present = [p for p in phases_order if p in df_plot['phase'].unique()]
        
        if not phases_present:
            print(f"⚠️ No valid phases for {metric_prefix}")
            plt.close()
            return
        
        # Use a simple approach without custom palette for compatibility
        try:
            sns.boxplot(data=df_plot, x='phase', y='value', order=phases_present, ax=ax)
        except Exception as e:
            print(f"⚠️ Boxplot failed for {metric_prefix}: {e}")
            plt.close()
            return
        
        ax.set_xlabel('Phase')
        ax.set_ylabel(metric_prefix.replace('_', ' ').title())
        ax.set_title(f'{metric_prefix.replace("_", " ").title()} Across Jump Phases')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Comparison plot saved: {save_path}")
        
        plt.close()
    
    def plot_athlete_radar(self, athlete_profile: pd.Series, 
                           athlete_name: str,
                           save_path: Optional[Path] = None):
        """
        Create radar/spider chart for athlete profile.
        
        Args:
            athlete_profile: Series with z-scored metrics
            athlete_name: Name for title
            save_path: Path to save figure
        """
        # Select key metrics for radar
        key_metrics = [
            'take_off_takeoff_knee_velocity_max',
            'mid_flight_mid_bsa_std',
            'late_flight_late_bsa_std',
            'landing_landing_scissor_mean'
        ]
        
        available_metrics = [m for m in key_metrics if m in athlete_profile.index]
        
        if len(available_metrics) < 3:
            print("⚠️ Not enough metrics for radar chart")
            return
        
        values = [athlete_profile[m] for m in available_metrics]
        labels = [m.replace('_', ' ')[:20] for m in available_metrics]
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
        values += values[:1]  # Close the polygon
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=9)
        ax.set_title(f'Athlete Profile: {athlete_name}', size=14, fontweight='bold', y=1.08)
        
        # Add reference circle at 0 (average)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Radar chart saved: {save_path}")
        
        plt.close()
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    def run_full_analysis(self):
        """
        Run complete phase-by-phase analysis.
        """
        if not self.load_data():
            return False
        
        print("\n" + "=" * 70)
        print("PHASE-BY-PHASE ANALYSIS")
        print("=" * 70)
        
        # Analyze all jumps
        results = []
        
        for jid in self.df_kpts['jump_id'].unique():
            print(f"📊 Analyzing {jid}...")
            
            result = self.analyze_jump(jid)
            if result:
                results.append(result)
        
        if not results:
            print("❌ No valid results")
            return False
        
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(self.output_dir / 'phase_metrics.csv', index=False)
        print(f"\n✅ Phase metrics saved: {self.output_dir / 'phase_metrics.csv'}")
        print(f"   Jumps analyzed: {len(results_df)}")
        
        # Generate visualizations
        print("\n📊 Generating visualizations...")
        
        # Plot timeline for first few jumps
        for jid in list(results_df['jump_id'])[:3]:
            self.plot_phase_timeline(jid, self.output_dir / f'timeline_{jid}.png')
        
        # Plot phase comparisons
        for metric in ['bsa_std', 'hip_height_std']:
            self.plot_phase_comparison(results_df, metric, 
                                        self.output_dir / f'comparison_{metric}.png')
        
        # Athlete profiles
        if self.df_scores is not None:
            profiles = self.create_athlete_profile(results_df)
            if not profiles.empty:
                profiles.to_csv(self.output_dir / 'athlete_profiles.csv')
                print(f"✅ Athlete profiles saved")
        
        print("\n" + "=" * 70)
        print("PHASE ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)
        
        return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE-BY-PHASE ANALYSIS FOR SKI JUMPING")
    print("=" * 70)
    
    analyzer = PhaseAnalyzer()
    analyzer.run_full_analysis()
