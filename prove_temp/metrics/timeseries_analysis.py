"""
Time-Series Analysis for Ski Jumping
=====================================

This module performs advanced time-series analysis on ski jumping data,
including pattern recognition, clustering, and temporal normalization.

ANALYSES IMPLEMENTED:
---------------------
1. Temporal Normalization: Align all jumps to standard duration
2. DTW Clustering: Group similar jump patterns using Dynamic Time Warping
3. Average Trajectory Analysis: Compare Top vs Flop performers
4. Frequency Analysis: Detect oscillation patterns
5. Phase Segmentation: Automatic detection of jump phases

THEORY:
-------
Time-series analysis allows us to compare jumps of different durations
and identify patterns that distinguish elite performers from others.
By normalizing time and clustering trajectories, we can find "prototypical"
jump patterns and identify deviations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')




class TimeSeriesAnalyzer:
    """
    Performs time-series analysis on ski jumping trajectories.
    
    This class provides tools for:
    - Normalizing jump durations to enable comparison
    - Clustering similar jump patterns
    - Identifying characteristic trajectories of top performers
    - Detecting oscillation frequencies and anomalies
    """
    
    def __init__(self):
        """Initialize the analyzer with paths and parameters."""
        
        # =====================================================================
        # PATH CONFIGURATION
        # =====================================================================
        self.base_path = Path(__file__).parent.parent
        self.keypoints_file = self.base_path / 'dataset' / 'keypoints_dataset.csv'
        self.phases_file = self.base_path / 'dataset' / 'jump_phases_SkiTB.csv'
        self.jp_data_file = self.base_path / 'dataset' / 'JP_data.csv'
        self.timeseries_metrics_file = self.base_path / 'metrics' / 'timeseries_metrics' / 'timeseries_per_frame.csv'
        
        # Output directory (timeseries_analysis subfolder)
        self.output_dir = self.base_path / 'metrics' / 'timeseries_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # =====================================================================
        # ANALYSIS PARAMETERS
        # =====================================================================
        self.normalized_length = 100  # Standard length for normalized series
        self.fps = 30  # Frame rate
        
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
            bool: True if loading successful
        """
        # Load keypoints
        if not self.keypoints_file.exists():
            print(f"❌ Keypoints not found: {self.keypoints_file}")
            return False
        
        self.df_kpts = pd.read_csv(self.keypoints_file)
        
        # Normalize jump IDs
        self.df_kpts['jump_id'] = self.df_kpts['jump_id'].apply(self._normalize_jid)
        
        # Extract frame numbers
        self.df_kpts['frame_idx'] = self.df_kpts['frame_name'].apply(self._extract_frame_num)
        
        # Remove duplicates
        self.df_kpts = self.df_kpts.drop_duplicates(subset=['jump_id', 'frame_idx'], keep='first')
        
        # Load phases
        if self.phases_file.exists():
            self.df_phases = pd.read_csv(self.phases_file)
            self.df_phases['jump_id'] = self.df_phases['jump_id'].apply(self._normalize_jid)
        else:
            self.df_phases = None
            print("⚠️ Phases file not found")
        
        # Load JP data for scores
        if self.jp_data_file.exists():
            self.df_jp = pd.read_csv(self.jp_data_file)
            # Compute style score (sum of middle 3 judges)
            self._compute_scores()
        else:
            self.df_jp = None
            print("⚠️ JP_data not found - score analysis unavailable")
        
        # Load per-frame metrics if available
        if self.timeseries_metrics_file.exists():
            self.df_ts_metrics = pd.read_csv(self.timeseries_metrics_file)
        else:
            self.df_ts_metrics = None
        
        print(f"✅ Loaded {len(self.df_kpts)} keypoint frames")
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
    
    def _compute_scores(self):
        """Compute Physical_Score and Style_Score from judge data."""
        scores = []
        for _, row in self.df_jp.iterrows():
            judge_scores = [row.get(f'AthleteJdg{x}', np.nan) for x in 'ABCDE']
            valid = [s for s in judge_scores if pd.notna(s)]
            
            if len(valid) >= 5:
                sorted_scores = sorted(valid)
                style = sum(sorted_scores[1:4])  # Middle 3
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
                'AthleteDistance': row.get('AthleteDistance', np.nan)
            })
        
        self.df_scores = pd.DataFrame(scores)
    
    # =========================================================================
    # UTILITY FUNCTIONS
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
    # 1. TEMPORAL NORMALIZATION
    # =========================================================================
    
    def normalize_timeseries(self, series: pd.Series, target_length: int = None) -> np.ndarray:
        """
        Normalize a time series to a standard length using interpolation.
        
        This allows comparison of jumps with different flight durations
        by mapping all sequences to a common time scale [0, 1].
        
        ALGORITHM:
        ----------
        1. Remove NaN values
        2. Create interpolation function
        3. Resample at evenly spaced points
        
        Args:
            series: Input time series (indexed by frame number)
            target_length: Target number of points (default: self.normalized_length)
            
        Returns:
            np.ndarray: Normalized series of length target_length
        """
        if target_length is None:
            target_length = self.normalized_length
        
        # Clean the series
        clean = series.dropna()
        if len(clean) < 3:
            return np.full(target_length, np.nan)
        
        # Original time points (normalized to 0-1)
        t_original = np.linspace(0, 1, len(clean))
        values = clean.values
        
        # Create interpolation function
        try:
            interp_func = interp1d(t_original, values, kind='cubic', fill_value='extrapolate')
        except:
            interp_func = interp1d(t_original, values, kind='linear', fill_value='extrapolate')
        
        # Resample at target points
        t_target = np.linspace(0, 1, target_length)
        normalized = interp_func(t_target)
        
        return normalized
    
    def extract_flight_trajectory(self, jump_id: str, 
                                   metric: str = 'body_ski_angle') -> Optional[np.ndarray]:
        """
        Extract and normalize the flight trajectory for a specific metric.
        
        Args:
            jump_id: Jump identifier
            metric: Which metric to extract ('body_ski_angle', 'knee_angle', 'hip_height')
            
        Returns:
            np.ndarray: Normalized trajectory, or None if unavailable
        """
        if self.df_phases is None:
            return None
        
        # Get phase info
        phase_row = self.df_phases[self.df_phases['jump_id'] == jump_id]
        if phase_row.empty:
            return None
        phase_row = phase_row.iloc[0]
        
        # Get flight window
        bsa_start = phase_row.get('bsa_start')
        landing = phase_row.get('landing')
        
        if pd.isna(bsa_start) or pd.isna(landing):
            return None
        
        bsa_start, landing = int(bsa_start), int(landing)
        
        # Get keypoints for this jump
        jump_df = self.df_kpts[self.df_kpts['jump_id'] == jump_id].copy()
        jump_df = jump_df[(jump_df['frame_idx'] >= bsa_start) & (jump_df['frame_idx'] <= landing)]
        jump_df = jump_df.sort_values('frame_idx')
        
        if len(jump_df) < 5:
            return None
        
        # Compute the metric for each frame
        values = {}
        
        for _, row in jump_df.iterrows():
            frame_idx = row['frame_idx']
            
            if metric == 'body_ski_angle':
                # Body-ski angle computation
                neck = self.get_point(row, 'neck')
                pelvis = self.get_point(row, 'center_pelvis')
                r_tip = self.get_point(row, 'r_ski_tip')
                r_tail = self.get_point(row, 'r_ski_tail')
                
                if neck is not None and pelvis is not None and r_tip is not None and r_tail is not None:
                    body_vec = neck - pelvis
                    ski_vec = r_tip - r_tail
                    
                    dot = np.dot(body_vec, ski_vec)
                    norms = np.linalg.norm(body_vec) * np.linalg.norm(ski_vec)
                    if norms > 0:
                        angle = np.degrees(np.arccos(np.clip(dot / norms, -1, 1)))
                        values[frame_idx] = angle
            
            elif metric == 'hip_height':
                pelvis = self.get_point(row, 'center_pelvis')
                if pelvis is not None:
                    values[frame_idx] = 1.0 - pelvis[1]  # Invert Y
            
            elif metric == 'knee_angle':
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
                            values[frame_idx] = angle
                            break
        
        if len(values) < 5:
            return None
        
        # Normalize to standard length
        series = pd.Series(values).sort_index()
        normalized = self.normalize_timeseries(series)
        
        return normalized
    

    
    # =========================================================================
    # 3. TOP VS FLOP ANALYSIS
    # =========================================================================
    
    def compare_top_vs_flop(self, metric: str = 'body_ski_angle',
                            top_n: int = 10, score_type: str = 'Style_Score') -> Dict:
        """
        Compare average trajectories of top performers vs bottom performers.
        
        This analysis reveals systematic differences in technique between
        successful and unsuccessful jumps.
        
        ALGORITHM:
        ----------
        1. Rank jumps by score
        2. Extract normalized trajectories for top and bottom groups
        3. Compute mean and confidence intervals
        4. Statistical comparison at each time point
        
        Args:
            metric: Which trajectory to compare
            top_n: Number of jumps in each group
            score_type: 'Style_Score' or 'Physical_Score'
            
        Returns:
            Dict with trajectories and statistical comparison
        """
        if self.df_scores is None:
            print("❌ Score data not available")
            return {}
        
        print(f"\n📊 Comparing Top {top_n} vs Flop {top_n} for {metric}...")
        
        # Get available jumps with scores
        available_jumps = self.df_kpts['jump_id'].unique()
        scores_with_data = self.df_scores[
            (self.df_scores['jump_id'].isin(available_jumps)) &
            (self.df_scores[score_type].notna())
        ].copy()
        
        if len(scores_with_data) < 2 * top_n:
            print(f"⚠️ Not enough jumps with scores: {len(scores_with_data)}")
            top_n = len(scores_with_data) // 3
            if top_n < 2:
                return {}
        
        # Sort by score
        scores_with_data = scores_with_data.sort_values(score_type, ascending=False)
        
        top_jumps = scores_with_data.head(top_n)['jump_id'].tolist()
        flop_jumps = scores_with_data.tail(top_n)['jump_id'].tolist()
        
        print(f"  Top jumps: {top_jumps}")
        print(f"  Flop jumps: {flop_jumps}")
        
        # Extract trajectories
        top_trajectories = []
        flop_trajectories = []
        
        for jid in top_jumps:
            traj = self.extract_flight_trajectory(jid, metric)
            if traj is not None and not np.all(np.isnan(traj)):
                top_trajectories.append(traj)
        
        for jid in flop_jumps:
            traj = self.extract_flight_trajectory(jid, metric)
            if traj is not None and not np.all(np.isnan(traj)):
                flop_trajectories.append(traj)
        
        print(f"  Valid trajectories - Top: {len(top_trajectories)}, Flop: {len(flop_trajectories)}")
        
        if len(top_trajectories) < 2 or len(flop_trajectories) < 2:
            return {}
        
        # Stack into matrices
        top_matrix = np.vstack(top_trajectories)
        flop_matrix = np.vstack(flop_trajectories)
        
        # Compute statistics
        result = {
            'top_mean': np.nanmean(top_matrix, axis=0),
            'top_std': np.nanstd(top_matrix, axis=0),
            'flop_mean': np.nanmean(flop_matrix, axis=0),
            'flop_std': np.nanstd(flop_matrix, axis=0),
            'top_n': len(top_trajectories),
            'flop_n': len(flop_trajectories),
            'time_axis': np.linspace(0, 100, self.normalized_length)
        }
        
        # Statistical significance at each point
        from scipy import stats
        p_values = []
        for i in range(self.normalized_length):
            top_vals = top_matrix[:, i]
            flop_vals = flop_matrix[:, i]
            top_valid = top_vals[~np.isnan(top_vals)]
            flop_valid = flop_vals[~np.isnan(flop_vals)]
            
            if len(top_valid) >= 2 and len(flop_valid) >= 2:
                _, p = stats.ttest_ind(top_valid, flop_valid)
                p_values.append(p)
            else:
                p_values.append(np.nan)
        
        result['p_values'] = np.array(p_values)
        result['significant_regions'] = np.array(p_values) < 0.05
        
        return result
    

    
    # =========================================================================
    # 5. VISUALIZATION
    # =========================================================================
    
    def plot_top_vs_flop(self, comparison_result: Dict, 
                         metric_name: str = 'Body-Ski Angle',
                         save_path: Optional[Path] = None):
        """
        Plot comparison of top vs flop trajectories.
        
        Args:
            comparison_result: Output from compare_top_vs_flop()
            metric_name: Name for plot title
            save_path: Path to save figure (optional)
        """
        if not comparison_result:
            print("⚠️ No data to plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        # Main trajectory plot
        ax1 = axes[0]
        time = comparison_result['time_axis']
        
        # Top performers (green)
        ax1.plot(time, comparison_result['top_mean'], 'g-', linewidth=2, label='Top Performers')
        ax1.fill_between(time, 
                         comparison_result['top_mean'] - comparison_result['top_std'],
                         comparison_result['top_mean'] + comparison_result['top_std'],
                         alpha=0.3, color='green')
        
        # Flop performers (red)
        ax1.plot(time, comparison_result['flop_mean'], 'r-', linewidth=2, label='Flop Performers')
        ax1.fill_between(time, 
                         comparison_result['flop_mean'] - comparison_result['flop_std'],
                         comparison_result['flop_mean'] + comparison_result['flop_std'],
                         alpha=0.3, color='red')
        
        # Mark significant regions
        sig = comparison_result['significant_regions']
        for i in range(len(time) - 1):
            if sig[i]:
                ax1.axvspan(time[i], time[i+1], alpha=0.1, color='blue')
        
        ax1.set_xlabel('Normalized Flight Time (%)')
        ax1.set_ylabel(metric_name)
        ax1.set_title(f'{metric_name}: Top {comparison_result["top_n"]} vs Flop {comparison_result["flop_n"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # P-value plot
        ax2 = axes[1]
        ax2.semilogy(time, comparison_result['p_values'], 'b-', linewidth=1)
        ax2.axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
        ax2.set_xlabel('Normalized Flight Time (%)')
        ax2.set_ylabel('p-value (log)')
        ax2.set_title('Statistical Significance (blue regions above)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")
        
        plt.close()
    

    
    # =========================================================================
    # MAIN ANALYSIS PIPELINE
    # =========================================================================
    
    def run_full_analysis(self):
        """
        Run complete time-series analysis pipeline.
        
        This executes all analyses and generates reports/plots.
        """
        if not self.load_data():
            return False
        
        print("\n" + "=" * 70)
        print("TIME-SERIES ANALYSIS")
        print("=" * 70)
        
        # =====================================================================
        # 1. EXTRACT ALL TRAJECTORIES
        # =====================================================================
        print("\n📊 Extracting flight trajectories...")
        
        trajectories = {}
        for jid in self.df_kpts['jump_id'].unique():
            traj = self.extract_flight_trajectory(jid, 'body_ski_angle')
            if traj is not None and not np.all(np.isnan(traj)):
                trajectories[jid] = traj
        
        print(f"  Extracted {len(trajectories)} valid trajectories")
        
        # =====================================================================
        # 2. TOP VS FLOP COMPARISON
        # =====================================================================
        print("\n📊 Comparing Top vs Flop performers...")
        
        for metric in ['body_ski_angle', 'hip_height']:
            comparison = self.compare_top_vs_flop(metric, top_n=5)
            
            if comparison:
                plot_path = self.output_dir / f'top_vs_flop_{metric}.png'
                self.plot_top_vs_flop(comparison, metric.replace('_', ' ').title(), plot_path)
        
        # =====================================================================
        # SUMMARY
        # =====================================================================
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)
        
        return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TIME-SERIES ANALYSIS FOR SKI JUMPING")
    print("=" * 70)
    
    analyzer = TimeSeriesAnalyzer()
    analyzer.run_full_analysis()
