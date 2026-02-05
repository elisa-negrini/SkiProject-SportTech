"""
Data Quality Check for Ski Jumping Metrics
==========================================

This script validates all computed metrics and identifies:
1. Physically impossible values (outside validity ranges)
2. Statistical outliers (beyond 3 standard deviations)
3. Missing data patterns

Output:
- data_quality/outliers_report.csv
- data_quality/data_quality_summary.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataQualityChecker:
    """
    Validates metrics data for physically impossible or statistically extreme values.
    """
    
    # Validity ranges based on biomechanical constraints
    VALIDITY_RANGES = {
        # Flight metrics (kept: std, jitter, mean_bsa)
        'flight_std': (0, 15),               # BSA shouldn't vary > 15 degrees
        'flight_jitter': (0, 10),            # Frame-to-frame variation
        'flight_mean_bsa': (0, 45),          # BSA inclination 0-45 degrees
        # REMOVED: flight_range, flight_trend (redundant features)
        
        # Body-ski angle
        'avg_body_ski_angle': (0, 45),       # Inclination angle
        'body_ski_inclination': (0, 45),     # Same metric, different name
        
        # V-style
        'avg_v_style_front': (10, 70),       # V-style angle range
        'avg_v_style_back': (10, 70),
        
        # Knee dynamics (kept: peak_velocity, angle_at_takeoff)
        'knee_peak_velocity': (50, 800),     # Degrees per second
        'knee_angle_at_takeoff': (90, 180),  # Full extension ~180
        'takeoff_knee_angle': (90, 180),
        # REMOVED: knee_mean_velocity, knee_extension_range (redundant)
        
        # Takeoff dynamics
        'takeoff_peak_velocity': (50, 800),
        'takeoff_timing_offset': (-10, 10),  # Frames from expected
        
        # Landing metrics (kept: hip_velocity, knee_compression)
        'landing_hip_velocity': (0, 3),      # Normalized velocity
        'landing_knee_compression': (0, 90), # Degrees
        # REMOVED: landing_hip_drop, landing_smoothness_score, landing_absorption_rate
        
        # Telemark
        'telemark_scissor_mean': (0, 0.30),  # Normalized
        'telemark_stability': (0, 50),       # Std of telemark position
        'telemark_leg_angle': (0, 90),
        'avg_telemark_leg_angle': (0, 90),
        
        # Symmetry
        'avg_symmetry_index_front': (0, 30),
        'avg_symmetry_index_back': (0, 30),
    }
    
    def __init__(self):
        """Initialize paths and configuration."""
        self.base_path = Path(__file__).parent.parent
        self.metrics_path = Path(__file__).parent
        
        # Input files
        self.timeseries_file = self.metrics_path / 'timeseries_metrics' / 'timeseries_summary.csv'
        self.core_metrics_file = self.metrics_path / 'core_metrics' / 'metrics_summary_per_jump.csv'
        self.advanced_file = self.metrics_path / 'advanced_metrics' / 'advanced_metrics_summary.csv'
        
        # Output directory
        self.output_dir = self.metrics_path / 'data_quality'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.outliers = []
        self.warnings = []
    
    def load_data(self) -> pd.DataFrame:
        """Load and merge all metrics data."""
        
        dfs = []
        
        if self.timeseries_file.exists():
            df_ts = pd.read_csv(self.timeseries_file)
            dfs.append(df_ts)
            print(f"✅ Loaded timeseries: {len(df_ts)} jumps")
        
        if self.core_metrics_file.exists():
            df_core = pd.read_csv(self.core_metrics_file)
            dfs.append(df_core)
            print(f"✅ Loaded core metrics: {len(df_core)} jumps")
        
        if self.advanced_file.exists():
            df_adv = pd.read_csv(self.advanced_file)
            dfs.append(df_adv)
            print(f"✅ Loaded advanced metrics: {len(df_adv)} jumps")
        
        if not dfs:
            print("❌ No metrics files found!")
            return pd.DataFrame()
        
        # Merge on jump_id
        df = dfs[0]
        for other_df in dfs[1:]:
            if 'jump_id' in other_df.columns:
                df = df.merge(other_df, on='jump_id', how='outer', suffixes=('', '_dup'))
                # Remove duplicate columns
                df = df[[c for c in df.columns if not c.endswith('_dup')]]
        
        return df
    
    def check_validity_ranges(self, df: pd.DataFrame):
        """Check each metric against its validity range."""
        
        print("\n" + "=" * 60)
        print("CHECKING VALIDITY RANGES")
        print("=" * 60)
        
        for metric, (min_val, max_val) in self.VALIDITY_RANGES.items():
            if metric not in df.columns:
                continue
            
            # Find values outside range
            mask_low = df[metric] < min_val
            mask_high = df[metric] > max_val
            
            invalid_low = df[mask_low][['jump_id', metric]].copy()
            invalid_high = df[mask_high][['jump_id', metric]].copy()
            
            if len(invalid_low) > 0:
                for _, row in invalid_low.iterrows():
                    self.outliers.append({
                        'jump_id': row['jump_id'],
                        'metric': metric,
                        'value': row[metric],
                        'min_valid': min_val,
                        'max_valid': max_val,
                        'issue': 'BELOW_MINIMUM',
                        'action': 'EXCLUDE'
                    })
                    print(f"  ❌ {row['jump_id']}: {metric}={row[metric]:.2f} < {min_val}")
            
            if len(invalid_high) > 0:
                for _, row in invalid_high.iterrows():
                    self.outliers.append({
                        'jump_id': row['jump_id'],
                        'metric': metric,
                        'value': row[metric],
                        'min_valid': min_val,
                        'max_valid': max_val,
                        'issue': 'ABOVE_MAXIMUM',
                        'action': 'EXCLUDE'
                    })
                    print(f"  ❌ {row['jump_id']}: {metric}={row[metric]:.2f} > {max_val}")
    
    def check_statistical_outliers(self, df: pd.DataFrame, z_threshold: float = 3.0):
        """Identify statistical outliers using z-score."""
        
        print("\n" + "=" * 60)
        print("CHECKING STATISTICAL OUTLIERS (z > 3)")
        print("=" * 60)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'jump_id']
        
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) < 3:
                continue
            
            mean = values.mean()
            std = values.std()
            
            if std == 0:
                continue
            
            z_scores = (values - mean) / std
            outlier_mask = abs(z_scores) > z_threshold
            
            if outlier_mask.any():
                outlier_indices = values[outlier_mask].index
                for idx in outlier_indices:
                    jump_id = df.loc[idx, 'jump_id']
                    value = df.loc[idx, col]
                    z = z_scores[idx]
                    
                    # Don't duplicate if already flagged by validity check
                    if not any(o['jump_id'] == jump_id and o['metric'] == col for o in self.outliers):
                        self.warnings.append({
                            'jump_id': jump_id,
                            'metric': col,
                            'value': value,
                            'z_score': z,
                            'issue': 'STATISTICAL_OUTLIER',
                            'action': 'VERIFY'
                        })
                        print(f"  ⚠️ {jump_id}: {col}={value:.2f} (z={z:.2f})")
    
    def generate_report(self, df: pd.DataFrame):
        """Generate summary report and CSV output."""
        
        # Save outliers CSV
        if self.outliers:
            df_outliers = pd.DataFrame(self.outliers)
            df_outliers.to_csv(self.output_dir / 'outliers_report.csv', index=False)
            print(f"\n✅ Outliers saved: {self.output_dir / 'outliers_report.csv'}")
        
        # Save warnings CSV
        if self.warnings:
            df_warnings = pd.DataFrame(self.warnings)
            df_warnings.to_csv(self.output_dir / 'warnings_report.csv', index=False)
            print(f"✅ Warnings saved: {self.output_dir / 'warnings_report.csv'}")
        
        # Generate text summary
        summary_file = self.output_dir / 'data_quality_summary.txt'
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("DATA QUALITY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total jumps analyzed: {len(df)}\n")
            f.write(f"Total metrics checked: {len(self.VALIDITY_RANGES)}\n\n")
            
            f.write("CRITICAL ISSUES (values outside physical range):\n")
            f.write("-" * 40 + "\n")
            if self.outliers:
                for o in self.outliers:
                    f.write(f"  {o['jump_id']}: {o['metric']}={o['value']:.2f} ")
                    f.write(f"(valid: {o['min_valid']}-{o['max_valid']}) -> {o['action']}\n")
            else:
                f.write("  None found!\n")
            
            f.write(f"\nTotal critical issues: {len(self.outliers)}\n\n")
            
            f.write("WARNINGS (statistical outliers, z > 3):\n")
            f.write("-" * 40 + "\n")
            if self.warnings:
                for w in self.warnings:
                    f.write(f"  {w['jump_id']}: {w['metric']}={w['value']:.2f} ")
                    f.write(f"(z={w['z_score']:.2f}) -> {w['action']}\n")
            else:
                f.write("  None found!\n")
            
            f.write(f"\nTotal warnings: {len(self.warnings)}\n\n")
            
            # Missing data summary
            f.write("MISSING DATA SUMMARY:\n")
            f.write("-" * 40 + "\n")
            
            for col in df.columns:
                if col == 'jump_id':
                    continue
                n_missing = df[col].isna().sum()
                if n_missing > 0:
                    pct = 100 * n_missing / len(df)
                    f.write(f"  {col}: {n_missing}/{len(df)} missing ({pct:.0f}%)\n")
        
        print(f"✅ Summary saved: {summary_file}")
    
    def run(self):
        """Main execution."""
        
        print("=" * 60)
        print("DATA QUALITY CHECK")
        print("=" * 60)
        
        df = self.load_data()
        
        if df.empty:
            print("No data to check!")
            return
        
        self.check_validity_ranges(df)
        self.check_statistical_outliers(df)
        self.generate_report(df)
        
        print("\n" + "=" * 60)
        print(f"SUMMARY: {len(self.outliers)} critical issues, {len(self.warnings)} warnings")
        print("=" * 60)


if __name__ == "__main__":
    checker = DataQualityChecker()
    checker.run()
