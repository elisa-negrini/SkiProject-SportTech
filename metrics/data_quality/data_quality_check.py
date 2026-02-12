"""
This script validates all computed metrics and identifies:
1. Physically impossible values (outside validity ranges)
2. Statistical outliers (beyond 3 standard deviations)
3. Missing data patterns

Output:
- data_quality/outliers_report.csv
- data_quality/warnings_report.csv
- data_quality/data_quality_summary.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class DataQualityChecker:
    """
    Validates metrics data for physically impossible or statistically extreme values.
    """
    # removed: flight_std: (0,15), flight_jitter: (0, 10), flight mean_bsa: (0, 45)
    # removed: avg_v_style_front: (10, 70), avg_v_style_back: (10, 70) 
    # body_ski_inclination: (0,45), knee_angle_at_takeoff; (90,180)
    # takeoff_peak_velocity: (50, 800), telemark_leg_angle: (0, 90),

    VALIDITY_RANGES = {
        # V-style angles
        'avg_v_style_front': (10, 70),
        'avg_v_style_back': (10, 70),
        
        # Body-ski angle
        'avg_body_ski_angle': (0, 45),
        
        # Knee dynamics
        'knee_peak_velocity': (50, 800),
        'takeoff_knee_angle': (90, 180),
        'takeoff_peak_velocity': (50, 800),
        
        # Takeoff dynamics
        'takeoff_timing_offset': (-10, 10),
        
        # Landing metrics
        'landing_hip_velocity': (0, 3),
        'landing_knee_compression': (0, 90),
        'telemark_scissor_mean': (0, 0.70),
        'telemark_stability': (0, 50),
        'avg_telemark_leg_angle': (0, 90),
        
        # Symmetry
        'avg_symmetry_index_back': (0, 30),
        'avg_symmetry_index_front': (0, 30),
    }
    
    def __init__(self):
        """Initialize paths and configuration."""
        self.base_path = Path(__file__).parent.parent.parent
        self.metrics_path = self.base_path / 'metrics'
        
        self.metrics_file = self.metrics_path / 'core_metrics' / 'metrics_summary_per_jump.csv'
        
        self.output_dir = self.metrics_path / 'data_quality'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.outliers = []
        self.warnings = []
    
    def load_data(self) -> pd.DataFrame:
        """Load metrics data from single CSV."""
        
        if not self.metrics_file.exists():
            print(f"❌ File not found: {self.metrics_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.metrics_file)
        print(f" Loaded metrics: {len(df)} jumps, {len(df.columns)} columns")
        print(f"   File: {self.metrics_file}")
        
        return df
    
    def check_validity_ranges(self, df: pd.DataFrame):
        """Check each metric against its validity range."""
        
        print("\n" + "=" * 60)
        print("CHECKING VALIDITY RANGES")
        print("=" * 60)
        
        issues_found = 0
        
        for metric, (min_val, max_val) in self.VALIDITY_RANGES.items():
            if metric not in df.columns:
                continue
            
            mask_low = (df[metric] < min_val) & (df[metric].notna())
            mask_high = (df[metric] > max_val) & (df[metric].notna())
            
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
                    issues_found += 1
            
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
                    issues_found += 1
        
        if issues_found == 0:
            print("   No validity range violations found!")
    
    def check_statistical_outliers(self, df: pd.DataFrame, z_threshold: float = 3.0):
        """Identify statistical outliers using z-score."""
        
        print("\n" + "=" * 60)
        print(f"CHECKING STATISTICAL OUTLIERS (z > {z_threshold})")
        print("=" * 60)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'jump_id' and c not in self.VALIDITY_RANGES]
        
        warnings_found = 0
        
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
                    
                    if not any(o['jump_id'] == jump_id and o['metric'] == col for o in self.outliers):
                        self.warnings.append({
                            'jump_id': jump_id,
                            'metric': col,
                            'value': value,
                            'z_score': z,
                            'mean': mean,
                            'std': std,
                            'issue': 'STATISTICAL_OUTLIER',
                            'action': 'VERIFY'
                        })
                        print(f"  ⚠️ {jump_id}: {col}={value:.2f} (z={z:.2f})")
                        warnings_found += 1
        
        if warnings_found == 0:
            print("  ✅ No statistical outliers found!")
    
    def generate_report(self, df: pd.DataFrame):
        """Generate summary report and CSV output."""
        
        if self.outliers:
            df_outliers = pd.DataFrame(self.outliers)
            outliers_path = self.output_dir / 'outliers_report.csv'
            df_outliers.to_csv(outliers_path, index=False)
            print(f"\n✅ Outliers report: {outliers_path}")
        
        if self.warnings:
            df_warnings = pd.DataFrame(self.warnings)
            warnings_path = self.output_dir / 'warnings_report.csv'
            df_warnings.to_csv(warnings_path, index=False)
            print(f"✅ Warnings report: {warnings_path}")
        
        summary_file = self.output_dir / 'data_quality_summary.txt'
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("DATA QUALITY REPORT - SKI JUMPING METRICS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Total jumps analyzed: {len(df)}\n")
            f.write(f"Total metrics checked: {len(self.VALIDITY_RANGES)}\n\n")
            
            f.write("CRITICAL ISSUES (values outside physical range):\n")
            f.write("-" * 70 + "\n")
            if self.outliers:
                for o in self.outliers:
                    f.write(f"  {o['jump_id']}: {o['metric']}={o['value']:.2f} ")
                    f.write(f"(valid range: {o['min_valid']}-{o['max_valid']}) ")
                    f.write(f"→ {o['action']}\n")
            else:
                f.write("  None found!\n")
            
            f.write(f"\nTotal critical issues: {len(self.outliers)}\n\n")
            
            f.write("WARNINGS (statistical outliers, z > 3):\n")
            f.write("-" * 70 + "\n")
            if self.warnings:
                for w in self.warnings:
                    f.write(f"  {w['jump_id']}: {w['metric']}={w['value']:.2f} ")
                    f.write(f"(z={w['z_score']:.2f}, μ={w['mean']:.2f}, σ={w['std']:.2f}) ")
                    f.write(f"→ {w['action']}\n")
            else:
                f.write("  None found!\n")
            
            f.write(f"\nTotal warnings: {len(self.warnings)}\n\n")
            
            f.write("MISSING DATA ANALYSIS:\n")
            f.write("-" * 70 + "\n")
            
            missing_data = {}
            for col in df.columns:
                if col == 'jump_id':
                    continue
                n_missing = df[col].isna().sum()
                if n_missing > 0:
                    pct = 100 * n_missing / len(df)
                    missing_data[col] = (n_missing, pct)
            
            if missing_data:
                for col, (n_missing, pct) in sorted(missing_data.items(), key=lambda x: -x[1][1]):
                    f.write(f"  {col}: {n_missing}/{len(df)} missing ({pct:.1f}%)\n")
            else:
                f.write("  No missing data found!\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"✅ Valid jumps: {len(df) - len(set(o['jump_id'] for o in self.outliers))}\n")
            f.write(f"❌ Jumps with critical issues: {len(set(o['jump_id'] for o in self.outliers))}\n")
            f.write(f"⚠️  Jumps with warnings: {len(set(w['jump_id'] for w in self.warnings))}\n")
        
        print(f" Summary report: {summary_file}")
    
    def run(self):
        
        print("=" * 60)
        print("DATA QUALITY CHECK - SKI JUMPING METRICS")
        print("=" * 60)
        
        df = self.load_data()
        
        if df.empty:
            print("❌ No data to check!")
            return
        
        self.check_validity_ranges(df)
        self.check_statistical_outliers(df)
        self.generate_report(df)
        
        print("\n" + "=" * 60)
        print(f"FINAL SUMMARY: {len(self.outliers)} critical issues, {len(self.warnings)} warnings")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    checker = DataQualityChecker()
    checker.run()