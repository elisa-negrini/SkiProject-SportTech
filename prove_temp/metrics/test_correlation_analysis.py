"""
TEST: Correlation Analysis & Feature Importance for Ski Jumping
================================================================
Analyzes correlations between time-series metrics and competition scores.

Computes:
1. Physical_Score = AthleteScore - Style_Score (pure physical performance)
2. Style_Score = Sum of middle 3 judges (technical/style evaluation)
3. Correlation matrix between metrics and scores
4. Feature Importance via Random Forest
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Optional: ML imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.inspection import permutation_importance
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ scikit-learn not installed - ML features disabled")


class CorrelationAnalyzer:
    """
    Analyzes correlations between computed metrics and competition scores.
    """
    
    def __init__(self):
        # --- PATHS ---
        self.base_path = Path(__file__).parent.parent.parent
        self.jp_data_file = self.base_path / 'dataset' / 'JP_data.csv'
        
        # Input metrics (from subfolders)
        self.metrics_file = self.base_path / 'metrics' / 'timeseries_metrics' / 'timeseries_summary.csv'
        self.old_metrics_file = self.base_path / 'metrics' / 'core_metrics' / 'metrics_summary_per_jump.csv'
        self.advanced_metrics_file = self.base_path / 'metrics' / 'advanced_metrics' / 'advanced_metrics_summary.csv'
        
        # Output (to correlations subfolder)
        self.output_dir = self.base_path / 'metrics' / 'correlations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_merged = self.output_dir / 'merged_scores_metrics.csv'
        self.output_correlations = self.output_dir / 'correlations.csv'
        self.output_plot_corr = self.output_dir / 'correlation_matrix.png'
        self.output_plot_importance = self.output_dir / 'feature_importance.png'
        
    def load_data(self) -> bool:
        """Load JP data and metrics."""
        
        if not self.jp_data_file.exists():
            print(f"❌ JP_data.csv not found: {self.jp_data_file}")
            return False
        
        self.df_jp = pd.read_csv(self.jp_data_file)
        print(f"✅ Loaded JP data: {len(self.df_jp)} jumps")
        
        # Load time-series metrics
        if self.metrics_file.exists():
            self.df_metrics = pd.read_csv(self.metrics_file)
            print(f"✅ Loaded time-series metrics: {len(self.df_metrics)} jumps")
        else:
            print(f"⚠️ Time-series metrics not found, run test_timeseries_metrics.py first")
            self.df_metrics = pd.DataFrame()
        
        # Load old metrics for comparison
        if self.old_metrics_file.exists():
            self.df_old_metrics = pd.read_csv(self.old_metrics_file)
            print(f"✅ Loaded old metrics: {len(self.df_old_metrics)} jumps")
        else:
            self.df_old_metrics = pd.DataFrame()
        
        # Load advanced metrics
        if self.advanced_metrics_file.exists():
            self.df_advanced_metrics = pd.read_csv(self.advanced_metrics_file)
            print(f"✅ Loaded advanced metrics: {len(self.df_advanced_metrics)} jumps")
        else:
            print(f"⚠️ Advanced metrics not found")
            self.df_advanced_metrics = pd.DataFrame()
        
        return True
    
    def compute_scores(self) -> pd.DataFrame:
        """
        Compute Physical_Score and Style_Score from JP_data.
        
        Physical_Score = AthleteScore - Style_Score
        Style_Score = Sum of middle 3 judges (removing min and max)
        
        Returns:
            DataFrame with ID, Physical_Score, Style_Score
        """
        scores = []
        
        for _, row in self.df_jp.iterrows():
            jump_id = row['ID']
            athlete_score = row.get('AthleteScore', np.nan)
            
            # Get 5 judge scores
            judge_scores = [
                row.get('AthleteJdgA', np.nan),
                row.get('AthleteJdgB', np.nan),
                row.get('AthleteJdgC', np.nan),
                row.get('AthleteJdgD', np.nan),
                row.get('AthleteJdgE', np.nan)
            ]
            
            # Remove NaN
            valid_scores = [s for s in judge_scores if pd.notna(s)]
            
            if len(valid_scores) >= 3 and pd.notna(athlete_score):
                # Sort and take middle 3 (remove min and max if 5 scores)
                sorted_scores = sorted(valid_scores)
                if len(sorted_scores) >= 5:
                    middle_3 = sorted_scores[1:4]  # Remove min and max
                elif len(sorted_scores) >= 3:
                    middle_3 = sorted_scores[:3]
                else:
                    middle_3 = sorted_scores
                
                style_score = sum(middle_3)
                physical_score = athlete_score - style_score
                
                scores.append({
                    'jump_id': jump_id,
                    'AthleteScore': athlete_score,
                    'Style_Score': style_score,
                    'Physical_Score': physical_score,
                    'AthleteDistance': row.get('AthleteDistance', np.nan),
                    'AthleteName': f"{row.get('AthleteName', '')} {row.get('AthleteSurname', '')}",
                    'HillHS': row.get('HillHS', np.nan)
                })
            else:
                scores.append({
                    'jump_id': jump_id,
                    'AthleteScore': athlete_score,
                    'Style_Score': np.nan,
                    'Physical_Score': np.nan,
                    'AthleteDistance': row.get('AthleteDistance', np.nan),
                    'AthleteName': f"{row.get('AthleteName', '')} {row.get('AthleteSurname', '')}",
                    'HillHS': row.get('HillHS', np.nan)
                })
        
        df_scores = pd.DataFrame(scores)
        
        print(f"\n📊 Score Computation:")
        print(f"   Valid Style_Score: {df_scores['Style_Score'].notna().sum()}")
        print(f"   Valid Physical_Score: {df_scores['Physical_Score'].notna().sum()}")
        
        return df_scores
    
    def merge_data(self, df_scores: pd.DataFrame) -> pd.DataFrame:
        """Merge scores with metrics."""
        
        # Start with scores
        df_merged = df_scores.copy()
        
        # Merge with time-series metrics
        if not self.df_metrics.empty:
            print(f"\n📊 Merging timeseries metrics...")
            print(f"   Before: {len(df_merged)} rows, {len(df_merged.columns)} cols")
            df_merged = df_merged.merge(
                self.df_metrics, 
                on='jump_id', 
                how='left'
            )
            print(f"   After: {len(df_merged)} rows, {len(df_merged.columns)} cols")
            print(f"   Timeseries data present: {df_merged['flight_std'].notna().sum() if 'flight_std' in df_merged.columns else 0} jumps")
        
        # Merge with core metrics
        if not self.df_old_metrics.empty:
            print(f"\n📊 Merging core metrics...")
            print(f"   Before: {len(df_merged)} rows, {len(df_merged.columns)} cols")
            # Select only columns that exist in df_old_metrics
            old_cols = ['jump_id', 'avg_body_ski_angle', 'avg_v_style_front', 'avg_v_style_back',
                       'takeoff_knee_angle', 'avg_symmetry_index_back',
                       'avg_telemark_proj_ski', 'avg_telemark_depth_ratio', 'avg_telemark_leg_angle']
            existing_cols = [c for c in old_cols if c in self.df_old_metrics.columns]
            df_merged = df_merged.merge(
                self.df_old_metrics[existing_cols],
                on='jump_id',
                how='left',
                suffixes=('', '_core')
            )
            print(f"   After: {len(df_merged)} rows, {len(df_merged.columns)} cols")
            print(f"   Core data present: {df_merged['avg_body_ski_angle'].notna().sum() if 'avg_body_ski_angle' in df_merged.columns else 0} jumps")
        
        # Merge with advanced metrics
        if not self.df_advanced_metrics.empty:
            print(f"\n📊 Merging advanced metrics...")
            print(f"   Before: {len(df_merged)} rows, {len(df_merged.columns)} cols")
            df_merged = df_merged.merge(
                self.df_advanced_metrics,
                on='jump_id',
                how='left',
                suffixes=('', '_adv')
            )
            print(f"   After: {len(df_merged)} rows, {len(df_merged.columns)} cols")
            print(f"   Advanced data present: {df_merged['takeoff_peak_velocity'].notna().sum() if 'takeoff_peak_velocity' in df_merged.columns else 0} jumps")
        
        # Save merged dataset
        df_merged.to_csv(self.output_merged, index=False)
        print(f"\n✅ Merged data saved: {self.output_merged}")
        print(f"   Total rows: {len(df_merged)}")
        print(f"   Total columns: {len(df_merged.columns)}")
        print(f"   Rows with ANY metrics: {df_merged.iloc[:, 7:].notna().any(axis=1).sum()}")
        
        return df_merged
    
    def compute_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute correlation matrix between metrics and scores.
        """
        
        # Define ALL metrics to analyze (complete feature set)
        metric_cols = [
            # Timeseries metrics
            'knee_peak_velocity', 'knee_angle_at_takeoff',
            'flight_std', 'flight_jitter', 'flight_mean_bsa',
            'landing_hip_velocity', 'landing_knee_compression',
            # Core geometric metrics
            'avg_body_ski_angle', 'avg_v_style_front', 'avg_v_style_back',
            'takeoff_knee_angle', 'avg_symmetry_index_back',
            'avg_telemark_proj_ski', 'avg_telemark_depth_ratio', 'avg_telemark_leg_angle',
            # Advanced metrics
            'takeoff_timing_offset', 'takeoff_peak_velocity',
            'telemark_scissor_mean', 'telemark_stability', 'landing_absorption_rate'
        ]
        
        # Target columns
        target_cols = ['Physical_Score', 'Style_Score', 'AthleteDistance', 'AthleteScore']
        
        # Filter to existing columns
        metric_cols = [c for c in metric_cols if c in df.columns]
        target_cols = [c for c in target_cols if c in df.columns]
        
        if not metric_cols or not target_cols:
            print("❌ No valid columns for correlation analysis")
            return pd.DataFrame()
        
        print(f"\n📈 Computing correlations...")
        print(f"   Metrics: {len(metric_cols)}")
        print(f"   Targets: {len(target_cols)}")
        
        # Compute correlation matrix
        all_cols = metric_cols + target_cols
        df_subset = df[all_cols].dropna(how='all')
        
        corr_results = []
        
        for metric in metric_cols:
            for target in target_cols:
                # Get valid pairs
                valid_mask = df_subset[[metric, target]].notna().all(axis=1)
                n_valid = valid_mask.sum()
                
                if n_valid >= 5:  # Minimum samples for correlation
                    x = df_subset.loc[valid_mask, metric]
                    y = df_subset.loc[valid_mask, target]
                    
                    # Pearson correlation
                    r, p_value = stats.pearsonr(x, y)
                    
                    # Spearman correlation (rank-based, more robust)
                    rho, p_spearman = stats.spearmanr(x, y)
                    
                    corr_results.append({
                        'metric': metric,
                        'target': target,
                        'pearson_r': r,
                        'pearson_p': p_value,
                        'spearman_rho': rho,
                        'spearman_p': p_spearman,
                        'n_samples': n_valid,
                        'significant': p_value < 0.05
                    })
        
        df_corr = pd.DataFrame(corr_results)
        df_corr.to_csv(self.output_correlations, index=False)
        print(f"\n✅ Correlations saved: {self.output_correlations}")
        
        # Print significant correlations
        if not df_corr.empty:
            sig = df_corr[df_corr['significant']]
            if not sig.empty:
                print("\n🎯 SIGNIFICANT CORRELATIONS (p < 0.05):")
                for _, row in sig.iterrows():
                    direction = "↑" if row['pearson_r'] > 0 else "↓"
                    print(f"   {row['metric']} vs {row['target']}: r={row['pearson_r']:.3f} {direction} (p={row['pearson_p']:.4f}, n={row['n_samples']})")
        
        return df_corr
    
    def plot_correlation_matrix(self, df: pd.DataFrame):
        """Plot correlation heatmap."""
        
        # Select numeric columns
        metric_cols = [
            'knee_peak_velocity', 'flight_std', 'flight_jitter', 'landing_smoothness_score',
            'Physical_Score', 'Style_Score', 'AthleteDistance'
        ]
        
        metric_cols = [c for c in metric_cols if c in df.columns]
        
        if len(metric_cols) < 3:
            print("⚠️ Not enough columns for correlation matrix plot")
            return
        
        df_subset = df[metric_cols].dropna()
        
        if len(df_subset) < 5:
            print(f"⚠️ Not enough samples for correlation matrix: {len(df_subset)}")
            return
        
        # Compute correlation matrix
        corr_matrix = df_subset.corr()
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax,
            annot_kws={'size': 9}
        )
        
        ax.set_title('Correlation Matrix: Time-Series Metrics vs Scores\n(TEST)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(self.output_plot_corr, dpi=150, bbox_inches='tight')
        print(f"\n✅ Correlation plot saved: {self.output_plot_corr}")
        plt.close()
    
    def run_feature_importance(self, df: pd.DataFrame):
        """
        Run Random Forest to get feature importance.
        Uses Leave-One-Out cross-validation due to small sample size.
        """
        
        if not HAS_SKLEARN:
            print("⚠️ scikit-learn not available, skipping feature importance")
            return
        
        # Feature columns (new metrics)
        feature_cols = [
            'knee_peak_velocity', 'knee_extension_range',
            'flight_std', 'flight_jitter', 'flight_trend',
            'landing_smoothness_score'
        ]
        
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        if len(feature_cols) < 2:
            print("⚠️ Not enough feature columns for ML")
            return
        
        # Targets
        targets = {
            'Physical_Score': 'Physical Score (Distance-related)',
            'Style_Score': 'Style Score (Judges)'
        }
        
        results = {}
        
        for target_col, target_name in targets.items():
            if target_col not in df.columns:
                continue
            
            # Prepare data (drop NaN rows)
            df_ml = df[feature_cols + [target_col]].dropna()
            
            if len(df_ml) < 10:
                print(f"⚠️ Not enough samples for {target_col}: {len(df_ml)}")
                continue
            
            X = df_ml[feature_cols].values
            y = df_ml[target_col].values
            
            print(f"\n🤖 Random Forest for {target_name}")
            print(f"   Samples: {len(df_ml)}")
            print(f"   Features: {len(feature_cols)}")
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Random Forest with LOO-CV
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=3,  # Prevent overfitting with small data
                random_state=42,
                n_jobs=-1
            )
            
            # Leave-One-Out Cross Validation
            loo = LeaveOneOut()
            scores = cross_val_score(rf, X_scaled, y, cv=loo, scoring='r2')
            
            print(f"   LOO CV R² mean: {scores.mean():.3f} (±{scores.std():.3f})")
            
            # Fit on all data for feature importance
            rf.fit(X_scaled, y)
            
            # Permutation importance (more reliable than built-in)
            perm_imp = permutation_importance(rf, X_scaled, y, n_repeats=30, random_state=42)
            
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance_mean': perm_imp.importances_mean,
                'importance_std': perm_imp.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            print(f"\n   Feature Importance ({target_col}):")
            for _, row in importance_df.iterrows():
                bar = '█' * int(row['importance_mean'] * 50) if row['importance_mean'] > 0 else ''
                print(f"      {row['feature']:25s}: {row['importance_mean']:6.3f} ±{row['importance_std']:.3f} {bar}")
            
            results[target_col] = importance_df
        
        # Plot feature importance
        if results:
            self._plot_feature_importance(results)
    
    def _plot_feature_importance(self, results: Dict):
        """Plot feature importance comparison."""
        
        n_targets = len(results)
        fig, axes = plt.subplots(1, n_targets, figsize=(7 * n_targets, 6))
        
        if n_targets == 1:
            axes = [axes]
        
        colors = ['#2ecc71', '#3498db']
        
        for idx, (target, df_imp) in enumerate(results.items()):
            ax = axes[idx]
            
            y_pos = np.arange(len(df_imp))
            
            ax.barh(y_pos, df_imp['importance_mean'], 
                    xerr=df_imp['importance_std'],
                    color=colors[idx % len(colors)],
                    alpha=0.8,
                    capsize=3)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_imp['feature'])
            ax.set_xlabel('Permutation Importance')
            ax.set_title(f'Feature Importance: {target}', fontweight='bold')
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        
        plt.suptitle('Random Forest Feature Importance Analysis (TEST)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plt.savefig(self.output_plot_importance, dpi=150, bbox_inches='tight')
        print(f"\n✅ Feature importance plot saved: {self.output_plot_importance}")
        plt.close()
    
    def run_analysis(self):
        """Run full correlation analysis pipeline."""
        
        if not self.load_data():
            return False
        
        print("\n" + "="*60)
        print("CORRELATION & FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Step 1: Compute scores
        df_scores = self.compute_scores()
        
        # Step 2: Merge with metrics
        df_merged = self.merge_data(df_scores)
        
        # Step 3: Compute correlations
        df_corr = self.compute_correlations(df_merged)
        
        # Step 4: Plot correlation matrix
        self.plot_correlation_matrix(df_merged)
        
        # Step 5: Feature importance (ML)
        self.run_feature_importance(df_merged)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return True


if __name__ == "__main__":
    analyzer = CorrelationAnalyzer()
    analyzer.run_analysis()
