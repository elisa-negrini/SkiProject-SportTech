"""
Intelligent Correlation Analysis for Ski Jumping
==================================================
Analyzes correlations between jump metrics and competition scores.

Targets: Physical_Score, Style_Score, AthleteScore
Features: All available biomechanics metrics

Output:
- correlations_detailed.csv (complete stats + CI)
- merged_data.csv (all data merged)
"""

from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class IntelligentCorrelationAnalyzer:
    """Advanced correlation analysis with robustness checks."""
    
    def __init__(self):
        # --- PATHS ---
        self.base_path = Path(__file__).parent.parent.parent
        self.jp_data_file = self.base_path / 'dataset' / 'JP_data.csv'
        self.timeseries_metrics_file = self.base_path / 'metrics' / 'core_metrics' / 'timeseries_metrics' / 'additional_timeseries_metrics.csv'
        self.core_metrics_file = self.base_path / 'metrics' / 'core_metrics' / 'metrics_summary_per_jump.csv'
        
        # Output directory
        self.output_dir = self.base_path / 'metrics' / 'correlations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_merged = self.output_dir / 'merged_data_complete.csv'
        self.output_correlations = self.output_dir / 'correlations_detailed.csv'
        self.output_summary = self.output_dir / 'correlation_summary.txt'
        
        # Data containers
        self.df_jp = None
        self.df_timeseries = None
        self.df_core = None
        self.df_merged = None
        
    def load_data(self) -> bool:
        """Load all data sources."""
        
        print("=" * 60)
        print("📥 LOADING DATA")
        print("=" * 60)
        
        # Load JP data
        if not self.jp_data_file.exists():
            print(f"❌ JP_data.csv not found: {self.jp_data_file}")
            return False
        
        self.df_jp = pd.read_csv(self.jp_data_file)
        print(f"✅ Loaded JP_data: {len(self.df_jp)} jumps, {len(self.df_jp.columns)} columns")
        
        # Load timeseries metrics
        if self.timeseries_metrics_file.exists():
            self.df_timeseries = pd.read_csv(self.timeseries_metrics_file)
            print(f"✅ Loaded timeseries metrics: {len(self.df_timeseries)} jumps, {len(self.df_timeseries.columns)} columns")
        else:
            print(f"⚠️  Timeseries metrics not found")
            self.df_timeseries = pd.DataFrame()
        
        # Load core metrics
        if self.core_metrics_file.exists():
            self.df_core = pd.read_csv(self.core_metrics_file)
            print(f"✅ Loaded core metrics: {len(self.df_core)} jumps, {len(self.df_core.columns)} columns")
        else:
            print(f"⚠️  Core metrics not found")
            self.df_core = pd.DataFrame()
        
        return True
    
    def compute_scores(self) -> pd.DataFrame:
        """
        Compute target scores:
        - Style_Score: Mean of middle 3 judges (remove min/max)
        - Physical_Score: AthleteScore - Style_Score
        - Keep: AthleteScore, AthleteDistance
        """
        
        print("\n" + "=" * 60)
        print("📊 COMPUTING SCORES")
        print("=" * 60)
        
        scores = []
        judges_cols = ['AthleteJdgA', 'AthleteJdgB', 'AthleteJdgC', 'AthleteJdgD', 'AthleteJdgE']
        
        for idx, row in self.df_jp.iterrows():
            record = {
                'ID': row['ID'],
                'AthleteScore': row.get('AthleteScore', np.nan),
                'AthleteDistance': row.get('AthleteDistance', np.nan)
            }
            
            # Get judge scores
            judges = [row.get(col, np.nan) for col in judges_cols]
            judges_clean = [j for j in judges if pd.notna(j)]
            
            if len(judges_clean) >= 3:
                # Sort and take middle 3
                judges_sorted = sorted(judges_clean)
                middle_3 = judges_sorted[1:-1] if len(judges_sorted) >= 3 else judges_sorted
                style_score = np.mean(middle_3)
                record['Style_Score'] = style_score
                
                # Physical = Athlete - Style
                if pd.notna(record['AthleteScore']):
                    record['Physical_Score'] = record['AthleteScore'] - style_score
                else:
                    record['Physical_Score'] = np.nan
            else:
                record['Style_Score'] = np.nan
                record['Physical_Score'] = np.nan
            
            scores.append(record)
        
        df_scores = pd.DataFrame(scores)
        
        print(f"\n📋 Score Summary:")
        print(f"   Valid AthleteScore: {df_scores['AthleteScore'].notna().sum()}/{len(df_scores)}")
        print(f"   Valid Style_Score: {df_scores['Style_Score'].notna().sum()}/{len(df_scores)}")
        print(f"   Valid Physical_Score: {df_scores['Physical_Score'].notna().sum()}/{len(df_scores)}")
        print(f"   Valid AthleteDistance: {df_scores['AthleteDistance'].notna().sum()}/{len(df_scores)}")
        
        return df_scores
    
    def merge_data(self, df_scores: pd.DataFrame) -> pd.DataFrame:
        """Merge scores with all metrics."""
        
        print("\n" + "=" * 60)
        print("🔀 MERGING DATA")
        print("=" * 60)
        
        # Start with scores
        df_merged = df_scores.copy()
        
        # Merge timeseries metrics
        if not self.df_timeseries.empty:
            # Rename ID column to match
            df_ts = self.df_timeseries.copy()
            if 'jump_id' in df_ts.columns:
                df_ts = df_ts.rename(columns={'jump_id': 'ID'})
            
            df_merged = df_merged.merge(df_ts, on='ID', how='left')
            print(f"✅ Merged timeseries metrics: now {len(df_merged.columns)} columns")
        
        # Merge core metrics
        if not self.df_core.empty:
            df_core = self.df_core.copy()
            if 'jump_id' in df_core.columns:
                df_core = df_core.rename(columns={'jump_id': 'ID'})
            
            df_merged = df_merged.merge(df_core, on='ID', how='left')
            print(f"✅ Merged core metrics: now {len(df_merged.columns)} columns")
        
        # Save merged data
        df_merged.to_csv(self.output_merged, index=False)
        print(f"\n📁 Merged data saved: {self.output_merged}")
        print(f"   Total rows: {len(df_merged)}")
        print(f"   Total columns: {len(df_merged.columns)}")
        
        return df_merged
    
    def compute_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute robust correlations with:
        - Pearson r + p-value + 95% CI
        - Spearman rho + p-value
        - Effect size (r²)
        - Sample size
        """
        
        print("\n" + "=" * 60)
        print("📈 COMPUTING CORRELATIONS")
        print("=" * 60)
        
        # Define targets
        target_cols = ['Physical_Score', 'Style_Score', 'AthleteScore']
        target_cols = [c for c in target_cols if c in df.columns]
        
        # Define ALL metrics (exclude ID and scores)
        metric_cols = [c for c in df.columns 
                      if c not in ['ID', 'AthleteScore', 'AthleteDistance', 
                                   'Physical_Score', 'Style_Score']]
        
        print(f"\n📋 Analysis Setup:")
        print(f"   Metrics to analyze: {len(metric_cols)}")
        print(f"   Target variables: {len(target_cols)}")
        print(f"   Total correlations: {len(metric_cols) * len(target_cols)}")
        
        corr_results = []
        
        for target in target_cols:
            for metric in metric_cols:
                
                # Get valid pairs (both metric and target non-null)
                valid_idx = df[[metric, target]].notna().all(axis=1)
                n_valid = valid_idx.sum()
                
                if n_valid < 3:  # Need at least 3 points
                    continue
                
                x = df.loc[valid_idx, metric].values
                y = df.loc[valid_idx, target].values
                
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(x, y)
                
                # Spearman correlation
                spearman_rho, spearman_p = spearmanr(x, y)
                
                # R-squared (effect size)
                r_squared = pearson_r ** 2
                
                # 95% Confidence Interval for Pearson r (Fisher z-transform)
                z = 0.5 * np.log((1 + pearson_r) / (1 - pearson_r))
                se = 1 / np.sqrt(n_valid - 3)
                ci_lower = np.tanh(z - 1.96 * se)
                ci_upper = np.tanh(z + 1.96 * se)
                
                # Significance
                is_significant = pearson_p < 0.05
                
                corr_results.append({
                    'metric': metric,
                    'target': target,
                    'n_samples': n_valid,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'r_squared': r_squared,
                    'ci_lower_95': ci_lower,
                    'ci_upper_95': ci_upper,
                    'spearman_rho': spearman_rho,
                    'spearman_p': spearman_p,
                    'significant_p05': is_significant,
                    'abs_r': abs(pearson_r)
                })
        
        df_corr = pd.DataFrame(corr_results)
        
        # Sort by absolute correlation
        df_corr = df_corr.sort_values('abs_r', ascending=False)
        
        # Save
        df_corr.to_csv(self.output_correlations, index=False)
        print(f"\n✅ Correlations computed: {len(df_corr)} results")
        print(f"   Saved to: {self.output_correlations}")
        
        return df_corr
    
    def generate_summary_report(self, df_corr: pd.DataFrame):
        """Generate text summary of findings."""
        
        print("\n" + "=" * 60)
        print("📝 GENERATING SUMMARY REPORT")
        print("=" * 60)
        
        with open(self.output_summary, 'w') as f:
            f.write("INTELLIGENT CORRELATION ANALYSIS - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Total correlations computed: {len(df_corr)}\n")
            f.write(f"Significant correlations (p<0.05): {df_corr['significant_p05'].sum()}\n\n")
            
            # By target
            for target in df_corr['target'].unique():
                df_target = df_corr[df_corr['target'] == target]
                f.write(f"\n{'='*70}\n")
                f.write(f"TARGET: {target}\n")
                f.write(f"{'='*70}\n\n")
                
                # Significant only
                df_sig = df_target[df_target['significant_p05']]
                if not df_sig.empty:
                    f.write(f"✅ SIGNIFICANT CORRELATIONS (p<0.05): {len(df_sig)}\n")
                    f.write("-" * 70 + "\n")
                    for _, row in df_sig.iterrows():
                        f.write(f"\n{row['metric']}:\n")
                        f.write(f"   r = {row['pearson_r']:>7.4f}  (p = {row['pearson_p']:.4f})\n")
                        f.write(f"   95% CI: [{row['ci_lower_95']:>7.4f}, {row['ci_upper_95']:>7.4f}]\n")
                        f.write(f"   R² = {row['r_squared']:.4f} ({row['r_squared']*100:.2f}% variance explained)\n")
                        f.write(f"   Spearman ρ = {row['spearman_rho']:>7.4f}  (p = {row['spearman_p']:.4f})\n")
                else:
                    f.write(f"❌ No significant correlations found (p<0.05)\n\n")
                
                # Top 10 by absolute value (all)
                f.write(f"\n📊 TOP 10 CORRELATIONS (all):\n")
                f.write("-" * 70 + "\n")
                for i, (_, row) in enumerate(df_target.head(10).iterrows(), 1):
                    sig_marker = "✓" if row['significant_p05'] else " "
                    f.write(f"{i:2d}. [{sig_marker}] {row['metric']:35s} r={row['pearson_r']:>7.4f}  (n={row['n_samples']:2d})\n")
        
        print(f"✅ Summary report saved: {self.output_summary}")
    
    def run_analysis(self):
        """Execute complete analysis."""
        
        print("\n\n")
        print("╔" + "═" * 58 + "╗")
        print("║" + " " * 12 + "INTELLIGENT CORRELATION ANALYSIS" + " " * 14 + "║")
        print("║" + " " * 15 + "Ski Jumping Performance Metrics" + " " * 13 + "║")
        print("╚" + "═" * 58 + "╝")
        
        # Step 1: Load
        if not self.load_data():
            return False
        
        # Step 2: Compute scores
        df_scores = self.compute_scores()
        
        # Step 3: Merge
        self.df_merged = self.merge_data(df_scores)
        
        # Step 4: Correlate
        df_corr = self.compute_correlations(self.df_merged)
        
        # Step 5: Summary
        self.generate_summary_report(df_corr)
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"\n📁 Output files:")
        print(f"   1. {self.output_merged.name}")
        print(f"   2. {self.output_correlations.name}")
        print(f"   3. {self.output_summary.name}")
        print("\n🎯 Next: Run intelligent_visualizations.py\n")
        
        return True


if __name__ == "__main__":
    analyzer = IntelligentCorrelationAnalyzer()
    analyzer.run_analysis()