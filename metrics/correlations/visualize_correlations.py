"""
Correlation Visualization Suite
================================

Generates comprehensive visualizations of correlation analysis:
1. Full heatmap (all metrics × all targets)
2. Significant correlations heatmap (p < 0.05)
3. Top correlations bar chart
4. Scatter plots for strongest correlations
5. Summary report with interpretation

Usage:
    python metrics/visualize_correlations.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class CorrelationVisualizer:
    """
    Creates publication-quality visualizations of correlation analysis.
    """
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.corr_file = self.base_path / 'metrics' / 'correlations' / 'correlations.csv'
        self.merged_file = self.base_path / 'metrics' / 'correlations' / 'merged_scores_metrics.csv'
        self.output_dir = self.base_path / 'metrics' / 'visualizations' / 'correlations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load correlation data."""
        if not self.corr_file.exists():
            print(f"❌ Correlations file not found: {self.corr_file}")
            print("   Run: python metrics/test_correlation_analysis.py")
            return False
        
        self.df_corr = pd.read_csv(self.corr_file)
        print(f"✅ Loaded {len(self.df_corr)} correlations")
        
        if self.merged_file.exists():
            self.df_merged = pd.read_csv(self.merged_file)
            print(f"✅ Loaded merged data: {len(self.df_merged)} jumps")
        else:
            self.df_merged = None
            print("⚠️ Merged data not found - scatter plots disabled")
        
        return True
    
    def plot_full_heatmap(self):
        """
        Plot 1: Full correlation heatmap (all metrics × all targets).
        """
        print("\n📊 Creating full heatmap...")
        
        # Pivot to matrix format
        pivot = self.df_corr.pivot_table(
            index='metric',
            columns='target',
            values='pearson_r',
            aggfunc='first'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Pearson r'},
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title('Correlation Matrix: All Metrics vs All Targets', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Target Variables', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output = self.output_dir / '1_heatmap_full.png'
        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"   ✅ Saved: {output}")
        plt.close()
    
    def plot_significant_heatmap(self):
        """
        Plot 2: Heatmap of only significant correlations (p < 0.05).
        """
        print("\n📊 Creating significant correlations heatmap...")
        
        # Filter significant
        df_sig = self.df_corr[self.df_corr['significant'] == True].copy()
        
        if df_sig.empty:
            print("   ⚠️ No significant correlations found")
            return
        
        # Pivot to matrix
        pivot = df_sig.pivot_table(
            index='metric',
            columns='target',
            values='pearson_r',
            aggfunc='first'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Pearson r'},
            linewidths=1,
            ax=ax
        )
        
        ax.set_title('Significant Correlations Only (p < 0.05)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Target Variables', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output = self.output_dir / '2_heatmap_significant.png'
        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"   ✅ Saved: {output}")
        plt.close()
    
    def plot_top_correlations_bar(self, top_n=15):
        """
        Plot 3: Bar chart of top N correlations by absolute value.
        """
        print(f"\n📊 Creating top {top_n} correlations bar chart...")
        
        # Add absolute value column
        df_plot = self.df_corr.copy()
        df_plot['abs_r'] = df_plot['pearson_r'].abs()
        
        # Sort and take top N
        df_top = df_plot.nlargest(top_n, 'abs_r')
        
        # Create labels
        df_top['label'] = df_top['metric'] + '\n→ ' + df_top['target']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color by sign and significance
        colors = []
        for _, row in df_top.iterrows():
            if row['significant']:
                color = 'darkred' if row['pearson_r'] < 0 else 'darkgreen'
            else:
                color = 'lightcoral' if row['pearson_r'] < 0 else 'lightgreen'
            colors.append(color)
        
        # Plot bars
        bars = ax.barh(
            range(len(df_top)),
            df_top['pearson_r'],
            color=colors,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Labels
        ax.set_yticks(range(len(df_top)))
        ax.set_yticklabels(df_top['label'], fontsize=9)
        ax.set_xlabel('Pearson Correlation Coefficient (r)', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Correlations by Strength', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels
        for i, (_, row) in enumerate(df_top.iterrows()):
            label = f"r={row['pearson_r']:.2f}"
            if row['significant']:
                label += f" (p={row['pearson_p']:.3f})"
            
            x_pos = row['pearson_r'] + (0.02 if row['pearson_r'] > 0 else -0.02)
            ha = 'left' if row['pearson_r'] > 0 else 'right'
            
            ax.text(x_pos, i, label, va='center', ha=ha, fontsize=8)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', edgecolor='black', label='Positive (p<0.05)'),
            Patch(facecolor='darkred', edgecolor='black', label='Negative (p<0.05)'),
            Patch(facecolor='lightgreen', edgecolor='black', label='Positive (p≥0.05)'),
            Patch(facecolor='lightcoral', edgecolor='black', label='Negative (p≥0.05)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        output = self.output_dir / f'3_top_{top_n}_correlations.png'
        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"   ✅ Saved: {output}")
        plt.close()
    
    def plot_scatter_top_correlations(self, top_n=6):
        """
        Plot 4: Scatter plots for top N correlations with regression lines.
        """
        if self.df_merged is None:
            print("\n⚠️ Skipping scatter plots (merged data not available)")
            return
        
        print(f"\n📊 Creating scatter plots for top {top_n} correlations...")
        
        # Get top correlations
        df_plot = self.df_corr.copy()
        df_plot['abs_r'] = df_plot['pearson_r'].abs()
        df_top = df_plot.nlargest(top_n, 'abs_r')
        
        # Create subplots
        n_cols = 3
        n_rows = int(np.ceil(top_n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, (_, row) in enumerate(df_top.iterrows()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            metric = row['metric']
            target = row['target']
            r = row['pearson_r']
            p = row['pearson_p']
            n = int(row['n_samples'])
            
            # Get data
            data = self.df_merged[[metric, target]].dropna()
            
            if len(data) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            x = data[metric]
            y = data[target]
            
            # Scatter plot
            ax.scatter(x, y, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            # Regression line
            z = np.polyfit(x, y, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            color = 'red' if r < 0 else 'green'
            ax.plot(x_line, p_fit(x_line), color=color, linewidth=2, 
                   linestyle='--', alpha=0.8, label=f'r={r:.3f}')
            
            # Labels
            ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel(target.replace('_', ' ').title(), fontsize=10)
            
            # Title with stats
            sig_marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            title = f'{metric} → {target}\nr={r:.3f}{sig_marker}, p={p:.4f}, n={n}'
            ax.set_title(title, fontsize=9, fontweight='bold')
            
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(df_top), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Scatter Plots: Top Correlations with Regression Lines', 
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        output = self.output_dir / f'4_scatter_top_{top_n}.png'
        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"   ✅ Saved: {output}")
        plt.close()
    
    def plot_correlation_by_target(self):
        """
        Plot 5: Grouped bar chart showing correlations by target variable.
        """
        print("\n📊 Creating correlation comparison by target...")
        
        # Get significant correlations only
        df_sig = self.df_corr[self.df_corr['significant'] == True].copy()
        
        if df_sig.empty:
            print("   ⚠️ No significant correlations to plot")
            return
        
        # Group by target
        targets = df_sig['target'].unique()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, target in enumerate(targets):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Filter data for this target
            df_target = df_sig[df_sig['target'] == target].copy()
            df_target = df_target.sort_values('pearson_r', ascending=True)
            
            # Plot
            colors = ['red' if r < 0 else 'green' for r in df_target['pearson_r']]
            bars = ax.barh(range(len(df_target)), df_target['pearson_r'], 
                          color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Labels
            ax.set_yticks(range(len(df_target)))
            ax.set_yticklabels(df_target['metric'].str.replace('_', ' '), fontsize=9)
            ax.set_xlabel('Pearson r', fontsize=11, fontweight='bold')
            ax.set_title(f'Target: {target.replace("_", " ")}', 
                        fontsize=12, fontweight='bold')
            
            # Add value labels
            for i, (_, row) in enumerate(df_target.iterrows()):
                label = f"{row['pearson_r']:.2f}"
                x_pos = row['pearson_r'] + (0.02 if row['pearson_r'] > 0 else -0.02)
                ha = 'left' if row['pearson_r'] > 0 else 'right'
                ax.text(x_pos, i, label, va='center', ha=ha, fontsize=8)
            
            ax.axvline(x=0, color='black', linewidth=1)
            ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Significant Correlations Grouped by Target Variable', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output = self.output_dir / '5_correlations_by_target.png'
        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"   ✅ Saved: {output}")
        plt.close()
    
    def generate_summary_report(self):
        """
        Generate a text summary report.
        """
        print("\n📝 Generating summary report...")
        
        output = self.output_dir / 'VISUALIZATION_SUMMARY.txt'
        
        with open(output, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("CORRELATION VISUALIZATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            # Overall stats
            n_total = len(self.df_corr)
            n_sig = (self.df_corr['significant'] == True).sum()
            n_metrics = self.df_corr['metric'].nunique()
            n_targets = self.df_corr['target'].nunique()
            
            f.write("📊 DATASET OVERVIEW\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total correlations computed: {n_total}\n")
            f.write(f"Significant correlations (p<0.05): {n_sig} ({n_sig/n_total*100:.1f}%)\n")
            f.write(f"Unique metrics: {n_metrics}\n")
            f.write(f"Target variables: {n_targets}\n\n")
            
            # Top correlations
            df_top = self.df_corr.nlargest(10, 'pearson_r', keep='all')
            f.write("🔝 TOP 10 POSITIVE CORRELATIONS\n")
            f.write("-" * 70 + "\n")
            for _, row in df_top.iterrows():
                sig = " ***" if row['significant'] else ""
                f.write(f"{row['metric']:30s} → {row['target']:20s} "
                       f"r={row['pearson_r']:6.3f} (p={row['pearson_p']:.4f}){sig}\n")
            
            f.write("\n")
            
            # Bottom correlations
            df_bottom = self.df_corr.nsmallest(10, 'pearson_r', keep='all')
            f.write("🔻 TOP 10 NEGATIVE CORRELATIONS\n")
            f.write("-" * 70 + "\n")
            for _, row in df_bottom.iterrows():
                sig = " ***" if row['significant'] else ""
                f.write(f"{row['metric']:30s} → {row['target']:20s} "
                       f"r={row['pearson_r']:6.3f} (p={row['pearson_p']:.4f}){sig}\n")
            
            f.write("\n")
            
            # Significant by target
            f.write("📈 SIGNIFICANT CORRELATIONS BY TARGET\n")
            f.write("-" * 70 + "\n")
            df_sig = self.df_corr[self.df_corr['significant'] == True]
            for target in df_sig['target'].unique():
                df_t = df_sig[df_sig['target'] == target]
                f.write(f"\n{target}:\n")
                for _, row in df_t.iterrows():
                    f.write(f"  - {row['metric']:30s} r={row['pearson_r']:6.3f} (n={int(row['n_samples'])})\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"   ✅ Saved: {output}")
    
    def run_all(self):
        """Generate all visualizations."""
        
        print("=" * 70)
        print("CORRELATION VISUALIZATION SUITE")
        print("=" * 70)
        
        if not self.load_data():
            return False
        
        # Generate all plots
        self.plot_full_heatmap()
        self.plot_significant_heatmap()
        self.plot_top_correlations_bar(top_n=15)
        self.plot_scatter_top_correlations(top_n=6)
        self.plot_correlation_by_target()
        self.generate_summary_report()
        
        print("\n" + "=" * 70)
        print("✅ ALL VISUALIZATIONS COMPLETE!")
        print("=" * 70)
        print(f"\n📂 Output directory: {self.output_dir}")
        print("\n📊 Generated files:")
        print("   1. 1_heatmap_full.png              - All correlations matrix")
        print("   2. 2_heatmap_significant.png       - Significant only (p<0.05)")
        print("   3. 3_top_15_correlations.png       - Bar chart ranked by strength")
        print("   4. 4_scatter_top_6.png             - Scatter plots with regression")
        print("   5. 5_correlations_by_target.png    - Grouped by target variable")
        print("   6. VISUALIZATION_SUMMARY.txt       - Text report")
        print("\n")
        
        return True


if __name__ == "__main__":
    viz = CorrelationVisualizer()
    viz.run_all()
