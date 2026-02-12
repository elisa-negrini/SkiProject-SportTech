from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'


class CorrelationVisualizer:
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.corr_file = self.base_path / 'metrics' / 'correlations' / 'correlations_detailed.csv'
        self.merged_file = self.base_path / 'metrics' / 'correlations' / 'merged_data_complete.csv'
        
        self.output_dir = self.base_path / 'metrics' / 'correlations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df_corr = None
        self.df_merged = None
    
    def load_data(self) -> bool:
        
        print("[*] LOADING DATA FOR VISUALIZATIONS")
        
        if not self.corr_file.exists():
            print(f"[!] Correlations file not found: {self.corr_file}")
            print("[*] Run correlation_analysis.py first")
            return False
        
        self.df_corr = pd.read_csv(self.corr_file)
        print(f"[+] Loaded correlations: {len(self.df_corr)} results")
        
        if self.merged_file.exists():
            self.df_merged = pd.read_csv(self.merged_file)
            print(f"[+] Loaded merged data: {len(self.df_merged)} jumps")
        else:
            print("[*] Merged data not found (some visualizations will be skipped)")
            self.df_merged = None
        
        return True
    
    def plot_heatmap_all(self):
        """Visualization 1: Full correlation heatmap (all metrics × all targets)."""
        print("\n[*] Creating full heatmap...")
        
        pivot = self.df_corr.pivot_table(
            index='metric',
            columns='target',
            values='pearson_r',
            aggfunc='first'
        )
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(pivot)*0.3)))
        
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
            ax=ax,
            annot_kws={'size': 9}
        )
        
        ax.set_title('Correlation Matrix: All Metrics vs Targets', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Target Variables', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        output = self.output_dir / '1_heatmap_all_correlations.png'
        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"    [+] Saved: {output.name}")
        plt.close()
    
    def plot_heatmap_significant(self):
        """Visualization 2: Significant correlations only (p < 0.05)."""
        print("[*] Creating significant correlations heatmap...")
        
        df_sig = self.df_corr[self.df_corr['significant_p05']]
        
        if df_sig.empty:
            print("    [!] No significant correlations found (p<0.05)")
            return
        
        pivot = df_sig.pivot_table(
            index='metric',
            columns='target',
            values='pearson_r',
            aggfunc='first'
        )
        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot)*0.4)))
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Pearson r'},
            linewidths=1,
            ax=ax,
            annot_kws={'size': 10, 'weight': 'bold'}
        )
        
        ax.set_title('Significant Correlations (p < 0.05)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Target Variables', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        output = self.output_dir / '2_heatmap_significant_correlations.png'
        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"    [+] Saved: {output.name}")
        plt.close()
    
    def plot_heatmap_clustering(self):
        """Visualization 3: Hierarchical clustering heatmap."""
        
        print("[*] Creating clustering heatmap...")
        
        pivot = self.df_corr.pivot_table(
            index='metric',
            columns='target',
            values='pearson_r',
            aggfunc='first'
        ).fillna(0)
        
        fig = plt.figure(figsize=(12, max(8, len(pivot)*0.25)))
        
        sns.clustermap(
            pivot,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Pearson r'},
            figsize=(12, max(8, len(pivot)*0.25)),
            annot=True,
            fmt='.2f',
            linewidths=0.5
        )
        
        plt.suptitle('Hierarchical Clustering of Correlations', 
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output = self.output_dir / '3_heatmap_clustering.png'
        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"    [+] Saved: {output.name}")
        plt.close()
    
    def plot_top_correlations_bar(self, top_n=20):
        """Visualization 4: Top N correlations bar chart."""
        
        print(f"[*] Creating top {top_n} correlations bar chart...")
        
        df_top = self.df_corr.nlargest(top_n, 'abs_r').sort_values('pearson_r')
        
        df_top['label'] = df_top['metric'] + '\n(' + df_top['target'].str.replace('_', ' ') + ')'
        
        fig, ax = plt.subplots(figsize=(12, max(8, top_n*0.3)))
        
        colors = ['#d62728' if p >= 0.05 else '#2ca02c' 
                  for p in df_top['pearson_p']]
        
        bars = ax.barh(range(len(df_top)), df_top['pearson_r'], color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(df_top)))
        ax.set_yticklabels(df_top['label'], fontsize=9)
        ax.set_xlabel('Pearson r', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Correlations (sorted by value)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        for i, (idx, row) in enumerate(df_top.iterrows()):
            ax.text(row['pearson_r'] + 0.02 if row['pearson_r'] > 0 else row['pearson_r'] - 0.02, i, 
                   f"{row['pearson_r']:.3f}", va='center', fontsize=8)
        
        legend_elements = [
            Patch(facecolor='#2ca02c', alpha=0.8, label='Significant (p<0.05)'),
            Patch(facecolor='#d62728', alpha=0.8, label='Not significant (p>=0.05)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        output = self.output_dir / '4_top_correlations_bar_chart.png'
        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"    [+] Saved: {output.name}")
        plt.close()
    
    def plot_heatmap_filtered_by_effect_size(self, threshold=0.2):
        """Visualization 5: Heatmap with metrics filtered by effect size.
        Shows only metrics that have |r| >= threshold for at least one target.
        """        
        print(f"[*] Creating filtered heatmap (effect size threshold: |r| >= {threshold})...")
        
        pivot = self.df_corr.pivot_table(
            index='metric',
            columns='target',
            values='pearson_r',
            aggfunc='first'
        )
        
        mask = (abs(pivot) >= threshold).any(axis=1)
        pivot_filtered = pivot[mask]
        
        if pivot_filtered.empty:
            print(f"    [!] No metrics with |r| >= {threshold}, skipping")
            return
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot_filtered)*0.4)))
        
        sns.heatmap(
            pivot_filtered,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Pearson r'},
            linewidths=1,
            ax=ax,
            annot_kws={'size': 10, 'weight': 'bold'}
        )
        
        ax.set_title(f'Metrics with Effect Size |r| >= {threshold} (at least one target)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Target Variables', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        output = self.output_dir / '5_heatmap_filtered_by_effect_size.png'
        plt.savefig(output, bbox_inches='tight', dpi=300)
        print(f"    [+] Saved: {output.name}")
        print(f"       Metrics shown: {len(pivot_filtered)}/{len(pivot)}")
        plt.close()
    
    def run_all(self):
        """Generate all visualizations."""
        
        print("\n\n")
        print("=" * 60)
        print("CORRELATION VISUALIZATIONS")
        print("Ski Jumping Performance Metrics")
        print("=" * 60)
        
        if not self.load_data():
            return False
        
        print("\n[*] Generating visualizations...\n")
        
        self.plot_heatmap_all()
        self.plot_heatmap_significant()
        self.plot_heatmap_clustering()
        self.plot_heatmap_filtered_by_effect_size(threshold=0.2)
        self.plot_top_correlations_bar(top_n=20)
        
        print("\n" + "=" * 60)
        print("[+] ALL VISUALIZATIONS COMPLETE!")
        print("=" * 60)
        print(f"\n[*] Output files in: {self.output_dir}\n")
        
        return True

if __name__ == "__main__":
    viz = CorrelationVisualizer()
    viz.run_all()