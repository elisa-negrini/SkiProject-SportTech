from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


SELECTED_METRICS = [
    "landing_knee_compression",
    "telemark_stability",
    "telemark_scissor_mean",
    "avg_v_style_front",
]

TARGETS = ["Style_Score", "Physical_Score", "AthleteDistance", "AthleteScore"]

SCATTER_PAIRS = [
    ("landing_knee_compression", "Style_Score"),
    ("telemark_stability", "Physical_Score"),
    ("telemark_scissor_mean", "Style_Score"),
    ("avg_v_style_front", "AthleteDistance"),
]


def corr_stats(df: pd.DataFrame, x: str, y: str) -> dict:
    pair = df[[x, y]].dropna()
    n = len(pair)
    if n < 3:
        return {"metric": x, "target": y, "n": n, "pearson_r": None, "p_value": None}

    r, p = pearsonr(pair[x], pair[y])
    return {"metric": x, "target": y, "n": n, "pearson_r": r, "p_value": p}


def save_tables(data: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_total = len(data)
    overview_rows = []

    for metric in SELECTED_METRICS:
        s = pd.to_numeric(data[metric], errors="coerce")
        valid = s.dropna()
        overview_rows.append(
            {
                "metric": metric,
                "available_samples": int(valid.shape[0]),
                "missing_samples": int(n_total - valid.shape[0]),
                "missing_pct": round((1 - valid.shape[0] / n_total) * 100, 1),
                "mean": valid.mean(),
                "std": valid.std(ddof=1),
                "median": valid.median(),
                "min": valid.min(),
                "max": valid.max(),
            }
        )

    overview_df = pd.DataFrame(overview_rows)
    overview_df.to_csv(output_dir / "table_metric_overview.csv", index=False)

    corr_rows = []
    for metric in SELECTED_METRICS:
        for target in TARGETS:
            corr_rows.append(corr_stats(data, metric, target))

    corr_df = pd.DataFrame(corr_rows)
    corr_df = corr_df.sort_values(["p_value", "n"], ascending=[True, False], na_position="last")
    corr_df.to_csv(output_dir / "table_metric_target_correlations.csv", index=False)

    return overview_df, corr_df


def save_plots(data: pd.DataFrame, overview_df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    # 1) Data coverage per metric
    plt.figure(figsize=(9, 4))
    ax = sns.barplot(data=overview_df, x="metric", y="available_samples", color="#4472C4")
    ax.set_title("Available samples for selected metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Number of jumps")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_01_metric_coverage.png", dpi=150)
    plt.close()

    # 2) Distributions (one panel per metric)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, SELECTED_METRICS):
        s = pd.to_numeric(data[metric], errors="coerce").dropna()
        sns.histplot(s, bins=12, kde=True, ax=ax, color="#5B9BD5")
        ax.axvline(s.median(), color="#C00000", linestyle="--", linewidth=1.2, label="Median")
        ax.set_title(f"{metric} (n={len(s)})")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Selected metric distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_02_metric_distributions.png", dpi=150)
    plt.close()

    # 3) Key metric-target scatter plots with regression line
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (metric, target) in zip(axes.flat, SCATTER_PAIRS):
        pair = data[[metric, target]].dropna()
        sns.regplot(
            data=pair,
            x=metric,
            y=target,
            ax=ax,
            scatter_kws={"s": 35, "alpha": 0.75, "color": "#2F5597"},
            line_kws={"color": "#C00000", "linewidth": 1.8},
        )
        if len(pair) >= 3:
            r, p = pearsonr(pair[metric], pair[target])
            note = f"r={r:.2f} | p={p:.3f} | n={len(pair)}"
        else:
            note = f"n={len(pair)} (not enough for correlation)"

        ax.text(
            0.03,
            0.97,
            note,
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            fontsize=9,
        )
        ax.set_title(f"{metric} vs {target}")
    fig.suptitle("Main metric-target relationships", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_03_metric_target_scatter.png", dpi=150)
    plt.close()

    # 4) Correlation heatmap among selected metrics and key scores
    heatmap_cols = SELECTED_METRICS + ["Style_Score", "Physical_Score", "AthleteDistance"]
    heat = data[heatmap_cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    plt.title("Correlation heatmap (selected metrics + key targets)")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_04_correlation_heatmap.png", dpi=150)
    plt.close()


def save_report(
    input_file: Path,
    output_dir: Path,
    data: pd.DataFrame,
    overview_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> None:
    report_file = output_dir / "findings_report.md"

    ranked = corr_df.dropna(subset=["pearson_r", "p_value"]).copy()
    ranked["abs_r"] = ranked["pearson_r"].abs()
    ranked = ranked.sort_values(["p_value", "abs_r"], ascending=[True, False])

    significant = ranked[ranked["p_value"] < 0.05].head(6)

    lines = [
        "# Basic Data Visualization Report",
        "",
        "## Scope",
        f"- Input data: `{input_file}`",
        f"- Total jumps in merged table: **{len(data)}**",
        f"- Selected metrics: `{', '.join(SELECTED_METRICS)}`",
        "- Note: `avg_v_style_front` is used as-is (no normalization, no adjustment).",
        "",
        "## Visualizations (what each plot shows)",
        "- `fig_01_metric_coverage.png`: available sample count per selected metric.",
        "- `fig_02_metric_distributions.png`: raw distributions of the 4 selected metrics.",
        "- `fig_03_metric_target_scatter.png`: key metric-target relationships with regression lines and r/p/n.",
        "- `fig_04_correlation_heatmap.png`: compact correlation map among selected metrics and key targets.",
        "",
        "## Tables",
        "- `table_metric_overview.csv`: data availability + basic descriptive statistics.",
        "- `table_metric_target_correlations.csv`: Pearson correlation, p-value, and sample size per metric-target pair.",
        "",
        "## Main interpretations (basic)",
    ]

    if significant.empty:
        lines.append("- No metric-target pair reached p < 0.05 in this filtered basic analysis.")
    else:
        for _, row in significant.iterrows():
            direction = "positive" if row["pearson_r"] > 0 else "negative"
            lines.append(
                f"- `{row['metric']}` vs `{row['target']}`: {direction} association "
                f"(r={row['pearson_r']:.2f}, p={row['p_value']:.3f}, n={int(row['n'])})."
            )

    low_coverage = overview_df[overview_df["available_samples"] < 20]
    if not low_coverage.empty:
        lines.append(
            "- Some selected metrics have low sample size (<20), so effect estimates should be treated as preliminary."
        )

    lines.append(
        "- Use this report as a first-pass descriptive analysis before any advanced modeling or causal interpretation."
    )

    report_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "correlations" / "merged_scores_metrics.csv"
    output_dir = base_dir / "findings_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")

    data = pd.read_csv(input_file)

    required_cols = ["jump_id"] + SELECTED_METRICS + TARGETS
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data = data[required_cols].copy()

    overview_df, corr_df = save_tables(data, output_dir)
    save_plots(data, overview_df, output_dir)
    save_report(input_file, output_dir, data, overview_df, corr_df)

    print("Basic findings report created.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

