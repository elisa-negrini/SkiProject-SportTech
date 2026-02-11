from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def significance_symbol(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "correlations_filtered.csv"
    output_file = base_dir / "selected_correlations_table.png"

    df = pd.read_csv(input_file)

    targets = ["Physical_Score", "Style_Score"]
    df = df[df["target"].isin(targets)].copy()

    corr_table = df.pivot_table(
        index="metric",
        columns="target",
        values="pearson_r",
        aggfunc="first",
    ).reindex(columns=targets)

    p_table = df.pivot_table(
        index="metric",
        columns="target",
        values="pearson_p",
        aggfunc="first",
    ).reindex(columns=targets)

    annot = corr_table.copy().astype(object)
    for metric in annot.index:
        for target in annot.columns:
            r_value = corr_table.loc[metric, target]
            p_value = p_table.loc[metric, target]
            if pd.isna(r_value):
                annot.loc[metric, target] = ""
            else:
                annot.loc[metric, target] = f"{r_value:.2f}{significance_symbol(p_value)}"

    plt.figure(figsize=(7, max(3, 0.5 * len(corr_table) + 1.5)))
    sns.heatmap(
        corr_table,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        annot=annot,
        fmt="",
        cbar_kws={"label": "Pearson r"},
    )

    plt.title("Selected Correlations", fontweight="bold")
    plt.xlabel("Target", fontweight="bold")
    plt.ylabel("Metric", fontweight="bold")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.figtext(
        0.5,
        0.015,
        "Legend (Pearson p-value): ** pearson_p < 0.05   * pearson_p < 0.10",
        ha="center",
        fontstyle="italic",
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to: {output_file}")


if __name__ == "__main__":
    main()
