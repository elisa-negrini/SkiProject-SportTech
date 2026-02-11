from pathlib import Path

import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "correlations_detailed.csv"
    output_file = base_dir / "correlations_filtered.csv"

    df = pd.read_csv(input_file)
    df["pearson_p"] = pd.to_numeric(df["pearson_p"], errors="coerce")

    filtered_df = df[(df["target"] != "AthleteScore") & (df["pearson_p"] <= 0.1)]
    filtered_df.to_csv(output_file, index=False)

    print(f"Saved {len(filtered_df)} rows to: {output_file}")


if __name__ == "__main__":
    main()
