# 📊 Metrics — Biomechanical Analysis of Ski Jumping

This folder contains the full pipeline for computing, analysing, and visualising **2D biomechanical metrics** extracted from the annotated keypoints.

Starting from the normalized keypoints in `dataset/keypoints_dataset.csv` and the phase boundaries in `dataset/jump_phases_SkiTB.csv`, the pipeline produces geometric and dynamic metrics, correlates them with judges' scores and distance, checks data quality, and generates visual overlays.

---

## 📁 Folder Structure

```
metrics/
├── README.md                              ← This file
├── core_metrics/                          ← Per-frame & per-jump geometric metrics
│   ├── metrics_computation.py             ← Main computation script
│   ├── metrics_per_frame.csv              ← Metrics for every frame
│   ├── metrics_summary_per_jump.csv       ← Aggregated metrics per jump
│   └── timeseries_metrics/                ← Time-series dynamic metrics
│       ├── test_timeseries_metrics.py     ← Computation script
│       ├── timeseries_per_frame.csv       ← Per-frame BSA, knee angle, hip height
│       └── additional_timeseries_metrics.csv  ← Per-jump summary (velocity, jitter, …)
├── correlations/                          ← Correlation analysis with judges' scores
│   ├── correlation_analysis.py            ← Merge data and compute correlations
│   ├── selected_correlations.py           ← Filter significant correlations (p ≤ 0.1)
│   ├── correlation_visualizations.py      ← Heatmaps & bar charts
│   ├── visualize_sel_cor.py               ← Filtered correlation table image
│   ├── merged_data_complete.csv           ← All metrics + scores merged
│   ├── correlations_detailed.csv          ← Full correlation results
│   ├── correlations_filtered.csv          ← Filtered subset (Physical & Style only)
│   └── correlation_summary.txt            ← Human-readable report
├── data_quality/                          ← Outlier detection & validation
│   ├── data_quality_check.py              ← Validation script
│   ├── outliers_report.csv                ← Values outside physical ranges
│   ├── warnings_report.csv                ← Statistical outliers (Z > 3)
│   └── data_quality_summary.txt           ← Summary report
├── metrics_visualizations/                ← Frame overlays with metric annotations
│   ├── metrics_visualizer.py              ← Interactive + batch visualiser
│   └── frame_overlays/                    ← Output: annotated JPGs per jump/metric
├── profile_analysis/                      ← Top vs Flop athlete comparison
│   ├── profile_analysis.py                ← Phase-segmented analysis
│   ├── comprehensive_metrics.csv          ← Phase-level metrics per jump
│   ├── top_vs_flop_comparison.png         ← BSA curves: Top 5 vs Bottom 5
│   └── top_vs_flop_trends.csv             ← Normalised curve data
└── style_penalty_model/                   ← Ridge regression predicting style penalty
    ├── style_penalty_model.py             ← Training + LOO-CV script
    ├── style_penalty_predictions.csv      ← Actual vs predicted penalties
    └── STYLE_PENALTY_FORMULA.txt          ← Trained formula + coefficients
```

---

## 🚀 Quick Start — How to Run

> **Prerequisites:** Make sure `dataset/keypoints_dataset.csv`, `dataset/jump_phases_SkiTB.csv`, and `dataset/JP_data.csv` are present.  
> All commands are run **from the project root** (`SkiProject-SportTech/`).

### Execution Order

The scripts have dependencies — follow this order:

| Step | Script | What it does | Output |
|------|--------|--------------|--------|
| **1** | `python metrics/core_metrics/metrics_computation.py` | Compute per-frame geometric metrics (V-style, BSA, symmetry, telemark, takeoff) | `metrics_per_frame.csv`, `metrics_summary_per_jump.csv` |
| **2** | `python metrics/core_metrics/timeseries_metrics/test_timeseries_metrics.py` | Compute dynamic time-series metrics (velocity, stability, landing impact) | `timeseries_per_frame.csv`, `additional_timeseries_metrics.csv` |
| **3** | `python metrics/data_quality/data_quality_check.py` | Validate metrics: physical range checks + statistical outlier detection | `outliers_report.csv`, `warnings_report.csv`, `data_quality_summary.txt` |
| **4** | `python metrics/correlations/correlation_analysis.py` | Merge metrics with judges' scores, compute Pearson & Spearman correlations | `merged_data_complete.csv`, `correlations_detailed.csv`, `correlation_summary.txt` |
`correlations_filtered.csv` |
| **5** | `python metrics/correlations/correlation_visualizations.py` | Generate correlation heatmaps and bar charts | `1_heatmap_all_correlations.png`, … `5_heatmap_filtered_by_effect_size.png` |
| **6** | `python metrics/profile_analysis/profile_analysis.py` | Phase-segmented metrics + Top 5 vs Bottom 5 comparison plot | `comprehensive_metrics.csv`, `top_vs_flop_comparison.png` |
| **7** | `python metrics/style_penalty_model/style_penalty_model.py` | Train Ridge penalty model (LOO-CV) on 3 features | `style_penalty_predictions.csv`, `STYLE_PENALTY_FORMULA.txt` |
| **8** | `python metrics/metrics_visualizations/metrics_visualizer.py` | Interactive/batch metric overlay on frames | `frame_overlays/<jump>/<metric>/viz_*.jpg` |

---

## 📊 Metrics Overview

All metrics are computed from the 23-keypoint skeleton (15 body + 8 ski) and normalised coordinates for resolution independence. We distinguish between **static geometric** metrics and **dynamic time-series** metrics; the latter are more robust to camera perspective changes.

### Core Geometric Metrics

Computed by `metrics_computation.py`, averaged over the relevant phase window.

| Metric | Unit | Description |
|--------|------|-------------|
| `avg_v_style_front` | degrees | V-style ski opening angle (front camera) |
| `avg_v_style_back` | degrees | V-style ski opening angle (back camera) |
| `avg_body_ski_angle` | degrees | Inclination between body axis (neck→pelvis) and ski axis |
| `avg_symmetry_index_back` | 0–1 | Ski symmetry (0 = perfectly symmetric) |
| `takeoff_knee_angle` | degrees | Knee extension angle at the take-off frame |
| `takeoff_timing_offset` | frames | Delay between peak velocity and take-off frame |
| `takeoff_peak_velocity` | deg/s | Peak angular velocity at take-off |
| `telemark_scissor_mean` | normalised | Average horizontal leg separation at landing |
| `telemark_stability` | degrees | Standard deviation of scissor distance during landing |

### Time-Series Dynamic Metrics

Computed by `test_timeseries_metrics.py` — focus on derivatives and variability for camera-robustness.

| Metric | Unit | Description |
|--------|------|-------------|
| `knee_peak_velocity` | deg/s | Maximum knee extension velocity during take-off |
| `flight_std` | degrees | Std deviation of BSA during flight (overall stability) |
| `flight_jitter` | degrees | Mean frame-to-frame BSA variation (micro-instability) |
| `flight_mean_bsa` | degrees | Average body-ski angle during flight |
| `landing_hip_velocity` | normalised | Hip descent velocity after landing (impact intensity) |
| `landing_knee_compression` | degrees | Knee flexion range during landing absorption |

### Design Rationale, Robustness to Camera Perspective

Since all data is 2D, absolute angles can be distorted by camera viewpoint. We prioritise:

| Metric Type | Robustness | Why |
|-------------|------------|-----|
| **Velocities** (derivatives) | High | Rate of change is less affected by projection |
| **Std deviations** | High | Measures relative variation, not absolute position |
| **Normalised ratios** | Medium | Ratios cancel out some projection effects |
| **Absolute angles** | Low | Directly affected by camera angle |

---

## 🖼️ Metric Visualizations

The `metrics_visualizer.py` script draws metric-specific overlays on the original frames and supports five metric types:

<!-- ════════════════════════════════════════════════════════════════
     📸 INSERT YOUR IMAGES HERE
     Replace the GitHub `user-attachments/assets/...` URLs below
     with the actual URLs after uploading screenshots to GitHub.
     
     Recommended: pick one representative frame per metric, e.g.
       - Take-off knee angle  → a frame near take-off
       - Body-ski angle       → a mid-flight frame
       - V-style angle        → a front-view flight frame
       - Symmetry index       → a back-view flight frame
       - Telemark scissor     → a landing frame
     
     Source frames are in:
       metrics/metrics_visualizations/frame_overlays/<jump_id>/<metric_name>/
     ════════════════════════════════════════════════════════════════ -->

<table border="0">
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/e7de4198-a349-4cc4-ad77-8417c19274e9" width="100%"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/979ac743-5b81-496f-addb-45aa4e77a31b" width="100%"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/d5df2001-f4b6-431c-bb9f-784d43e4ef5d" width="100%"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/cec4f3a2-9718-4f67-b9e1-3484ef398839" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>Take-off Knee Angle</b></td>
    <td align="center"><b>Body-Ski Angle</b></td>
    <td align="center"><b>V-Style Angle</b></td>
    <td align="center"><b>Symmetry Index</b></td>
  </tr>
</table>

---

## 🔬 Analysis Modules

### 1. Core Metrics Computation (`core_metrics/`)

Computes per-frame geometric metrics from the normalised keypoints and aggregates them into per-jump summaries.

Validity filters are applied to discard physically impossible values (e.g. angles outside [0°, 180°]).

### 2. Time-Series Metrics (`core_metrics/timeseries_metrics/`)

Computes dynamic metrics from frame-to-frame changes: knee extension velocity at take-off, body-ski angle stability during flight, and hip/knee dynamics at landing.

### 3. Data Quality Check (`data_quality/`)

Validates all computed metrics in two passes:

1. **Physical range check**, flags values outside plausible biomechanical ranges (e.g. knee angle > 180°)
2. **Statistical outlier detection**, flags values with Z-score > 3


### 4. Correlation Analysis (`correlations/`)

Merges all metrics with performance scores from `JP_data.csv` (judge marks, athlete score and distance). Computes **Style_Score** (mean of middle 3 judges) and **Physical_Score** (AthleteScore − Style_Score). Calculates Pearson *r*, Spearman *ρ*, p-values, 95% confidence intervals, and R² for each metric–score pair.


### 5. Profile Analysis (`profile_analysis/`)

Segments each jump into phases (takeoff, early/mid/late flight, landing) and computes phase-specific metrics. Compares **Top 5** vs **Bottom 5** jumps (ranked by Style_Score) by normalising their BSA flight curves to a common timeline and plotting mean ± std.


### 6. Style Penalty Model (`style_penalty_model/`)

A **Ridge regression** model that predicts how many style points a jumper loses based on three biomechanical features:

| Feature | Weight | Interpretation |
|---------|--------|----------------|
| `telemark_scissor_mean` | −0.56 (45%) | Better telemark → less penalty |
| `flight_std` | +0.44 (36%) | More unstable flight → more penalty |
| `landing_hip_velocity` | −0.24 (19%) | Harder landing → more penalty |

Training uses **Leave-One-Out cross-validation** for unbiased evaluation on the small dataset, composed of 32 samples.

---

## 📖 How to Interpret Results

### Reading `metrics_summary_per_jump.csv`

Each row is one jump. Look for:
- **V-style angles** in the 30°–45° range → good aerodynamic position
- **BSA** in the 5°-20° range → body well aligned with skis
- **Symmetry index** close to 0 → skis are symmetric
- **Takeoff knee angle** close to 180° → full extension at take-off
- **Telemark scissor** around 0.05–0.25 → proper landing leg separation

### Reading `correlations_detailed.csv`

- **`pearson_r`**: linear correlation (−1 to +1). Values > 0.3 or < −0.3 are noteworthy.
- **`pearson_p`**: p-value. Values < 0.05 are statistically significant; < 0.10 are marginal.

---

## ⚠️ Main Limitations

1. **Small dataset**: only 32 annotated jumps, limiting statistical power. Some correlations may not reach significance.
2. **Low performance heterogeneity**:  all jumps are winners (or podiums) in FIS World Cup, so differences between athletes are small, and there is really low variance between jumps. A more diverse dataset would yield stronger signals.
3. **2D and perspective**: metrics are computed from multi-camera views. Cameras are not calibrated, the videos are taken in different competition, on different hills and on different HS (hill size). That means we have different perspective and different views from the cameras. With the 2D perspective we are not able to extract completely reliable and robus metrics.
4. **Style penalty model performance**: the Ridge model achieves R² < 0 on LOO-CV, meaning it underperforms a simple mean prediction. This is expected given the limited data and feature set, the formula should be interpreted as directional insight, not as an accurate predictor.

---

## 🔗 Related

- [Main project README](../README.md), full project overview
- [Dataset](../dataset/) — annotated keypoints, jump phases, and athlete data
