# 📊 Metrics Documentation - Ski Jumping Analysis

This folder contains all computed metrics and analysis results for ski jumping biomechanics.

---

## 📁 Folder Structure

```
metrics/
├── README.md                          ← This file
├── advanced_metrics/                  ← Phase-specific advanced metrics
├── core_metrics/                      ← Basic per-frame geometric metrics
├── correlations/                      ← Correlation analysis with scores
├── data_quality/                      ← Outlier detection reports
├── models/                            ← ML models and predictions
├── phase_analysis/                    ← Phase-by-phase breakdown
├── timeseries_analysis/               ← Clustering and temporal patterns
├── timeseries_metrics/                ← Time-series based dynamics
└── visualizations/                    ← Plots and charts
```

---

## 🎯 Metric Categories

### 1. **Core Metrics** (`core_metrics/metrics_summary_per_jump.csv`)

Static geometric measurements averaged over flight.

| Metric | Unit | Description | Ideal Value |
|--------|------|-------------|-------------|
| `avg_v_style_front` | degrees | V-style opening angle (front view) | 30-45° |
| `avg_v_style_back` | degrees | V-style opening angle (back view) | 30-45° |
| `takeoff_knee_angle` | degrees | Knee angle at take-off moment | 160-180° (extended) |
| `avg_body_ski_angle` | degrees | Body-ski inclination (aerodynamic) | 10-20° |
| `avg_symmetry_index_back` | 0-1 | Ski symmetry (0=perfect, 1=asymmetric) | <0.1 |
| `avg_telemark_offset_x` | normalized | Horizontal leg separation in telemark | 0.05-0.15 |
| `flight_stability_std` | degrees | Standard deviation of body-ski angle | <3° (stable) |

**Source:** Computed from normalized keypoints in `keypoints_dataset.csv`

---

### 2. **Time-Series Metrics** (`timeseries_metrics/timeseries_summary.csv`)

Dynamic measurements focusing on **velocity** and **variability** (more robust to camera perspective).

#### 2.1 Take-Off Explosiveness

| Metric | Unit | Description | Interpretation |
|--------|------|-------------|----------------|
| `knee_peak_velocity` | deg/s | Maximum knee extension velocity | Higher = more explosive |
| `knee_angle_at_takeoff` | degrees | Knee angle at take-off frame | Context (not for models) |

**Why these?**
- `knee_peak_velocity` captures explosive power better than static angles
- Removed `knee_mean_velocity` and `knee_extension_range` (redundant, r > 0.8 with peak)

---

#### 2.2 Flight Stability (Frozenness)

| Metric | Unit | Description | Interpretation |
|--------|------|-------------|----------------|
| `flight_std` | degrees | Std dev of body-ski angle | Lower = more stable |
| `flight_jitter` | degrees | Frame-to-frame angle variation | Lower = smoother |
| `flight_mean_bsa` | degrees | Average body-ski angle | 10-20° optimal |

**Difference between `flight_std` and `flight_jitter`:**

```
flight_std = Overall variability (total range of movement)
flight_jitter = High-frequency tremors (micro-instability)

Example:
  Smooth opening: flight_std=3°, flight_jitter=0.5° ✅
  Unstable:       flight_std=3°, flight_jitter=2.5° ❌
```

**Removed features:**
- `flight_range`: Redundant with `flight_std` (r > 0.85)
- `flight_trend`: Low predictive power

---

#### 2.3 Landing Absorption

| Metric | Unit | Description | Interpretation |
|--------|------|-------------|----------------|
| `landing_hip_velocity` | m/s (norm) | Hip descent velocity after landing | Lower = softer ⭐ |
| `landing_knee_compression` | degrees | Knee flexion during absorption | 40-60° optimal |

**Why `landing_hip_velocity` is the best predictor:**
- Correlation with Style_Score: **r = -0.65** (strongest of all metrics!)
- Measures "hardness" of landing directly

**Removed features:**
- `landing_hip_drop`: Depends on jump height, not technique
- `landing_smoothness_score`: Engineered feature (model can learn this)

---

### 3. **Advanced Metrics** (`advanced_metrics/advanced_metrics_summary.csv`)

Phase-specific refined measurements.

| Metric | Unit | Description | Phase |
|--------|------|-------------|-------|
| `takeoff_timing_offset` | frames | Delay of peak velocity vs take-off frame | Take-off |
| `takeoff_peak_velocity` | deg/s | Peak angular velocity at take-off | Take-off |
| `telemark_scissor_mean` | normalized | Average scissor distance (height-normalized) | Landing |
| `telemark_stability` | degrees | Std dev of scissor position | Landing |

---

### 4. **Phase Analysis** (`phase_analysis/phase_metrics.csv`)

Detailed breakdown per jump phase:
- **In-run**: Approach dynamics
- **Take-off**: Explosive jump moment (±10 frames)
- **Early flight**: Stabilization after take-off
- **Mid flight**: Stable V-style position
- **Late flight**: Preparation for landing
- **Landing**: Telemark and absorption

**Use Case:** Identify which phase is an athlete's strength/weakness.

---

### 5. **Clustering** (`timeseries_analysis/`)

Groups jumps with similar flight trajectory patterns using Dynamic Time Warping (DTW).

| File | Description |
|------|-------------|
| `cluster_assignments_DTW.csv` | Cluster labels (standard DTW) |
| `cluster_assignments_CDTW.csv` | Cluster labels (constrained DTW, faster) |
| `clustering_comparison.txt` | Quality comparison (silhouette scores) |

**Results:**
- DTW Silhouette: 0.554 (better separation) ✅
- CDTW Silhouette: 0.376 (faster but less accurate)
- **Recommendation:** Use DTW for final analysis

---

### 6. **Correlations** (`correlations/`)

Statistical relationships between metrics and performance scores.

| File | Description |
|------|-------------|
| `correlations.csv` | Pearson/Spearman correlations for all metrics |
| `merged_scores_metrics.csv` | Combined dataset (metrics + Style/Physical scores) |

**Top Correlations with Style_Score:**
1. `landing_hip_velocity`: r = **-0.65** ⭐⭐⭐
2. `flight_std`: r = **-0.52** ⭐⭐
3. `knee_peak_velocity`: r = **0.48** ⭐⭐

---

### 7. **ML Models** (`models/`)

Predictive models for Style and Physical scores.

| File | Description |
|------|-------------|
| `ml_summary_report.txt` | Performance metrics for all models |
| `STYLE_PENALTY_FORMULA.txt` | Linear formula for style penalties |
| `feature_selection_report.txt` | Which features were used/excluded |
| `importance_*.csv` | Feature importance rankings |

**Excluded Features (to prevent bias/leakage):**
- Data leakage: `normalized_distance`, `AthleteDistance`, `DistancePoints`
- Demographic: `AthleteGender`, `AthleteNat`, `HillLocation`
- Redundant: `flight_range`, `flight_trend`, `knee_mean_velocity`, `knee_extension_range`, `landing_hip_drop`, `landing_smoothness_score`

---

### 8. **Data Quality** (`data_quality/`)

Outlier detection and validation.

| File | Description |
|------|-------------|
| `outliers_report.csv` | Jumps with physically impossible values |
| `warnings_report.csv` | Statistical outliers (Z-score > 3) |
| `data_quality_summary.txt` | Overview report |

---

## 📋 Final Feature List

After redundancy cleanup, these are the **recommended features** for modeling:

### Time-Series Features (7 total)
| Feature | Category | Importance |
|---------|----------|------------|
| `knee_peak_velocity` | Take-off | ⭐⭐⭐ |
| `knee_angle_at_takeoff` | Take-off | ⭐ (context only) |
| `flight_std` | Flight | ⭐⭐⭐ |
| `flight_jitter` | Flight | ⭐⭐ |
| `flight_mean_bsa` | Flight | ⭐ |
| `landing_hip_velocity` | Landing | ⭐⭐⭐ (best predictor!) |
| `landing_knee_compression` | Landing | ⭐⭐ |

### Core Features (from `metrics_summary_per_jump.csv`)
| Feature | Category | Importance |
|---------|----------|------------|
| `avg_v_style_front` | Flight | ⭐⭐⭐ |
| `avg_body_ski_angle` | Flight | ⭐⭐ |
| `avg_symmetry_index_back` | Flight | ⭐⭐ |
| `telemark_scissor_mean` | Landing | ⭐⭐ |
| `telemark_stability` | Landing | ⭐⭐ |

---

## 🔬 Robustness to Camera Perspective

| Metric Type | Perspective Robustness | Example |
|-------------|------------------------|---------|
| **Absolute Angles** | ⚠️ LOW | BSA affected by diagonal view |
| **Velocity (derivatives)** | ✅ HIGH | Knee velocity less distorted |
| **Std Dev (variability)** | ✅ HIGH | `flight_std` measures relative change |
| **Ratios (normalized)** | ✅ MEDIUM | `telemark_scissor` / hip_height |

**Recommendation:** Prefer dynamic metrics (velocity, std) over static angles for predictive models.

---

## 🚀 Recomputing Metrics

To regenerate all metrics after changes:

```bash
# 1. Core metrics (from keypoints)
python utils/metrics_calculator.py

# 2. Time-series metrics
python metrics/test_timeseries_metrics.py

# 3. Advanced metrics
python metrics/advanced_metrics.py

# 4. Correlations
python metrics/test_correlation_analysis.py

# 5. ML models
python metrics/ml_models.py

# 6. Phase analysis
python metrics/phase_analysis.py

# 7. Clustering
python metrics/timeseries_analysis.py

# 8. Data quality check
python metrics/data_quality_check.py

# 9. Visualizations
python metrics/visualizations/create_visualizations.py
```

---

## ❓ Glossary

### Angles
- **BSA (Body-Ski Angle)**: Angle between body axis (neck→pelvis) and ski axis
- **V-Style**: Opening angle between left and right ski
- **Telemark Scissor**: Horizontal leg separation during landing

### Dynamics
- **Velocity**: Rate of change (deg/frame → deg/s)
- **Jitter**: Frame-to-frame variation (high-frequency noise)
- **Std Dev**: Spread of values (stability indicator)

### Phases
- **Take-off**: Explosive jump moment (±10 frames from `take_off_frame`)
- **BSA Window**: `bsa_start` to `bsa_end` from `jump_phases_SkiTB.csv`
- **Landing**: From `landing` frame + 15 frames for absorption analysis

---

## 📈 Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2026-02-05 | Removed redundant features (6 total) |
| 1.5 | 2026-02-04 | Added CDTW clustering comparison |
| 1.0 | 2026-02-01 | Initial metrics implementation |

---

**Last Updated:** 2026-02-05  
**Metrics Version:** 2.0 (After redundancy cleanup)
