# Ski Jump Pose Estimation 🎿

> From keypoint annotation to biomechanical metrics and score correlation — building a new dataset and analysis pipeline for ski jumping pose estimation and performance evaluation.

<!-- 📸 Add a hero image or GIF of an annotated ski jump here -->
<!-- ![Annotated Jump Example](docs/images/hero_demo.gif) -->

---

## 📖 Project Overview

This project develops an end-to-end system for **2D metrics extraction from ski jumper skeletons (+ skis)**, starting from raw competition footage from the [Ski-TB Dataset](https://cvlab.epfl.ch/research/datasets/ski-tb/).

The main objectives are:

1. **Annotate a custom ski jumping dataset** — manually label a 23-keypoint skeleton (body + skis) on competition videos and enrich them through interpolation and normalization.
2. **Extract biomechanical metrics** — compute 2D geometric and dynamic metrics (joint angles, V-style, body-ski inclination, flight stability, landing quality, …) and correlate them with judges' scores.
3. **Train a Ski Pose Estimation model** — use a transformer-based deep learning model to predict ski positions given only the body skeleton.

### Future Applications

| # | Application |
|---|-------------|
| 01 | **Fan Engagement Enhancement** — Real-time jump analysis for broadcasts, providing viewers with instant visual feedback on jump technique and performance metrics |
| 02 | **Coaching Tool** — Detailed biomechanical feedback for athletes and coaching staff, enabling frame-by-frame technique comparison and improvement tracking |

---

## 🏔️ Milestone 1 — Dataset Annotation

### Overview

We annotated **32 ski jumps** from the [Ski-TB Dataset](https://cvlab.epfl.ch/research/datasets/ski-tb/) using a custom **23-keypoint skeleton model** (17 body joints + 6 ski keypoints). All annotations were created with **[Roboflow](https://roboflow.com/)**.

<!-- 📸 Add skeleton diagram + Roboflow logo here -->
<!-- ![23-Keypoint Skeleton](docs/images/skeleton_model.png) -->
<!-- ![Roboflow Logo](docs/images/roboflow_logo.png) -->

### 23-Keypoint Skeleton

| Body Part | Keypoints |
|-----------|-----------|
| **Head & Neck** | 1-Head, 2-Neck |
| **Right Arm** | 3-R_Shoulder, 4-R_Elbow, 5-R_Wrist |
| **Left Arm** | 6-L_Shoulder, 7-L_Elbow, 8-L_Wrist |
| **Torso** | 9-Center_Pelvis |
| **Left Leg** | 10-L_Hip, 11-L_Knee, 12-L_Ankle, 13-L_Foot |
| **Left Ski** | 14-L_Ski_Binding, 15-L_Ski_Tail, 16-L_Ski_Tip |
| **Right Leg** | 17-R_Hip, 18-R_Knee, 19-R_Ankle, 20-R_Foot |
| **Right Ski** | 21-R_Ski_Binding, 22-R_Ski_Tail, 23-R_Ski_Tip |

### Annotation Pipeline

Each jump video contains approximately **350 frames**. Out of these, around **60 key frames** are manually annotated in Roboflow in COCO format. The remaining frames are then completed through the following automated pipeline:

1. **Extract** — parse jump-specific annotations from the exported COCO file.
2. **Filter** — validate and filter bounding boxes, removing erroneous detections.
3. **Interpolate** — linearly interpolate keypoints between annotated frames to obtain annotations for all ~350 frames.
4. **Normalize** — normalize keypoint coordinates relative to the bounding box, making them resolution- and position-independent for downstream metric computation.
5. **Visualize** — generate annotated images and overlay videos for quality inspection.

### Folder Structure

```
annotation_preprocessing/
├── main.py                        # Master workflow orchestrator
└── utils/
    ├── annotation_manager.py      # Extract jump-specific annotations from COCO file
    ├── box_filter.py              # Filter and validate bounding boxes
    ├── interpolator.py            # Linear interpolation between annotated frames
    ├── normalizer.py              # Normalize keypoints to bounding box coordinates
    └── visualizer.py              # Generate annotated images and videos
```

### Output — Dataset Folder

The annotation pipeline outputs data into the `dataset/` folder, organized as follows:

| Path | Description |
|------|-------------|
| `frames/JP00XX/` | Raw video frames organized by jump ID |
| `annotations/JP00XX/` | Processed COCO annotations + visualization overlays |
| `keypoints_dataset.csv` | Normalized keypoints for all frames, ready for metrics computation |
| `jump_phases_SkiTB.csv` | Frame ranges for each jump phase (take-off, v-style, flight, landing, telemark) |
| `JP_data.csv` | Athlete metadata: name, nationality, scores, judges' evaluations, hill info |

---

## 📐 Milestone 2 — Biomechanical Metrics

### Overview

Starting from the annotated and normalized keypoints, we computed a set of **2D biomechanical metrics** to quantitatively describe each jump. These metrics account for the inherent limitations of a 2D perspective (e.g., foreshortening, camera angle variability) by favoring dynamic measures (velocities, standard deviations) over static absolute angles where possible.

### Folder Structure

```
metrics/
├── core_metrics/                  # Per-frame geometric metrics (angles, positions)
│   ├── metrics_computation.py     # Main metrics computation script
│   ├── metrics_per_frame.csv      # Metrics for each frame
│   ├── metrics_summary_per_jump.csv # Aggregated metrics per jump
│   └── timeseries_metrics/        # Time-series dynamics (velocity, jitter)
├── correlations/                  # Statistical correlation with judges' scores
│   ├── correlation_analysis.py
│   └── *.csv / *.png              # Results and heatmaps
├── data_quality/                  # Outlier detection and data validation
├── metrics_visualizations/        # Overlay visualizations on frames
├── profile_analysis/              # Top vs. flop athlete comparisons
└── style_penalty_model/           # ML model predicting style penalties
```

### Key Metrics

| Category | Metrics | Description |
|----------|---------|-------------|
| **V-Style** | `avg_v_style_front`, `avg_v_style_back` | Ski opening angle from front and back views |
| **Body-Ski Angle** | `avg_body_ski_angle` | Inclination between body axis and ski axis during flight |
| **Take-off** | `takeoff_knee_angle`, `knee_peak_velocity` | Knee extension angle and explosive velocity at jump |
| **Flight Stability** | `flight_std`, `flight_jitter` | Variability and micro-instability of body-ski angle |
| **Landing** | `landing_hip_velocity`, `landing_knee_compression`, `telemark_offset` | Landing softness and telemark quality |

### Metric Visualizations

<!-- 📸 Add metric visualization images here, one for each -->
<!-- ![V-Style Front Angle](docs/images/v_style_front.png) -->
<!-- ![V-Style Back Angle](docs/images/v_style_back.png) -->
<!-- ![Body-Ski Angle](docs/images/body_ski_angle.png) -->

### ⚠️ Disclaimer on Results

We are aware that some of the results obtained from the metrics and correlation analyses are **not all statistically significant**, for two main reasons:

1. **Small dataset** — with only 32 annotated jumps, the sample size limits the statistical power of any analysis.
2. **Low performance heterogeneity** — all jumps in the dataset come from top-level international FIS World Cup competitions. Since all athletes perform at a very high level, it is inherently difficult to distinguish between "good" and "less good" performances, making predictions and meaningful analyses harder. A more heterogeneous dataset (e.g., including amateur-level jumps) would likely yield more significant and differentiated results.

---

## 🤖 Milestone 3 — Ski Pose Estimation Model

### Overview

The **SkiPoseModel** is a transformer-based deep learning model originally introduced in [this paper](https://github.com/kaulquappe23/ski-pose-prediction). Its goal is to **predict the position of the 6 ski keypoints given only the body skeleton** of the jumper (with ski joints masked during training).

We adapted the model to our custom 23-keypoint dataset and trained it on the following data split:

| Split | Samples |
|-------|---------|
| Train | 7,729 |
| Val | 1,656 |
| Test | 1,657 |

### Folder Structure

```
SkiPoseModel/
├── main.py                      # Training / testing / demo entry point
├── model.py                     # AdaptationNetwork (PyTorch Lightning module)
├── datamodule.py                # SkijumpDataModule and SkijumpDataset
├── transformer.py               # Transformer architecture blocks
├── preprocess.py                # COCO JSON → pickle preprocessing
├── postprocess_visualize.py     # Visualization & ski linearization
├── domainadapt_flags.py         # Configuration flags
├── requirements.txt             # Model-specific dependencies
├── dataset/                     # Raw dataset (COCO JSON + frames)
├── dataset_preprocessed/        # Preprocessed splits (train.pkl, val.pkl, test.pkl)
└── results/                     # Predictions, checkpoints, visualizations
    ├── checkpoints/             # Saved model weights
    ├── test_results.pkl         # Raw test predictions
    └── test_results_linearized.pkl  # Post-processed (linearized skis)
```

### Prediction & Post-Processing

The model predicts the 6 ski keypoints (3 per ski: binding, tail, tip). After inference, a **PCA-based linearization** step is applied to force the predicted ski points onto a straight line, producing more physically plausible results.

<!-- 📸 Add 4-5 side-by-side images of GT skeleton vs. predicted skeleton here -->
<!-- ![Prediction Examples](docs/images/skiposemodel_predictions.png) -->

---

## 🚀 Installation & Setup

### Prerequisites

- **Python**: 3.9+
- **GPU** (required for SkiPoseModel): NVIDIA GPU with CUDA 11.8+

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/SkiProject-SportTech.git
cd SkiProject-SportTech

# 2. Create virtual environment
python -m venv sport_tech_env

# Windows
sport_tech_env\Scripts\activate

# Linux/Mac
source sport_tech_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python -c "import torch; print('PyTorch:', torch.__version__)"
```

---

## ▶️ How to Run

### 1. Annotation Preprocessing

Place your Roboflow COCO export inside `annotation_preprocessing/raw_annotations/`, then run:

```bash
python annotation_preprocessing/main.py
```

This will extract, filter, interpolate, normalize, and visualize the annotations. Output is saved to the `dataset/` folder.

### 2. SkiPoseModel (GPU Required)

> ⚠️ Training and inference require an **NVIDIA GPU with CUDA support**.

```bash
cd SkiPoseModel

# Preprocess dataset (COCO JSON → pickle)
python preprocess.py

# Train the model
python main.py --mode train

# Test the model
python main.py --mode test

# Post-process & visualize predictions (ski linearization)
python postprocess_visualize.py
```

### 3. Metrics Computation

Compute biomechanical metrics and run analyses using the scripts in the `metrics/` folder:

```bash
# Core geometric metrics
python metrics/core_metrics/metrics_computation.py

# Correlation analysis with judges' scores
python metrics/correlations/correlation_analysis.py

# Data quality checks
python metrics/data_quality/data_quality_check.py

# Profile analysis (top vs. flop)
python metrics/profile_analysis/profile_analysis.py

# Style penalty model
python metrics/style_penalty_model/style_penalty_model.py
```

### 4. Interactive Dashboard (Streamlit)

The dashboard provides two main screens:

- **Gallery Explorer** — browse and filter the dataset by jump, athlete, and phase; explore annotated frames with skeleton overlays.
- **Metric Analysis** — visualize metrics filtered by athlete and metric type, with summary statistics.

```bash
streamlit run dashboard/Dashboard.py
```

<!-- 📸 Add a short demo video/GIF for each dashboard screen here -->
<!-- ![Gallery Explorer Demo](docs/videos/gallery_explorer_demo.gif) -->
<!-- ![Metric Analysis Demo](docs/videos/metric_analysis_demo.gif) -->

---

## 💾 Dataset & Checkpoints Download

> ⚠️ **Note**: The dataset and model checkpoints are not included in the repository due to size constraints.

| Resource | Link | Size | Description |
|----------|------|------|-------------|
| **Full Dataset** | *Google Drive link here* | ~X GB | Frames, annotations, metadata |
| **Model Checkpoints** | *Google Drive link here* | ~X MB | Pre-trained SkiPoseModel weights |

**After downloading, place files as follows:**
```
SkiProject-SportTech/
├── dataset/
│   ├── frames/           ← Extract frames here
│   ├── annotations/      ← Extract annotations here
│   └── *.csv             ← Place CSV files here
│
└── SkiPoseModel/
    └── results/
        └── checkpoints/  ← Place model checkpoints here
```

---

## 👥 Team

| Name | Email | ID |
|------|-------|-----|
| Diego Conti | diego.conti@studenti.unitn.it | 257632 |
| Elisa Negrini | elisa.negrini@studenti.unitn.it | 258422 |
| Federico Molteni | federico.molteni@studenti.unitn.it | 243030 |

---

## 🙏 Acknowledgments

- **[Ski-TB Dataset](https://cvlab.epfl.ch/research/datasets/ski-tb/)** — Base dataset for ski jumping videos
- **[Roboflow](https://roboflow.com/)** — Annotation platform
- **[Ski Pose Prediction](https://github.com/kaulquappe23/ski-pose-prediction)** — Original SkiPoseModel paper and code
- **[PyTorch Lightning](https://lightning.ai/)** — Deep learning framework
- **[Streamlit](https://streamlit.io/)** — Dashboard framework

---

**Sport Tech 2025/2026** — University of Trento

