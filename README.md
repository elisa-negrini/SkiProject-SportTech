<table><tr><td>

# Ski Jump Pose Estimation 🎿

> From keypoint annotation to biomechanical metrics and score correlation — building a new dataset and analysis pipeline for ski jumping pose estimation and performance evaluation.

</td><td>

<img width="220" alt="skeleton" src="https://github.com/user-attachments/assets/dd336d0f-baa0-465f-98ce-3f10ff7773e0" />

</td></tr></table>


This project develops an end-to-end system for **2D metrics extraction from ski jumper skeletons (+ skis)**, starting from raw competition footage from the [Ski-TB Dataset](https://cvlab.epfl.ch/research/datasets/ski-tb/).

The main objectives are:

1. **Annotate a custom ski jumping dataset** — manually label a 23-keypoint skeleton (body + skis) on competition videos and enrich them through interpolation and normalization.
2. **Extract biomechanical metrics** — compute 2D geometric and dynamic metrics (joint angles, V-style, body-ski inclination, flight stability, landing quality, …) and correlate them with judges' scores.
3. **Train a Ski Pose Estimation model** — use a transformer-based deep learning model to predict ski positions given only the body skeleton.

### Future Applications

**Fan Engagement Enhancement** — Real-time jump analysis for broadcasts, providing viewers with instant visual feedback on jump technique and performance metrics.

**Coaching Tool** — Detailed biomechanical feedback for athletes and coaching staff, enabling frame-by-frame technique comparison and improvement tracking.

---



## Objective 1 — Dataset Annotation &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![roboflow](https://github.com/user-attachments/assets/cce73566-6d18-4dfe-ada0-f013f580c3bc)



We annotated **32 ski jumps** from the [Ski-TB Dataset](https://cvlab.epfl.ch/research/datasets/ski-tb/) using a custom **23-keypoint skeleton model** (17 body joints + 6 ski keypoints). All annotations were created with **[Roboflow](https://roboflow.com/)**.

<!-- 📸 Add skeleton diagram + Roboflow logo here -->
<!-- ![23-Keypoint Skeleton](docs/images/skeleton_model.png) -->
<!-- ![Roboflow Logo](docs/images/roboflow_logo.png) -->



### Annotation Pipeline

Each jump video contains approximately **350 frames**. Out of these, around **60 key frames** are manually annotated in Roboflow in COCO format. The remaining frames are then completed through the following automated pipeline:

1. **Extract** — parse jump-specific annotations from the exported COCO file.
2. **Filter** — validate and filter bounding boxes, removing erroneous detections.
3. **Interpolate** — linearly interpolate keypoints between annotated frames to obtain annotations for all ~350 frames.
4. **Normalize** — normalize keypoint coordinates relative to the bounding box, making them resolution- and position-independent for downstream metric computation.
5. **Visualize** — generate annotated images and overlay videos for quality inspection.

### Folder Structure & Output

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

The annotation pipeline outputs data into the `dataset/` folder:

```
dataset/
├── frames/JP00XX/                 # Raw video frames organized by jump ID
├── annotations/JP00XX/            # Processed COCO annotations + visualization overlays
├── keypoints_dataset.csv          # Normalized keypoints for all frames, ready for metrics computation
├── jump_phases_SkiTB.csv          # Frame ranges for each jump phase (take-off, v-style, flight, landing, telemark)
└── JP_data.csv                    # Athlete metadata: name, nationality, scores, judges' evaluations, hill info
```

---

##  Objective 2 — Biomechanical Metrics


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

<table border="0">
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/3745bbe6-a11f-42e0-a7d5-743ccaa7030f" width="100%"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/979ac743-5b81-496f-addb-45aa4e77a31b" width="100%"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/d5df2001-f4b6-431c-bb9f-784d43e4ef5d" width="100%"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/cec4f3a2-9718-4f67-b9e1-3484ef398839" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>Take Off knee angle</b></td>
    <td align="center"><b>Body Ski Angle</b></td>
    <td align="center"><b>V style Angle</b></td>
    <td align="center"><b>Symmetry Index</b></td>
  </tr>
</table>

### ⚠️ Disclaimer on Results

We are aware that some of the results obtained from the metrics and correlation analyses are **not all statistically significant**, for two main reasons:

1. **Small dataset** — with only 32 annotated jumps, the sample size limits the statistical power of any analysis.
2. **Low performance heterogeneity** — all jumps in the dataset come from top-level international FIS World Cup competitions. Since all athletes perform at a very high level, it is inherently difficult to distinguish between "good" and "less good" performances, making predictions and meaningful analyses harder. A more heterogeneous dataset (e.g., including amateur-level jumps) would likely yield more significant and differentiated results.

---

##  Objective 3 — Ski Pose Estimation Model


The **SkiPoseModel** is a transformer-based deep learning model originally introduced in [this paper](https://github.com/kaulquappe23/ski-pose-prediction). Its goal is to **predict the position of the 8 ski keypoints given only the body skeleton** of the jumper (with ski joints masked during training).

We adapted the model to our custom 23-keypoint dataset and trained it on a total of 11,042 samples (7,729 train, 1,656 validation, 1,657 test).

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

The model predicts the 6 ski keypoints (4 per ski). After inference, a **PCA-based linearization** step is applied to force the predicted ski points onto a straight line, producing more physically plausible results. Here are some visualizations of the results.
<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td><img src="https://github.com/user-attachments/assets/fdbc37a1-48df-4754-ac4a-919e245e46ae" width="100%" alt="test_0005"></td>
    <td><img src="https://github.com/user-attachments/assets/dfa237a2-e2e8-4446-bc5e-a9a290fe165f" width="100%" alt="test_0068"></td>
    <td><img src="https://github.com/user-attachments/assets/9541719f-b34d-4d69-8ffc-5103d63c8f66" width="100%" alt="test_0093"></td>
    <td><img src="https://github.com/user-attachments/assets/9c0caa91-4dfe-49d5-80ae-685d5417a9aa" width="100%" alt="test_0037"></td>
  </tr>
 
</table>



---

##  Installation & Setup

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

