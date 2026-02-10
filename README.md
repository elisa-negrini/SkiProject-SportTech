# Ski Jump Pose Estimation 🎿

> From keypoint annotation to metrics and score correlation: developing a new dataset for ski pose estimation and performance analysis.

<!-- 📸 SUGGESTED: Add a hero image or GIF of an annotated ski jump here -->
<!-- ![Annotated Jump Example](docs/images/hero_demo.gif) -->

---

## 📖 Project Description

This project presents a comprehensive system for **2D metrics extraction from ski jumper skeletons (+skis)**, designed to analyze ski jumping performance through computer vision and biomechanical analysis.

### Key Features

- **Custom 23-Keypoint Skeleton**: Body joints + ski keypoints for complete pose representation
- **Bio-mechanical Metrics**: Joint angles, body alignment, skeletal positioning analysis
- **Ski Jump-Specific Metrics**: V-style angle, body-ski inclination, telemark evaluation
- **Jump Performance Analysis**: Comprehensive scoring and style penalty prediction
- **Deep Learning Model**: Transformer-based ski reconstruction from body pose (SkiPoseModel)
- **Interactive Dashboard**: Streamlit-based visualization and exploration tool

### Future Applications

| # | Application |
|---|-------------|
| 01 | **Fan Engagement Enhancement** - Real-time jump analysis for broadcasts |
| 02 | **Coaching Tool** - Detailed biomechanical feedback for athletes |

---

## 📁 Project Structure

```
SkiProject-SportTech/
│
├── 📄 README.md                           # This file
├── 📄 requirements.txt                    # Python dependencies
├── 📄 .gitignore                          # Git ignore rules
│
├── 📂 annotation_preprocessing/           # Annotation pipeline from Roboflow
│   ├── main.py                            # Master workflow orchestrator
│   └── utils/                             # Processing utilities
│
├── 📂 dashboard/                          # Streamlit interactive dashboard
│   ├── Dashboard.py                       # Main dashboard entry point
│   └── pages/                             # Dashboard pages (Gallery, Metrics)
│
├── 📂 dataset/                            # Main dataset folder
│   ├── frames/                            # Video frames (JP0001-JP0045)
│   ├── annotations/                       # Processed COCO annotations
│   ├── keypoints_dataset.csv              # Normalized keypoints dataset
│   ├── jump_phases_SkiTB.csv              # Jump phase annotations
│   └── JP_data.csv                        # Athlete & jump metadata
│
├── 📂 metrics/                            # Metrics computation & analysis
│   ├── core_metrics/                      # Geometric metrics per frame/jump
│   ├── timeseries_analysis/               # Temporal patterns & clustering
│   ├── correlations/                      # Score correlation analysis
│   ├── style_penalty_model/               # ML models for style prediction
│   └── visualizations/                    # Generated plots and charts
│
└── 📂 SkiPoseModel/                       # Deep learning ski reconstruction
    ├── main.py                            # Training/inference entry point
    ├── model.py                           # Transformer-based network
    └── datamodule.py                      # Data loading utilities
```

---

## 💾 Dataset & Model Checkpoints Download

### Dataset Description

Our dataset is built upon the **[Ski-TB Dataset](https://cvlab.epfl.ch/research/datasets/ski-tb/)** with custom **23-keypoint skeleton annotations**:

| Component | Description |
|-----------|-------------|
| **Jumps** | 45 professional ski jumps (JP0001 - JP0045) |
| **Athletes** | International FIS World Cup competitors (Men & Women) |
| **Keypoints** | 23 points: 17 body joints + 6 ski points |
| **Annotations** | COCO-format keypoint annotations with interpolation |
| **Phases** | Take-off, V-style, Flight, Landing, Telemark |
| **Metadata** | Athlete info, scores, judges' evaluations, hill data |

### 23-Keypoint Skeleton Model

<!-- 📸 SUGGESTED: Add skeleton diagram image here -->
<!-- ![23-Keypoint Skeleton](docs/images/skeleton_model.png) -->

| Body Part | Keypoints |
|-----------|-----------|
| **Head & Neck** | 1-Head, 2-Neck |
| **Right Arm** | 3-R_Shoulder, 4-R_Elbow, 5-R_Wrist |
| **Left Arm** | 6-L_Shoulder, 7-L_Elbow, 8-L_Wrist |
| **Torso** | 9-Center_Pelvis |
| **Right Leg** | 17-R_Hip, 18-R_Knee, 19-R_Ankle, 20-R_Foot |
| **Left Leg** | 10-L_Hip, 11-L_Knee, 12-L_Ankle, 13-L_Foot |
| **Right Ski** | 21-R_Ski_Binding, 22-R_Ski_Tail, 23-R_Ski_Tip |
| **Left Ski** | 14-L_Ski_Binding, 15-L_Ski_Tail, 16-L_Ski_Tip |

### Download Links

> ⚠️ **Note**: The dataset and model checkpoints are not included in the repository due to size constraints.

| Resource | Link | Size | Description |
|----------|------|------|-------------|
| **Full Dataset** | LINK DI GOOGLE DRIVE QUI | ~X GB | Frames, annotations, metadata |
| **Model Checkpoints** | LINK DI GOOGLE DRIVE QUI | ~X MB | Pre-trained SkiPoseModel weights |

**After downloading, place files as follows:**
```
SkiProject-SportTech/
├── dataset/
│   ├── frames/           ← Extract frames here
│   ├── annotations/      ← Extract annotations here
│   └── *.csv             ← Place CSV files here
│
└── SkiPoseModel/
    └── checkpoints/      ← Place model checkpoints here
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python**: 3.9+
- **GPU** (optional): NVIDIA GPU with CUDA 11.8+ for SkiPoseModel training

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

# 4. Download dataset and checkpoints (see links above)
# 5. Verify setup
python -c "import torch; print('PyTorch:', torch.__version__)"
```

---

## ▶️ How to Run the Project

### Quick Start Commands

| Task | Command |
|------|---------|
| **Run Dashboard** | `streamlit run dashboard/Dashboard.py` |
| **Process Annotations** | `python annotation_preprocessing/main.py` |
| **Compute Metrics** | `python metrics/core_metrics/metrics_computation.py` |
| **Train SkiPoseModel** | `python SkiPoseModel/main.py --mode train` |
| **Inference SkiPoseModel** | `python SkiPoseModel/main.py --mode test` |

### Full Pipeline Workflow

```bash
# Step 1: Process raw annotations (filtering, interpolation, normalization)
python annotation_preprocessing/main.py

# Step 2: Compute biomechanical metrics
python metrics/core_metrics/metrics_computation.py

# Step 3: Explore results in dashboard
streamlit run dashboard/Dashboard.py

# Step 4 (Optional): Train ski reconstruction model
cd SkiPoseModel
python main.py --mode train
```

---

## 📂 Folder Descriptions

### `annotation_preprocessing/`

Pipeline for processing raw annotations from Roboflow to interpolated, normalized keypoints.

| File | Purpose |
|------|---------|
| `main.py` | Interactive workflow orchestrator |
| `utils/annotation_manager.py` | Extract jump-specific annotations from COCO file |
| `utils/box_filter.py` | Filter and validate bounding boxes |
| `utils/interpolator.py` | Linear interpolation between annotated frames |
| `utils/normalizer.py` | Normalize keypoints to bounding box coordinates |
| `utils/visualizer.py` | Generate annotated images and videos |

**Workflow:**
1. Extract annotations by jump tag
2. Filter bounding boxes
3. Interpolate keypoints between frames
4. Normalize to consistent coordinate system
5. Generate visualization videos

<!-- 📸 SUGGESTED: Add annotated jump GIF here -->
<!-- ![Annotation Pipeline Output](docs/images/annotation_demo.gif) -->

---

### `dataset/`

Main data storage containing frames, annotations, and metadata.

| File/Folder | Description |
|-------------|-------------|
| `frames/JP00XX/` | Raw video frames organized by jump ID |
| `annotations/JP00XX/` | Processed COCO annotations + visualizations |
| `keypoints_dataset.csv` | Normalized keypoints ready for metrics computation |
| `jump_phases_SkiTB.csv` | Frame ranges for each jump phase (take-off, v-style, flight, landing, telemark) |
| `JP_data.csv` | Athlete metadata: name, nationality, scores, judges' evaluations, hill info |

**Dataset Statistics:**
- 45 competition jumps
- ~30 FPS video extraction
- Mixed hill types: Normal (K90), Large (K120), Flying (K200)

---

### `dashboard/`

Interactive Streamlit application for data exploration and analysis.

| Page | Description |
|------|-------------|
| **Dashboard.py** | Home page with project overview |
| **Gallery Explorer** | Browse frames with skeleton overlay, filter by jump/phase |
| **Metric Analysis** | Interactive charts for biomechanical metrics comparison |

**Run Dashboard:**
```bash
streamlit run dashboard/Dashboard.py
```

<!-- 📸 SUGGESTED: Add dashboard screenshot here -->
<!-- ![Dashboard Screenshot](docs/images/dashboard_preview.png) -->

---

### `metrics/`

Comprehensive biomechanical metrics computation and analysis.

| Subfolder | Content |
|-----------|---------|
| `core_metrics/` | Per-frame geometric metrics (angles, positions) |
| `timeseries_analysis/` | Temporal dynamics (velocity, stability, jitter) |
| `correlations/` | Statistical correlation with judges' scores |
| `style_penalty_model/` | ML models predicting style penalties |
| `visualizations/` | Generated charts and plots |

**Key Metrics Computed:**

| Category | Metrics |
|----------|---------|
| **V-Style** | Ski opening angle (front/back view) |
| **Body-Ski** | Inclination angle during flight |
| **Take-off** | Knee angle, extension velocity |
| **Flight** | Stability (std), jitter, smoothness |
| **Landing** | Hip velocity, knee compression, telemark offset |

---

### `SkiPoseModel/`

Deep learning model for ski position reconstruction from body pose.

**Architecture:** Transformer-based network (PyTorch Lightning)

**Task:** Given body keypoints (with skis masked), predict complete skeleton including ski positions.

| File | Purpose |
|------|---------|
| `main.py` | Training/testing/demo entry point |
| `model.py` | AdaptationNetwork (Lightning module) |
| `datamodule.py` | Dataset loading and preprocessing |
| `transformer.py` | Transformer architecture blocks |
| `preprocess.py` | COCO JSON → pickle conversion |
| `postprocess_visualize.py` | Visualization utilities |

**Training:**
```bash
cd SkiPoseModel
python preprocess.py  # Prepare data
python main.py --mode train
```

---

## 🎨 Visualization Features

### Skeleton Color Coding

| Body Part | Color |
|-----------|-------|
| Head & Neck | Cyan |
| Right Arm | Light Blue |
| Left Arm | Orange |
| Torso | Black |
| Left Leg | Yellow |
| Right Leg | Purple |
| Skis | Pink |

Each keypoint is labeled with its number (1-23) for easy identification.

---

## 👥 Team

| Name | Email | ID |
|------|-------|-----|
| Diego Conti | diego.conti@studenti.unitn.it | 257632 |
| Elisa Negrini | elisa.negrini@studenti.unitn.it | 258422 |
| Federico Molteni | federico.molteni@studenti.unitn.it | 243030 |

---

## 📜 License

*To be defined*

---

## 🙏 Acknowledgments

- **[Ski-TB Dataset](https://cvlab.epfl.ch/research/datasets/ski-tb/)** - Base dataset for ski jumping videos
- **[Roboflow](https://roboflow.com/)** - Annotation platform
- **[PyTorch Lightning](https://lightning.ai/)** - Deep learning framework
- **[Streamlit](https://streamlit.io/)** - Dashboard framework

---

**Sport Tech 2025/2026** - University of Trento

