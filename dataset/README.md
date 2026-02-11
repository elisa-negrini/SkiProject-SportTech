# 📂 Dataset — Ski Jump Pose Annotations & Metadata

This folder contains the core dataset used across the entire project pipeline: annotation preprocessing, biomechanical metrics computation, SkiPoseModel training, and the interactive dashboard.

---

## 📁 Folder Structure

```
dataset/
├── frames/JP00XX/              # Raw video frames organized by jump ID
├── annotations/JP00XX/         # Processed COCO annotations + visualization overlays
├── keypoints_dataset.csv       # Normalized keypoints for all frames, ready for metrics computation
├── jump_phases_SkiTB.csv       # Frame ranges for each jump phase (take-off, v-style, flight, landing, telemark)
└── JP_data.csv                 # Athlete metadata: name, nationality, scores, judges' evaluations, hill info
```

---

## ⚠️ Data Availability

| Item | In GitHub repo? | Notes |
|------|:-:|-------|
| `keypoints_dataset.csv` | ✅ | Ready to use |
| `jump_phases_SkiTB.csv` | ✅ | Ready to use |
| `JP_data.csv` | ✅ | Ready to use |
| `annotations/` | ⚠️ Partial | Only COCO JSON files are on GitHub. Full annotations (including visualization overlays) are available on [Google Drive](https://drive.google.com/drive/folders/10cKnZdP3x-tIoTHMw_nk2GYT9gc-uXRN?usp=drive_link). To get visualizations, replace the `annotations/` folder from the repo with the one from Drive. |
| `frames/` | ❌ | Too large for GitHub. Download from [Google Drive](https://drive.google.com/drive/folders/10cKnZdP3x-tIoTHMw_nk2GYT9gc-uXRN?usp=drive_link) and place inside `dataset/`. |

---

## 📄 File Descriptions

### `keypoints_dataset.csv`

Normalized 2D keypoint coordinates for every annotated frame across all 32 jumps.

| Column | Description |
|--------|-------------|
| `jump_id` | Jump identifier (e.g. `JP0005`) |
| `frame_name` | Frame filename (e.g. `00176.jpg`) |
| `image_id` | COCO image ID |
| `kpt_{N}_x`, `kpt_{N}_y` | Normalized x/y coordinates of keypoint N (bounding-box relative, range 0–1) |
| `kpt_{N}_v` | Visibility flag for keypoint N (COCO convention: 0 = not labeled, 1 = occluded, 2 = visible) |

The file contains 23 keypoints per frame (15 body joints + 8 ski keypoints). This is the primary input for the `metrics/` pipeline and the `SkiPoseModel`.

---

### `jump_phases_SkiTB.csv`

Frame-level annotations for each jump phase, manually defined from video inspection.

| Column | Description |
|--------|-------------|
| `jump_id` | Jump identifier (e.g. `jump5`) |
| `take_off_measurable` | 1 if take-off is clearly visible, 0 otherwise |
| `take_off_frame` | Frame number of the take-off moment |
| `v_style_measurable` | 1 if V-style phase is measurable |
| `v_style_front_start/end` | Frame range for V-style from front camera view |
| `v_style_back_start/end` | Frame range for V-style from back camera view |
| `body_ski_measurable` | 1 if body-ski angle is measurable |
| `bsa_start/end` | Frame range for body-ski angle measurement |
| `landing` | Frame number of the landing moment |
| `telemark_measurable` | 1 if telemark phase is measurable |
| `telemark_start/end` | Frame range for telemark phase |
| `telemark_f_l` | Camera view during telemark: `f` (front) or `l` (lateral) |
| `notes` | Optional notes (e.g. weather, annotation difficulties) |

---

### `JP_data.csv`

Athlete and competition metadata for each jump, sourced from official FIS results.

| Column | Description |
|--------|-------------|
| `ID` | Jump identifier (e.g. `JP0001`) |
| `AthleteName`, `AthleteSurname` | Athlete full name |
| `AthleteNat` | Nationality (IOC code) |
| `AthleteGender` | `M` or `W` |
| `AthleteSpeed` | In-run speed (km/h) |
| `AthleteScore` | Total competition score |
| `AthleteDistance` | Jump distance (meters) |
| `AthleteJdgA–E` | Individual judges' style scores |
| `AthleteGate` | Start gate number |
| `HillLocation`, `HillNat` | Hill name and country |
| `HillK`, `HillHS` | K-point and hill size (meters) |
| `Time`, `Weather` | Time of day and weather conditions |
| `Season`, `Date` | Competition season and date |
| `Link` | YouTube video link |

---

### `frames/JP00XX/`

Raw video frames extracted from the [Ski-TB Dataset](https://machinelearning.uniud.it/datasets/skitb/) competition footage, organized by jump ID. Each subfolder (e.g. `JP0005/`) contains ~350 sequential `.jpg` frames.

> **Not included in the repository** — download from Google Drive (see the main [README](../README.md#-dataset--checkpoints-download)).

---

### `annotations/JP00XX/`

Processed COCO-format annotations for each jump, produced by the `annotation_preprocessing/` pipeline. Each subfolder contains:
- **COCO JSON file** — keypoint annotations for all frames of that jump (included on GitHub).
- **Visualization overlays** — annotated images with skeleton overlays for quality inspection (only on Google Drive).

> On GitHub, only the JSON annotation files are included. If you also want the visualization overlays, replace the `annotations/` folder from the repo with the full version from [Google Drive](https://drive.google.com/drive/folders/10cKnZdP3x-tIoTHMw_nk2GYT9gc-uXRN?usp=drive_link).

---

## 🔗 How This Dataset Is Used

| Module | Uses |
|--------|------|
| **`annotation_preprocessing/`** | Produces `annotations/`, `frames/`, and `keypoints_dataset.csv` |
| **`metrics/`** | Reads `keypoints_dataset.csv`, `jump_phases_SkiTB.csv`, and `JP_data.csv` to compute biomechanical metrics |
| **`SkiPoseModel/`** | Reads COCO annotations from `annotations/` for model training |
| **`dashboard/`** | Reads all dataset files for interactive exploration and visualization |
