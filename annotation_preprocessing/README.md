#  Annotation Preprocessing 

This folder contains the **annotation preprocessing pipeline** that transforms raw COCO-format exports from [Roboflow](https://roboflow.com/) into clean, interpolated, and normalized keypoint annotations ready for downstream metric computation and model training.

---

##  Folder Structure

```
annotation_preprocessing/
├── main.py                        # Workflow orchestrator
├── raw_annotations/               # Input COCO annotations from Roboflow
└── utils/                         # Function classes used for annotation pipeline
```


---

## Pipeline Overview & Modules

The pipeline is executed interactively through `main.py`. For each jump, the following steps are performed in sequence:

| Step | Module | Description |
|------|--------|-------------|
| **1. Extract** | `annotation_manager.py` | Parses the full COCO export and extracts annotations for a specific jump using Roboflow user tags.  |
| **2. Filter Boxes** | `box_filter.py`  | Filters bounding boxes by removing entries corresponding to frames that were manually excluded (removed/occluded). |
| **3. Interpolate** | `interpolator.py`  | Linearly interpolates keypoints between manually annotated frames (~60) to fill all ~350 frames of each jump.  |
| **4. Visualize** | `visualizer.py`  | Generates annotated images with skeleton overlays and compiles them into an MP4 video for quality inspection. Draws color-coded skeletons and bounding boxes on video frames. |
| **5. Normalize** | `normalizer.py`  | Applies anatomical normalization (pelvis-rooted, hybrid body-scale) and fixed scaling to map all keypoints into a resolution- and position-independent [0, 1] coordinate space. Generates grid visualizations and outputs per-jump normalized data. |

After all jumps are processed, the `Normalizer` aggregates all normalized annotations into a single `keypoints_dataset.csv` file.

---

## ▶️ How to Run

**1.** Download `raw_annotations/` from [Google Drive](https://drive.google.com/drive/folders/10cKnZdP3x-tIoTHMw_nk2GYT9gc-uXRN?usp=drive_link) and place it inside `annotation_preprocessing/`.

**2.** Make sure the `dataset/frames/` folder is also available (otherwise download from [Google Drive](https://drive.google.com/drive/folders/10cKnZdP3x-tIoTHMw_nk2GYT9gc-uXRN?usp=drive_link)).

**3.** Run the pipeline from the project root:

```bash
python annotation_preprocessing/main.py
```

**4.** The script will interactively ask for:
- **Jump range** — e.g. `10-15` or `6`
- **Interpolation** — whether to interpolate between annotated frames
- **Visualization** — whether to generate overlay video
- **Normalization** — whether to normalize keypoints

---

## Output

All output is written to the `dataset/` folder:

```
dataset/
├── annotations/JP00XX/train/
│   ├── annotations_jump{N}.json                  # Extracted per-jump COCO annotations
│   ├── annotations_interpolated_jump{N}.coco.json # Interpolated annotations (all frames)
│   ├── annotations_normalized_jump{N}.coco.json   # Normalized annotations
│   └── visualization_normalized_grid_jump{N}.png  # Normalization quality check grid
├── annotations/JP00XX/visualizations/
│   ├── frame_XXXXX.jpg                            # Skeleton overlay images
│   └── output_video_JP00XX.mp4                    # Overlay video
└── keypoints_dataset.csv                          # Aggregated normalized keypoints (all jumps)
```

The generated `keypoints_dataset.csv` is the primary input for the `metrics/` pipeline.

---
