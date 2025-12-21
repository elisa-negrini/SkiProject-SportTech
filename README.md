# Ski Jump Analysis Project 🎿

A comprehensive pipeline for ski jump video analysis using pose estimation and keypoint interpolation.

---

## 📁 Project Structure

```
SkiProject-SportTech/
│
├── 📄 requirements.txt                    # Python dependencies
├── 📄 .gitignore                          # Git ignore rules
│
├── 📂 dataset/
│   ├── 📂 frames/                         # Video frames organized by jump
│   │   ├── 📂 JP0006/                     # these are all the frames, not filtered
│   │   ├── 📂 JP0007/
│   │   └── 📂 JP00XX/
│   │
│   └── 📂 annotations/                    # Processed annotations & visualizations
│       └── 📂 JP00XX/
│           ├── 📂 train/
│           │   ├── 📄 annotations_jumpX.json                    # Filtered annotations
│           │   └── 📄 annotations_interpolated_jumpX.coco.json  # Interpolated annotations
│           │
│           └── 📂 visualizations/         # Output visualizations
│               ├── 📄 frame_XXXXX.jpg 
│               └── 📄 ...
│               └── 📄 output_video_JP00XX.mp4   # 'Final' video
│
├── 📂 raw_annotations/                    # Downloaded Roboflow annotations
│   └── 📂 train/
│       └── 📄 _annotations.coco.json      # Full COCO annotations from Roboflow (all jumps mixed togethere)
│
├── 📂 utils/              
│   ├── 📄 __init__.py                     # Creates annotated frame images
│   ├── 📄 annotations_manager.py          # Extracts jump-specific annotations
│   ├── 📄 box_filter.py                   # Extraxt and filter boxes annotations
│   ├── 📄 interpolator.py                 # Keypoint interpolation between frames
│   └── 📄 visualizer.py                   # Generates annotated frame images and video
│
└── 📄 main.py                             # Master workflow orchestrator
├── 📂 utils/
│    ├── 📄 datamodule.py           # Dataset Loader
│    ├── 📄 domainadapt_flags.py    # Config Settings
│    ├── 📄 main.py                 # Training Entry
│    ├── 📄 model.py                # Adaptation Network
│    ├── 📄 preprocess.py           # Data Preparation
│    ├── 📄 transformer.py          # Transformer Blocks
│    └── 📄 utils.py                # Helper Functions
```

---

## 📋 File Descriptions

### Core Processing Scripts

| File | Purpose |
|------|---------|
| **filter_frames.py** | Filters video frames based on predefined conditions: 1 frame out of 10 + frames close to camera switch|
| **add_frames_interactive.py** | Interactive tool to manually add important frames at critical jump phases |
| **extract_annotations.py** | Extracts annotations for a specific jump from the main COCO file using tags (it is the only way to filter the annotations for different jumps, as the same option on roboflow is not free) |
| **interpolation.py** | Performs linear interpolation of keypoints between annotated frames adjusting the skeleton to the bounding boxes annotations that are present in the SkiTB dataset|
| **main_annotation.py** | Main workflow orchestrator that runs all processing steps sequentially |

### Visualization Scripts

| File | Purpose |
|------|---------|
| **visualize_interpolation.py** | Creates annotated images with skeleton overlay and bounding boxes |
| **create_video.py** | Generates MP4 video from annotated frames using OpenCV |

---

## 🚀 Usage Guide

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd SkiProject-SportTech

# Create virtual environment (optional but recommended)
python -m venv sport_tech_env
source sport_tech_env/bin/activate  # On Windows: sport_tech_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Processing Workflow

#### **Step 1: Filter Initial Frames**
```bash
python filter_frames.py
```
- Automatically filters video frames based on predefined criteria
- Reduces dataset size while preserving important motion sequences
- Output: Filtered frames in a local folder out of the repo (da sistemare)

#### **Step 2: Add Critical Frames Manually**
```bash
python add_frames_interactive.py
```
- Manually review and add frames at critical moments: example when the skier is very close to the camera and moves very fast
- This ensures key poses are not missed by automatic filtering

#### **Step 3: Upload to Roboflow**
1. Upload filtered frames on the Roboflow project [Roboflow](https://roboflow.com/)
2. Each jump should be uploaded in one job

#### **Step 4: Annotate the Jump**
1. In Roboflow, annotate all keypoints for the specific jump (e.g., Jump 9)
2. Use the 23-keypoint skeleton model:
   - Head, neck, shoulders, elbows, wrists
   - Torso, hips, knees, ankles, feet
   - Skis

#### **Step 5: Tag the Dataset**
1. In Roboflow, select the job for the specific jump
2. Add tag: `jumpX` (e.g., `jump9` for Jump 9)
3. This allows filtering annotations by jump number later

#### **Step 6: Download Annotations**
1. Export dataset from Roboflow in **COCO Keypoint** format
2. Download the ZIP file
3. **Clear** the `raw_annotations/` folder (remove old data)
4. Extract the downloaded files into `raw_annotations/`
5. Verify structure: `raw_annotations/train/_annotations.coco.json` exists

#### **Step 7: Run Main Workflow**
```bash
python main_annotation.py
```

**Interactive Prompts:**
1. **Enter Jump Number**: Type the jump number (e.g., `9`)
2. **Interpolate frames?** (y/n): 
   - `y`: Performs keypoint interpolation between annotated frames
   - `n`: Skips interpolation (uses only original annotations)
3. **Visualize and create video?** (y/n):
   - `y`: Creates annotated images and video
   - `n`: Skips visualization
4. **Analyze another jump?** (y/n):
   - `y`: Repeats workflow for different jump
   - `n`: Exits program

**Output:**
- Filtered annotations: `dataset/annotations/JP00XX/train/annotations_jumpX.json`
- Interpolated annotations: `dataset/annotations/JP00XX/train/annotations_interpolated_jumpX.coco.json`
- Visualizations: `dataset/annotations/JP00XX/visualizations/` frames + video
---

## 🎨 Visualization Features

### Skeleton Color Coding
- **Cyan**: Head and neck
- **Light Blue**: Right arm
- **Orange**: Left arm
- **Black**: Torso center points
- **Yellow**: Left leg
- **Purple**: Right leg
- **Pink**: Skis

### Keypoint Numbering
Each keypoint is labeled with its number (1-23) in fluorescent yellow-green for easy identification.

