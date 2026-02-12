# SkiPoseModel - Ski Reconstruction from Human Pose

<img width="180" height="812" alt="test_0028" src="https://github.com/user-attachments/assets/cf20c8d1-5103-4fc5-8560-9f9dcc37907d" align="right"/>


SkiPoseModel is a deep learning model designed to reconstruct ski positions from human skeletal data. Given a skier's body pose (15 body joints), the model predicts the positions of ski keypoints (8 in total). This is achieved through a transformer-based architecture that learns to infer ski locations based on body joint positions.

### Model Objective

The core task is: **Given 15 body joints (with skis masked), predict the ski joint positions.**

---

## Repository Structure

```
SkiPoseModel/
├── main.py                      # Main training/testing/demo entry point
├── model.py                     # AdaptationNetwork (Lightning module)
├── datamodule.py                # SkijumpDataModule and SkijumpDataset
├── transformer.py               # Transformer architecture components
├── preprocess.py                # Data preprocessing pipeline (COCO JSON → pickle)
├── postprocess_visualize.py     # Visualization and post-processing of results
├── domainadapt_flags.py         # Configuration flags for all scripts
├── dataset/                     # Raw dataset (COCO JSON annotations)
├── dataset_preprocessed/        # Preprocessed datasets (train.pkl, val.pkl, test.pkl)
├── results/                     # Output predictions and visualizations
└── requirements.txt             # Python dependencies
```

---

## Prerequisites & Setup

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support is required for training
- **CUDA Version**: 11.8+ (ensure CUDA Toolkit and cuDNN are installed)
- **Python**: 3.9+

### Installation Steps

#### 1. Install Dependencies

Create and activate a virtual environment:

```bash
python -m venv sport_tech_env
# Windows
sport_tech_env\Scripts\activate
# Linux/Mac
source sport_tech_env/bin/activate
```

Install required packages:

```bash
cd SkiPoseModel
pip install -r requirements.txt
```

#### 2. Prepare Weights & Biases (Optional)

If using WandB for experiment tracking:

```bash
wandb login
```

#### 3. Verify GPU Availability

```bash
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```

---

## Data Preparation

### Preprocessing Pipeline

Before training, raw COCO-format annotations must be converted to pickle files:

```bash
python preprocess.py 
```

**What it does:**
- Loads COCO JSON annotations from `raw_dataset_dir`
- Remaps keypoints to 23-joint skeleton format
- Extracts 2D pose coordinates
- Splits data: 70% train, 15% validation, 15% test
- Saves pickle files: `train.pkl`, `val.pkl`, `test.pkl`

**Expected output:**
```
dataset_preprocessed/
├── train.pkl
├── val.pkl
└── test.pkl
```

---

## Training

### Start Training
If you wish you can change the parameters in domainadapt_flags.py

```bash
python main.py --mode=train \
    --n_epochs=100 
```

### Training Output

- **Checkpoints**: Saved in `checkpoints/{dataset}_{timestamp}/`
- **Metrics**: Logged to WandB (loss curves, validation metrics)
- **Best Model**: Top 3 checkpoints saved based on validation loss

---

## Testing & Evaluation

### Run Inference on Test Set

```bash
python main.py --mode=test \
    --load_checkpoint="path_of_the_most_recent_checkpoint" 
```

**Output:**
- `test_results.pkl` - Contains predictions and ground truth for analysis

---

## Post-Processing & Visualization

After testing, visualize results with ski linearization:

```bash
python postprocess_visualize.py 
```

This:
- Applies PCA-based linearization to ski points to force the predicted points on a straight line
- Generates visualizations with GT and predicted skiis

---

### Training Strategy

- **Masking**: During training, ski joints are masked
- **Loss Function**: MSE between predicted and ground truth masked joints
- **Validation**: Loss computed only on ski joints (indices 19,20,21,22,12,13,14,15)

---

