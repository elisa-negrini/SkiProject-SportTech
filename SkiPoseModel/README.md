# SkiPoseModel - Ski Reconstruction from Human Pose

## Overview

SkiPoseModel is a deep learning model designed to reconstruct ski positions from human skeletal data. Given a skier's body pose (23 body joints), the model predicts the positions of ski keypoints that are occluded or difficult to detect in video footage. This is achieved through a transformer-based architecture that learns to infer ski locations based on body joint positions.

### Model Objective

The core task is: **Given partial skeletal information (body joints with skis masked), predict the complete skeleton including ski joint positions.**

---

## Repository Structure

### File Organization

```
SkiPoseModel/
├── main.py                      # Main training/testing/demo entry point
├── model.py                     # AdaptationNetwork (Lightning module)
├── datamodule.py                # SkijumpDataModule and SkijumpDataset
├── transformer.py               # Transformer architecture components
├── preprocess.py                # Data preprocessing pipeline (COCO JSON → pickle)
├── postprocess_visualize.py     # Visualization and post-processing of results
├── domainadapt_flags.py         # Configuration flags for all scripts
├── dataset/                     # Raw dataset (COCO JSON annotations + frames)
├── dataset_preprocessed/        # Preprocessed datasets (train.pkl, val.pkl, test.pkl)
├── results/                     # Output predictions and visualizations
└── requirements.txt             # Python dependencies
```

### File Descriptions

| File | Purpose |
|------|---------|
| **main.py** | Entry point. Controls train/test/demo modes using PyTorch Lightning trainer |
| **model.py** | `AdaptationNetwork` - Lightning module with encoder, masking logic, and loss functions |
| **datamodule.py** | Data loading infrastructure: `SkijumpDataModule` (train/val/test loaders) and `SkijumpDataset` (pickle loading) |
| **transformer.py** | Transformer building blocks: `PreNorm`, `Attention`, `FeedForward`, `Transformer` classes |
| **preprocess.py** | Converts COCO-format JSON annotations to pickle files with 2D pose data splits |
| **postprocess_visualize.py** | PCA-based ski linearization and visualization of predictions vs ground truth |
| **domainadapt_flags.py** | Centralized configuration: dataset paths, learning rate, batch size, mode, etc. |
| **dataset/** | Raw data: COCO JSON files and extracted frames (input for preprocessing) |
| **dataset_preprocessed/** | Processed pickle files ready for training (output from preprocess.py) |
| **results/** | Saved predictions, checkpoints, and visualizations |

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
python preprocess.py \
    --raw_dataset_dir="../dataset" \
    --dataset_dir="./dataset_preprocessed" \
    --n_joints=23 \
    --train_split=0.7 \
    --val_split=0.85
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

```bash
python main.py --mode=train \
    --dataset_dir="./dataset_preprocessed" \
    --checkpoint_dir="./checkpoints" \
    --n_epochs=25 \
    --batch_size=64 \
    --lr=2e-4 \
    --dataset="SKIJUMP"
```

### Configuration Flags

| Flag | Default | Description |
|------|---------|-------------|
| `mode` | `"train"` | Execution mode: `train`, `test`, `demo` |
| `dataset_dir` | `"./dataset_preprocessed"` | Path to preprocessed dataset |
| `checkpoint_dir` | `"./checkpoints"` | Directory to save model checkpoints |
| `n_epochs` | `25` | Number of training epochs |
| `batch_size` | `64` | Batch size for DataLoader |
| `lr` | `2e-4` | Learning rate for Adam optimizer |
| `num_workers` | `auto` | Number of DataLoader workers (defaults to CPU count / 2) |
| `n_joints` | `23` | Number of body joints in skeleton |
| `ski_joints` | `[19,20,21,22,12,13,14,15]` | Indices of ski joints to predict |
| `project_name` | `"SkiPose"` | WandB project name |
| `dataset` | `"SKIJUMP"` | Dataset identifier |

### Training Output

- **Checkpoints**: Saved in `checkpoints/{dataset}_{timestamp}/`
- **Metrics**: Logged to WandB (loss curves, validation metrics)
- **Best Model**: Top 3 checkpoints saved based on validation loss

---

## Testing & Evaluation

### Run Inference on Test Set

```bash
python main.py --mode=test \
    --dataset_dir="./dataset_preprocessed" \
    --load_checkpoint="./checkpoints/SKIJUMP_20240215_142530/last.ckpt" \
    --batch_size=64
```

**Output:**
- `test_results.pkl` - Contains predictions and ground truth for analysis

### Demo Mode

Test on a single checkpoint without WandB logging:

```bash
python main.py --mode=demo \
    --load_checkpoint="./checkpoints/SKIJUMP_20240215_142530/epoch=10-val_loss=0.0045.ckpt"
```

---

## Post-Processing & Visualization

After testing, visualize results with ski linearization:

```bash
python postprocess_visualize.py \
    --ski_joints=[19,20,21,22,12,13,14,15] \
    --mode=test
```

This:
- Applies PCA-based linearization to ski points
- Generates comparison plots (GT vs predictions)
- Saves visualizations in `comparison_linearization/`

---

## Model Architecture

### Encoder Design

```
Input (batch, 23, 2)
    ↓
nn.Linear(2 → 128)
    ↓
nn.LeakyReLU
    ↓
Transformer(dim=128, depth=6, heads=16, mlp_dim=128)
    ↓
nn.Linear(128 → 2)
    ↓
Output (batch, 23, 2)
```

### Training Strategy

- **Masking**: During training, ski joints + random 4-7 body joints are masked (set to 0)
- **Loss Function**: MSE between predicted and ground truth masked joints
- **Validation**: Loss computed only on ski joints (indices 19,20,21,22,12,13,14,15)
- **Optimizer**: Adam with ReduceLROnPlateau scheduler

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` (e.g., 32 or 16) |
| No preprocessed data | Run `preprocess.py` before training |
| Checkpoint not found | Verify path and mode (use `--mode=demo` for non-training) |
| WandB errors | Remove `wandb login` or set `--project_name=""` |

---

## Dependencies

See `requirements.txt` for a complete list. Key packages:

- **PyTorch**: Deep learning framework
- **PyTorch Lightning**: Training abstraction
- **einops**: Tensor operations
- **WandB**: Experiment tracking
- **scikit-learn**: PCA linearization
- **tqdm**: Progress bars

---

## References

- Transformer Architecture: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- PyTorch Lightning: [Documentation](https://pytorch-lightning.readthedocs.io/)
- OpenPose Keypoint Format: COCO 23-joint skeleton model
