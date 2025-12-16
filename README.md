# Microtubule Segmentation Pipeline

Automated segmentation of microtubules from 3D volumetric electron microscopy data (FlyEM Hemibrain dataset). Uses a 3D U-Net trained on manually annotated subvolumes to detect microtubules, then skeletonizes the predictions.

## Data access
This repository only contains code. All data, logs, models, and large files are accessible in Aatmik's directory in the Grace cluster. You access these files by connecting to Grace with `ssh {NetID}@grace.ycrc.yale.edu` and navigating to `/home/am3833/fly/`.

### Final trained model details
The actual model used to generate the final segmentation is at `/home/am3833/fly/segmentation/best_model/best_final_model.pt`. It can be loaded with the PyTorch library.

### Validation metrics for the final model

**Validation set (standard 80/20 split)**
- Dice: 0.827
- Precision: 0.866
- Recall: 0.800
- Loss: 0.017

**Train set**
- Dice: 0.813
- Precision: 0.841
- Recall: 0.815
- Loss: 0.017

The model was trained with the folowing configuration:
- dropout = 0.2
- freeze_encoder = false
- in_channels = 1
- kernel_size = 3
- learning_rate = 0.0005
- network_channels = 32, 64, 128, 256
- num_epochs = 1000 (final model stopped at 298)
- num_res_units = 6
- num_workers = 4
- out_channels = 1
- patience = 70
- pos_weight = 1
- random_Seed = 42
- spatial_dims = 3
- strides = 2, 2, 2
- weight_decay = 0.0002

## Two Workflows

1. **Training** (`run_train.sh`): Train the U-Net on annotated 100³ voxel subvolumes
2. **Inference Pipeline** (`batch_pipeline.sh`): For each neuron: fetch data → run inference → merge & skeletonize

## Project Structure

```
.
├── run_train.sh                 # SLURM job: training
├── README.md
├── segmentation/
│   ├── batch_pipeline.sh        # SLURM job: inference pipeline
│   ├── unet_pretrain.py         # Training script
│   ├── preprocess.py            # Fetch & tile neuron volumes
│   ├── inference.py             # Run U-Net on tiles
│   ├── postprocess.py           # Merge tiles & skeletonize
│   ├── dataset.py               # PyTorch dataset class
│   ├── lc_bodyids.txt           # List of neuron IDs to process
│   ├── requirements.txt         # Python dependencies
│   └── util_files/
│       ├── train_helper.py      # Training utilities, model configs, loss functions
│       └── voxel_utils.py       # Tensorstore data fetching
```

## Setup

### 1. Prerequisites

- Python 3.10+
- CUDA-capable GPU (for training and inference)
- Google Cloud credentials (for accessing Hemibrain data via tensorstore)
- Access to Yale HPC cluster (or adapt SLURM scripts for your cluster)

### 2. Install Dependencies

```bash
cd segmentation
pip install -r requirements.txt
```

### 3. Configure Google Cloud Credentials

The pipeline fetches data from Google Cloud Storage via tensorstore. Set up authentication:

```bash
gcloud auth application-default login
```

Or set the environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="~/.config/gcloud/application_default_credentials.json"
```

### 4. Update Paths

**Important**: Update these files for your environment:

| File | What to change |
|------|----------------|
| `segmentation/paths.py` | Set `BASE_DIR` to your segmentation directory |
| `run_train.sh` | Update working directory and venv path |
| `segmentation/batch_pipeline.sh` | Update `BASE_DIR` and venv path |

The Python scripts (`preprocess.py`, `inference.py`, `unet_pretrain.py`) import paths from `paths.py`, so you only need to configure that file once.

## Training

### Training Data

Training uses manually annotated 100×100×100 voxel subvolumes (8nm/voxel = 800×800×800 nm):
- **Images**: `training/subvols/image/*.npy` - grayscale EM data
- **Labels**: `training/labeled_binary_large/*.npy` - binary microtubule masks

### Run Training

```bash
sbatch run_train.sh
```

Or directly:
```bash
cd segmentation
python unet_pretrain.py
```

Training logs to Weights & Biases. Checkpoints saved to `checkpoints/`.

### Hyperparameters

Edit `util_files/train_helper.py` to modify:
- `network_channels`: U-Net channel depths
- `learning_rate`, `batch_size`, `num_epochs`
- `patience`: Early stopping patience

## Inference Pipeline

### Input

`lc_bodyids.txt`: One neuron body ID per line (from Hemibrain dataset)

### Run Pipeline

```bash
cd segmentation
sbatch batch_pipeline.sh
```

### Pipeline Stages

For each body ID:

1. **Preprocess** (`preprocess.py`)
   - Fetches neuron skeleton from neuprint
   - Tiles the neuron volume with overlapping 96³ subvolumes
   - Saves grayscale data and metadata to `preprocess_output/`

2. **Inference** (`inference.py`)
   - Loads trained model checkpoint
   - Runs inference on all tiles
   - Saves probability maps to `unet_output/`

3. **Postprocess** (`postprocess.py`)
   - Merges overlapping tiles with distance-weighted blending
   - Thresholds and skeletonizes the segmentation
   - Extracts skeleton coordinates
   - Saves to `final_output/{bodyId}.npy`

### Output Format

Each output file is a numpy array of shape `(N, 3)` containing the (z, y, x) coordinates of skeletonized microtubule voxels for that neuron.

## Model Architecture

We use MONAI's standard 3D U-Net implementation (`monai.networks.nets.UNet`) with residual units. This is a well-established encoder-decoder architecture for volumetric segmentation.

### Architecture Details

```
Input (1, 96, 96, 96)
    │
    ▼
Encoder Block 1: 1 → 32 channels, 6 residual units
    │ stride-2 conv (downsample to 48³)
    ▼
Encoder Block 2: 32 → 64 channels, 6 residual units
    │ stride-2 conv (downsample to 24³)
    ▼
Encoder Block 3: 64 → 128 channels, 6 residual units
    │ stride-2 conv (downsample to 12³)
    ▼
Bottleneck: 128 → 256 channels, 6 residual units
    │
    ▼
Decoder Block 1: 256 → 128 channels + skip connection
    │ upsample to 24³
    ▼
Decoder Block 2: 128 → 64 channels + skip connection
    │ upsample to 48³
    ▼
Decoder Block 3: 64 → 32 channels + skip connection
    │ upsample to 96³
    ▼
Output Conv: 32 → 1 channel
    │
    ▼
Output (1, 96, 96, 96) → sigmoid → probability
```

### Training Details
- **Input**: 96×96×96 voxel grayscale volumes (cropped from 100³ to avoid border artifacts)
- **Output**: Binary segmentation probability map
- **Augmentation**: All 48 possible 3D rotation/flip combinations (uniform sampling)
- **Optimizer**: AdamW with learning rate 5e-4, weight decay 2e-4


## Troubleshooting

### Tensorstore authentication errors
Ensure Google Cloud credentials are set up correctly. On the cluster, you may need to copy credentials:
```bash
cp ~/.config/gcloud/application_default_credentials.json ~/project/
```

### Out of memory during inference
Reduce batch size in `inference.py` or request more GPU memory.

### Pipeline job failures
Check `logs/` directory. The pipeline uses state files in `state/` to track progress and can resume from failures.

## Contact

Original author: Aatmik Mallya (aatmik54@gmail.com)
