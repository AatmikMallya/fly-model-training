# Segmentation Methods

This document describes the methods used to generate the microtubule segmentation dataset.

---

## 1. Data Source

### Electron Microscopy Dataset

We used the FlyEM Hemibrain connectome dataset (Scheffer et al., 2020), a FIB-SEM volume of the adult *Drosophila melanogaster* central brain. The dataset has an isotropic resolution of 8 nm per voxel, with total dimensions of approximately 41,344 × 37,888 × 34,367 voxels (Z × Y × X).

### Data Access

EM image data was accessed programmatically via Google Cloud Storage using the tensorstore library with the neuroglancer precomputed format. For inference, we used CLAHE-enhanced (Contrast-Limited Adaptive Histogram Equalization) grayscale images. Neuron body segmentation masks were obtained from the same dataset.

Neuron skeleton data was retrieved from the neuprint database (neuprint.janelia.org) using the neuprint-python client library.

---

## 2. Training Data Preparation

### Subvolume Selection

We randomly sampled 100 subvolumes from the Hemibrain dataset for manual annotation. Each subvolume was 100 × 100 × 100 voxels (800 × 800 × 800 nm at 8 nm/voxel resolution).

### Manual Annotation

Microtubules were manually traced within each subvolume using annotation software in Python.

Subvolumes containing ambiguous structures (where microtubules could not be reliably distinguished from other cellular components) were excluded from the training set.

### Label Format

Annotations were stored as binary masks (numpy arrays) with the same dimensions as the corresponding image volumes. Voxels belonging to microtubules were labeled as 1; all other voxels were labeled as 0.

---

## 3. Model Architecture

### Network Design

We used a 3D U-Net architecture implemented in the MONAI framework (monai.networks.nets.UNet). The U-Net follows an encoder-decoder structure with skip connections.

### Architecture Details

| Component | Specification |
|-----------|---------------|
| Input size | 96 × 96 × 96 voxels (single channel) |
| Encoder channels | 32 → 64 → 128 → 256 |
| Decoder channels | 256 → 128 → 64 → 32 → 1 |
| Downsampling | Stride-2 convolutions (3 steps) |
| Upsampling | Transposed convolutions with skip concatenation |
| Residual units | 6 per encoder/decoder block |
| Kernel size | 3 × 3 × 3 |
| Normalization | Instance normalization (affine=True) |
| Dropout | 0.2 |
| Output activation | Sigmoid (probability output) |

The encoder progressively downsamples the input from 96³ to 12³ voxels while increasing feature channels. The bottleneck operates at 6³ spatial resolution with 256 channels. The decoder upsamples back to the original resolution, concatenating skip connections from corresponding encoder levels to preserve fine spatial details.

Training volumes of 100³ voxels were cropped to 96³ (indices 2:98 in each dimension) to avoid border artifacts from the manual annotation process.

---

## 4. Training Procedure

### Loss Function

We used a weighted combination of binary cross-entropy (BCE) and Dice loss:

$$\mathcal{L} = \text{BCE}(p, t) + 0.05 \times \text{Dice}(p, t)$$

where:
- BCE = −[t · log(σ(p)) + (1−t) · log(1−σ(p))]
- Dice = 1 − (2·|P ∩ T| + ε) / (|P| + |T| + ε)
- p = model output (logits), t = ground truth, σ = sigmoid function, ε = 10⁻⁶

The low weight on Dice loss (0.05) prioritized stable BCE gradients while still encouraging overlap optimization.

### Optimization

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 5 × 10⁻⁴ |
| Weight decay | 2 × 10⁻⁴ |
| Betas | (0.9, 0.999) |
| Batch size | 1 |
| Gradient clipping | max_norm = 1.0 |

Learning rate was reduced by a factor of 0.5 if validation loss did not improve for 25 consecutive epochs (ReduceLROnPlateau scheduler).

### Data Augmentation

To maximize the effective training set size and encourage rotation-invariant predictions, we applied all 48 orientation-preserving symmetries of a cube during training.

### Train/Validation Split

The 100 annotated volumes were split 80/20 for training and validation. Every 5th sample (indices 0, 5, 10, ...) was assigned to the validation set; all remaining samples were used for training. This deterministic split ensured reproducibility.

### Early Stopping and Convergence

Training was configured for a maximum of 1000 epochs with early stopping triggered after 70 epochs without improvement in validation Dice score. Mixed-precision training (FP16) was enabled using PyTorch's GradScaler for computational efficiency.

The final model stopped at epoch 298 with the following validation metrics:

| Metric | Value |
|--------|-------|
| Dice coefficient | 0.827 |
| Precision | 0.866 |
| Recall | 0.800 |
| Loss | 0.017 |

Training metrics at the same epoch: Dice = 0.813, Precision = 0.841, Recall = 0.815, Loss = 0.017. The similar performance on training and validation sets indicates the model generalized well without overfitting.

---

## 5. Inference Pipeline

Inference was performed on a per-neuron basis, processing approximately 18,000 neurons from the Hemibrain dataset. For each neuron, the pipeline consisted of three stages: preprocessing, model inference, and postprocessing. The entire pipeline ran for two weeks on a single A5000 GPU.

### 5.1 Preprocessing

**Tile placement:** For each neuron body ID, we retrieved the skeleton reconstruction from neuprint. The skeleton provided a sparse representation of the neuron's 3D trajectory through the volume.

We generated overlapping 96³ voxel tiles to cover the entire neuron volume using an adaptive placement algorithm:

1. **Path traversal**: Walk along skeleton branches, placing tiles at intervals of 81 voxels (96 − 15 voxel minimum overlap).

2. **Radius-adaptive coverage**: At skeleton points where the neuron radius exceeded 32 voxels (one-third of tile size), additional tiles were placed in concentric rings around the skeleton to ensure coverage of thick dendritic or axonal regions.

3. **KD-tree optimization**: A spatial index was maintained to avoid redundant tile placement while ensuring complete coverage.

**Boundary expansion:** After initial tile placement, we iteratively checked for incomplete coverage at tile boundaries. If more than 20% of a tile's face contained neuron segmentation, an adjacent tile was added in that direction. This expansion was repeated for up to 5 iterations to capture neuron branches extending beyond the initial skeleton-based coverage.

**Empty tile filtering:** Tiles containing no neuron segmentation (based on the Hemibrain body ID mask) were discarded to reduce unnecessary computation.

### 5.2 Model Inference

For each tile, we:
1. Fetched CLAHE-enhanced grayscale EM data via tensorstore
2. Normalized pixel intensities to [0, 1] by dividing by 255
3. Masked the volume with the neuron body segmentation to zero out background
4. Ran the trained U-Net to obtain per-voxel microtubule probabilities

### 5.3 Postprocessing: Tile Merging and Skeletonization

**Distance-weighted blending:** Overlapping tile predictions were merged using a 3D distance-weighted blending scheme. Each tile was assigned a weight map:
- Center region (60³ voxels): weight = 1.0
- Edge regions: linear falloff from 1.0 to 0.1 over 18 voxels

The final probability at each voxel was computed as:

$$P(x,y,z) = \frac{\sum_i w_i(x,y,z) \cdot p_i(x,y,z)}{\sum_i w_i(x,y,z)}$$

where the sum is over all tiles containing that voxel. This blending reduced seam artifacts at tile boundaries.

**Thresholding:** Merged probability maps were thresholded at p = 0.5 to produce binary segmentation masks.

**Skeletonization:** Binary segmentations were skeletonized using the scikit-image skeletonize_3d function, reducing the volumetric segmentation to a 1-voxel-thick centerline representation. This compressed the final segmentation to 40GB in total.

**Connected component processing:** To manage memory for large neurons, tiles were processed in connected components using a wavefront propagation algorithm. Tiles were grouped by spatial overlap, merged and skeletonized in clusters, then combined into the final output.

---

## 6. Output Format

### Data Structure

For each neuron, the output is a numpy array of shape (N, 3) containing the 3D coordinates (z, y, x) of all skeletonized microtubule voxels. Coordinates are in the native Hemibrain voxel space (8 nm resolution, origin at dataset corner).

```python
import numpy as np
coords = np.load('skeleton_points_12345.npy')  # shape: (N, 3)
# coords[i] = [z, y, x] position of i-th skeleton voxel
```

### Dataset Scale

| Metric | Value |
|--------|-------|
| Neurons processed | ~18,000 |
| Total output size | ~40 GB |
| Format | numpy .npy files |

---

## 8. Software and Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.0.0 | Deep learning framework |
| MONAI | ≥1.3.0 | Medical image analysis, U-Net implementation |
| tensorstore | ≥0.1.45 | Hemibrain data access |
| neuprint-python | ≥0.4.20 | Skeleton retrieval |
| scikit-image | ≥0.21.0 | Skeletonization |
| numpy | ≥1.24.0 | Array operations |
| scipy | ≥1.10.0 | Spatial algorithms (KDTree) |
| networkx | ≥3.0 | Connected component analysis |

---

## 9. References

1. Hemibrain: https://doi.org/10.7554/eLife.57443

2. MONAI package: https://arxiv.org/abs/2211.02701

3. neuprint: https://doi.org/10.1101/2020.01.16.909465

4. scikit-image https://doi.org/10.7717/peerj.453

## Contact

Original author: Aatmik Mallya (aatmik54@gmail.com)
