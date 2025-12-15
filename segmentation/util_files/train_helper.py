# train_helper.py
# Training utilities for microtubule segmentation

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
import wandb
import platform
import datetime

from monai.networks.nets import (
    UNet, SwinUNETR, SegResNet, DynUNet, AttentionUnet,
    BasicUNet, VNet, SegResNetDS, SegResNetVAE, HighResNet, BasicUNetPlusPlus
)
from monai.transforms import Compose, RandFlipd, RandRotate90d, OneOf

# ==================== Hyperparameters ====================
hyperparameters = {
    'experiment_name': 'ninja-creami',
    # Data parameters
    'image_dir': 'training/subvols/image',
    'label_dir': 'training/labeled_binary_large',
    'checkpoint_dir': 'checkpoints',
    # Training parameters
    'batch_size': 1,
    'num_epochs': 1000,
    'learning_rate': 5e-4,
    'weight_decay': 2e-4,
    'num_workers': 4,
    'random_seed': 42,
    'patience': 70,
    'pos_weight': 1.0,
    'device': 'cuda',
    # Model parameters
    'spatial_dims': 3,
    'in_channels': 1,
    'out_channels': 1,
    'kernel_size': 3,
    'network_channels': (32, 64, 128, 256),
    'strides': (2, 2, 2),
    'num_res_units': 6,
    'dropout': 0.2,
    'freeze_encoder': False,
    # System parameters
    'torch_version': torch.__version__,
    'numpy_version': np.__version__,
    'python_version': platform.python_version(),
    'system_platform': platform.system(),
    'training_start_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

# Export hyperparameters as module-level constants
EXPERIMENT_NAME = hyperparameters['experiment_name']
IMAGE_DIR = hyperparameters['image_dir']
LABEL_DIR = hyperparameters['label_dir']
CHECKPOINT_DIR = hyperparameters['checkpoint_dir']
BATCH_SIZE = hyperparameters['batch_size']
NUM_EPOCHS = hyperparameters['num_epochs']
LEARNING_RATE = hyperparameters['learning_rate']
WEIGHT_DECAY = hyperparameters['weight_decay']
NUM_WORKERS = hyperparameters['num_workers']
RANDOM_SEED = hyperparameters['random_seed']
PATIENCE = hyperparameters['patience']
POS_WEIGHT = hyperparameters['pos_weight']
SPATIAL_DIMS = hyperparameters['spatial_dims']
IN_CHANNELS = hyperparameters['in_channels']
OUT_CHANNELS = hyperparameters['out_channels']
NETWORK_CHANNELS = hyperparameters['network_channels']
KERNEL_SIZE = hyperparameters['kernel_size']
NUM_RES_UNITS = hyperparameters['num_res_units']
STRIDES = hyperparameters['strides']
DROPOUT = hyperparameters['dropout']
FREEZE_ENCODER = hyperparameters['freeze_encoder']
DEVICE = hyperparameters['device']


# ==================== Model Initialization ====================
def initialize_model(model_type="unet"):
    """Initialize a segmentation model based on the specified type."""
    model_configs = {
        "unet": {
            "class": UNet,
            "params": {
                "spatial_dims": SPATIAL_DIMS,
                "in_channels": IN_CHANNELS,
                "out_channels": OUT_CHANNELS,
                "channels": NETWORK_CHANNELS,
                "kernel_size": KERNEL_SIZE,
                "num_res_units": NUM_RES_UNITS,
                "strides": STRIDES,
                "dropout": DROPOUT,
                "norm": ("INSTANCE", {"affine": True}),
            }
        },
        "vnet": {
            "class": VNet,
            "params": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "dropout_prob_down": 0.3,
                "dropout_prob_up": (0.3, 0.3),
                "bias": False,
                "act": ("elu", {"inplace": True})
            }
        },
        "segresnet": {
            "class": SegResNet,
            "params": {
                "spatial_dims": 3,
                "init_filters": 16,
                "in_channels": 1,
                "out_channels": 1,
                "dropout_prob": 0.3,
                "use_conv_final": True,
                "blocks_down": (1, 2, 2, 4),
                "blocks_up": (1, 1, 1),
                "act": ("RELU", {"inplace": True}),
                "norm": ("GROUP", {"num_groups": 8})
            }
        },
        "attentionunet": {
            "class": AttentionUnet,
            "params": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "channels": (16, 32, 64, 128),
                "strides": (2, 2, 2),
                "dropout": 0.3,
                "kernel_size": 3,
                "up_kernel_size": 3,
            }
        },
        "dynunet": {
            "class": DynUNet,
            "params": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
                "strides": [(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
                "upsample_kernel_size": [(2, 2, 2), (2, 2, 2), (2, 2, 2)],
                "norm_name": ('INSTANCE', {'affine': True}),
                "deep_supervision": True,
                "deep_supr_num": 1,
                "res_block": True,
                "dropout": DROPOUT
            }
        },
        "basicunet": {
            "class": BasicUNet,
            "params": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "features": (16, 32, 64, 128, 256, 16),
                "act": ("LeakyReLU", {"inplace": True, "negative_slope": 0.1}),
                "norm": ("instance", {"affine": True}),
                "dropout": 0.3,
                "upsample": "deconv"
            }
        },
        "basicunetplusplus": {
            "class": BasicUNetPlusPlus,
            "params": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "features": (32, 32, 64, 128, 256, 32),
                "act": ("LeakyReLU", {"inplace": True, "negative_slope": 0.1}),
                "norm": ("instance", {"affine": True}),
                "dropout": 0.2,
                "upsample": "pixelshuffle",
                "deep_supervision": True
            }
        }
    }

    if model_type not in model_configs:
        raise ValueError(f"Model type {model_type} not supported. Choose from: {list(model_configs.keys())}")

    config = model_configs[model_type]
    model = config["class"](**config["params"])
    print(f"Initialized {model_type.upper()}")
    return model


# ==================== Loss Functions ====================
def bce_dice_loss(outputs, targets, pos_weight, alpha=0.5, gamma=1):
    """Combined BCE and Dice loss for segmentation."""
    prob = torch.sigmoid(outputs)
    eps = 1e-6
    intersection = (prob * targets).sum()
    union = prob.sum() + targets.sum()
    dice = 1 - (2. * intersection + eps) / (union + eps)

    return 0.05 * dice + F.binary_cross_entropy_with_logits(
        outputs,
        targets,
        reduction='mean'
    )


# ==================== Metrics ====================
def compute_segmentation_metrics(outputs, labels, threshold=0.5):
    """Compute segmentation metrics at standard and strict thresholds."""
    if isinstance(outputs, list):
        outputs = outputs[-1]

    outputs = outputs.detach()
    prob = torch.sigmoid(outputs)

    metrics = {}

    for thresh, suffix in [(threshold, ''), (0.6, '_strict')]:
        pred_masks = (prob > thresh).float()
        true_masks = labels.float()

        true_positives = (pred_masks * true_masks).sum()
        true_negatives = ((1 - pred_masks) * (1 - true_masks)).sum()
        false_positives = (pred_masks * (1 - true_masks)).sum()
        false_negatives = ((1 - pred_masks) * true_masks).sum()

        eps = 1e-6

        precision = (true_positives + eps) / (true_positives + false_positives + eps)
        recall = (true_positives + eps) / (true_positives + false_negatives + eps)
        dice = (2 * true_positives + eps) / (2 * true_positives + false_positives + false_negatives + eps)

        total_pixels = true_positives + true_negatives + false_positives + false_negatives
        accuracy = (true_positives + true_negatives) / total_pixels

        sensitivity = recall
        specificity = (true_negatives + eps) / (true_negatives + false_positives + eps)
        balanced_accuracy = (sensitivity + specificity) / 2

        metrics.update({
            f'dice{suffix}': dice.item(),
            f'precision{suffix}': precision.item(),
            f'recall{suffix}': recall.item(),
            f'accuracy{suffix}': accuracy.item(),
            f'balanced_accuracy{suffix}': balanced_accuracy.item(),
            f'true_positives{suffix}': true_positives.item(),
            f'true_negatives{suffix}': true_negatives.item(),
            f'false_positives{suffix}': false_positives.item(),
            f'false_negatives{suffix}': false_negatives.item(),
            f'specificity{suffix}': specificity.item()
        })

        if suffix == '':
            metrics['f1'] = dice.item()
            if true_positives + false_positives > 0:
                metrics['positive_predictive_value'] = (true_positives / (true_positives + false_positives)).item()
            else:
                metrics['positive_predictive_value'] = 0.0

            if true_negatives + false_negatives > 0:
                metrics['negative_predictive_value'] = (true_negatives / (true_negatives + false_negatives)).item()
            else:
                metrics['negative_predictive_value'] = 0.0

    return metrics


def format_metrics(metrics, prefix=''):
    """Format metrics for pretty printing."""
    metric_strings = []
    for key in ['loss', 'dice', 'precision', 'recall', 'accuracy', 'balanced_accuracy']:
        if key in metrics:
            metric_strings.append(f"{key.capitalize()}: {metrics[key]:.4f}")
    return f"{prefix} - " + ", ".join(metric_strings)


# ==================== Data Augmentation ====================
def get_uniform_transforms():
    """
    Generate all possible combinations of rotations and flips for 3D augmentation.
    Returns a MONAI OneOf transform that randomly selects from 48 possible orientations.
    """
    flips = [[], [(0,)]]

    rotations = []
    for face_up in range(6):
        for rotation in range(4):
            if face_up == 0:
                rots = [(0, 1)] * rotation
            elif face_up == 1:
                rots = [(0, 2), (0, 2)] + [(0, 1)] * rotation
            elif face_up == 2:
                rots = [(0, 2)] + [(0, 1)] * rotation
            elif face_up == 3:
                rots = [(0, 2), (0, 2), (0, 2)] + [(0, 1)] * rotation
            elif face_up == 4:
                rots = [(1, 2)] + [(1, 0)] * rotation
            else:
                rots = [(1, 2), (1, 2), (1, 2)] + [(1, 0)] * rotation
            rotations.append(rots)

    all_transforms = []
    keys = ['image', 'label']

    for flip_axes, rotation_sequence in product(flips, rotations):
        transform_list = []
        for axis in flip_axes:
            transform_list.append(RandFlipd(keys=keys, prob=1.0, spatial_axis=axis))
        for rot_axes in rotation_sequence:
            transform_list.append(RandRotate90d(keys=keys, prob=1.0, max_k=1, spatial_axes=rot_axes))
        all_transforms.append(Compose(transform_list))

    return OneOf(all_transforms, weights=[1/len(all_transforms)] * len(all_transforms))


# ==================== Checkpointing ====================
def save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, checkpoint_dir, filename):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_metrics': val_metrics,
        'hyperparameters': hyperparameters
    }
    torch.save(checkpoint, f"{checkpoint_dir}/{filename}")
    wandb.save(f"{checkpoint_dir}/{filename}")


def log_final_metrics(training_duration, epoch_times, completed_epochs, best_val_loss, best_val_dice, checkpoint_dir):
    """Log final training metrics."""
    final_metrics = {
        'training_duration_seconds': training_duration.total_seconds(),
        'training_duration_formatted': str(training_duration),
        'average_epoch_time_seconds': np.mean(epoch_times),
        'fastest_epoch_seconds': np.min(epoch_times),
        'slowest_epoch_seconds': np.max(epoch_times),
        'epoch_time_std_seconds': np.std(epoch_times),
        'completed_epochs': completed_epochs,
        'best_val_loss': best_val_loss,
        'best_val_dice': best_val_dice
    }
    return final_metrics
