# unet_pretrain.py
"""
Training script for microtubule segmentation using MONAI 3D U-Net.

Trains a 3D U-Net on annotated 100³ voxel subvolumes to segment microtubules
from electron microscopy data. Uses mixed precision training, data augmentation
(all 48 rotation/flip combinations), and logs to Weights & Biases.

Usage:
    python unet_pretrain.py

Configuration is controlled via util_files/train_helper.py hyperparameters dict.
"""
# ================== Imports ==================
import os
import sys
import random
import datetime

import numpy as np
import wandb

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from monai.utils import set_determinism

np.set_printoptions(precision=5, suppress=True)

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util_files import train_helper
from dataset import MicrotubuleDataset

hyperparameters = train_helper.hyperparameters

# ================== Configuration Variables ==================
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

DEVICE = hyperparameters['device']
print(f"Using device: {DEVICE}")
torch.backends.cudnn.benchmark = True

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
set_determinism(seed=RANDOM_SEED)


# ================== Training Function ==================
def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    optimizer,
    experiment_name,
    checkpoint_dir,
    pos_weight=POS_WEIGHT,
    scheduler=None,
    scaler=None
):
    """Train the segmentation model with mixed precision support."""
    wandb.init(project="microtubule-segmentation", name=experiment_name, config=hyperparameters)
    best_val_loss = float('inf')
    best_val_dice = 0.0
    patience_counter = 0
    training_start_time = datetime.datetime.now()
    epoch_times = []

    for epoch in range(num_epochs):
        epoch_start_time = datetime.datetime.now()

        # Training phase
        model.train()
        train_loss = 0
        train_metrics = {
            'dice': 0, 'precision': 0, 'recall': 0, 'dice_strict': 0,
            'accuracy': 0, 'balanced_accuracy': 0
        }
        num_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(scaler is not None)):
                outputs = model(images)
                loss = train_helper.bce_dice_loss(outputs, labels, pos_weight)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            batch_metrics = train_helper.compute_segmentation_metrics(outputs, labels)
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key]

        # Average training metrics
        avg_train_loss = train_loss / num_batches
        for key in train_metrics:
            train_metrics[key] /= num_batches
        train_metrics['loss'] = avg_train_loss

        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = {
            'dice': 0, 'precision': 0, 'recall': 0, 'dice_strict': 0,
            'accuracy': 0, 'balanced_accuracy': 0
        }
        num_val_batches = len(val_loader)

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(enabled=(scaler is not None)):
                    outputs = model(images)
                    loss = train_helper.bce_dice_loss(outputs, labels, pos_weight)

                val_loss += loss.item()
                batch_metrics = train_helper.compute_segmentation_metrics(outputs, labels)
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]

        # Average validation metrics
        avg_val_loss = val_loss / num_val_batches
        for key in val_metrics:
            val_metrics[key] /= num_val_batches
        val_metrics['loss'] = avg_val_loss

        epoch_end_time = datetime.datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration.total_seconds())

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # Log metrics
        wandb.log({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_dice': train_metrics['dice'],
            'val_dice': val_metrics['dice'],
            'train_dice_strict': train_metrics.get('dice_strict', 0),
            'val_dice_strict': val_metrics.get('dice_strict', 0),
            'train_precision': train_metrics['precision'],
            'val_precision': val_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_recall': val_metrics['recall'],
            'train_bal_accuracy': train_metrics['balanced_accuracy'],
            'val_bal_accuracy': val_metrics['balanced_accuracy'],
            'epoch': epoch + 1,
            'epoch_time': epoch_duration.total_seconds(),
            'learning_rate': optimizer.param_groups[0]['lr'],
        })

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(train_helper.format_metrics(train_metrics, 'Train'))
        print(train_helper.format_metrics(val_metrics, 'Val  '), end=' ')

        # Save best model
        if val_metrics['dice'] > best_val_dice:
            print('* New best model (Dice) ', end=' ')
            best_val_dice = val_metrics['dice']
            patience_counter = 0
            train_helper.save_checkpoint(
                model, optimizer, epoch, avg_val_loss,
                val_metrics, checkpoint_dir, 'best_dice_model.pt'
            )

        if avg_val_loss < best_val_loss:
            print('* New best model (Loss) ', end=' ')
            best_val_loss = avg_val_loss
            train_helper.save_checkpoint(
                model, optimizer, epoch, avg_val_loss,
                val_metrics, checkpoint_dir, 'best_loss_model.pt'
            )
        else:
            print()
            patience_counter += 1
        print()

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

    training_duration = datetime.datetime.now() - training_start_time
    train_helper.log_final_metrics(
        training_duration, epoch_times, epoch + 1,
        best_val_loss, best_val_dice, checkpoint_dir
    )
    wandb.finish()


def main():
    """Main training entry point."""
    # Initialize model
    model = train_helper.initialize_model('unet')
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    # Create datasets
    dataset = MicrotubuleDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
    )

    dataset_length = len(dataset)
    if dataset_length == 0:
        raise ValueError("Dataset is empty. Please check your data directories.")

    # Split into train/val
    val_indices = list(range(0, dataset_length, 5))
    train_indices = [i for i in range(dataset_length) if i not in val_indices]

    train_dataset = MicrotubuleDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        indices=train_indices,
        transforms=train_helper.get_uniform_transforms()
    )

    val_dataset = MicrotubuleDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        indices=val_indices,
        transforms=train_helper.get_uniform_transforms()
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Initialize optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=25,
        verbose=True
    )

    scaler = GradScaler()

    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        optimizer=optimizer,
        experiment_name=EXPERIMENT_NAME,
        checkpoint_dir=CHECKPOINT_DIR,
        scheduler=scheduler,
        scaler=scaler
    )


if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    main()
