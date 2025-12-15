# ================== Imports ==================
import os
import random
import glob
import pickle
import datetime
import platform
from pathlib import Path
from time import time
import importlib

import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import KFold
from tqdm import tqdm
from matplotlib import rc
np.set_printoptions(precision=5, suppress=True)
rc('animation', embed_limit=500)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.cuda.amp import autocast, GradScaler

from monai.utils import set_determinism
from monai.losses import DiceLoss
from monai.networks.nets import UNet

# Local imports remain the same
def import_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

home_dir = '/vast/palmer/home.grace/am3833/fly/segmentation'
utils = import_module('utils', f'{home_dir}/util_files/utils.py')
config = import_module('config', f'{home_dir}/util_files/config.py')
voxel_utils = import_module('voxel_utils', f'{home_dir}/util_files/voxel_utils.py')
segmentation = import_module('segmentation', f'{home_dir}/util_files/segmentation.py')
analysis = import_module('analysis', f'{home_dir}/util_files/analysis.py')
train_helper = import_module('train_helper', f'{home_dir}/util_files/train_helper.py')
dataset = import_module('dataset', f'{home_dir}/dataset.py')

MicrotubuleDataset = dataset.MicrotubuleDataset
hyperparameters = train_helper.hyperparameters

# ================== Configuration Variables ==================
# Keep original hyperparameters
EXPERIMENT_NAME = hyperparameters['experiment_name']
IMAGE_DIR = hyperparameters['image_dir']
LABEL_DIR = hyperparameters['label_dir']
CHECKPOINT_DIR = hyperparameters['checkpoint_dir']
BATCH_SIZE = hyperparameters['batch_size']  # Keeping as 1
NUM_EPOCHS = hyperparameters['num_epochs']
LEARNING_RATE = hyperparameters['learning_rate']
WEIGHT_DECAY = hyperparameters['weight_decay']
NUM_WORKERS = hyperparameters['num_workers']
RANDOM_SEED = hyperparameters['random_seed']
PATIENCE = hyperparameters['patience']
POS_WEIGHT = hyperparameters['pos_weight']
DEVICE = hyperparameters['device']

def get_dataloader(dataset, batch_size, shuffle, num_workers):
    """Helper function to create an appropriately configured DataLoader"""
    dataloader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'pin_memory': True
    }
    
    if num_workers > 0:
        dataloader_args['num_workers'] = num_workers
        dataloader_args['persistent_workers'] = True
    
    return DataLoader(dataset, **dataloader_args)

def train_and_evaluate(model, train_loader, val_loader, optimizer, device, num_epochs=500, 
                      early_stopping_patience=70, current_size=None, current_repeat=None):
    """Optimized training function with mixed precision."""
    best_val_loss = float('inf')
    patience_counter = 0
    best_metrics = None
    scaler = GradScaler()
    
    pbar = tqdm(range(num_epochs), 
                desc=f'Size={current_size}, Repeat={current_repeat+1}',
                leave=True)
    
    for epoch in pbar:
        # Training phase
        model.train()
        train_loss = 0
        train_metrics = {'dice': 0, 'precision': 0, 'recall': 0}
        
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Use mixed precision training
            with autocast():
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = train_helper.bce_dice_loss(outputs, labels, pos_weight=POS_WEIGHT)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            with torch.no_grad():
                batch_metrics = train_helper.compute_segmentation_metrics(outputs.detach(), labels)
                for key in train_metrics:
                    train_metrics[key] += batch_metrics[key]
        
        num_batches = len(train_loader)
        train_loss /= num_batches
        for key in train_metrics:
            train_metrics[key] /= num_batches
            
        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = {'dice': 0, 'precision': 0, 'recall': 0}
        
        with torch.no_grad(), autocast():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = train_helper.bce_dice_loss(outputs, labels, pos_weight=POS_WEIGHT)
                val_loss += loss.item()
                
                batch_metrics = train_helper.compute_segmentation_metrics(outputs, labels)
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]
        
        num_val_batches = len(val_loader)
        val_loss /= num_val_batches
        for key in val_metrics:
            val_metrics[key] /= num_val_batches
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics.copy(),
                'val_metrics': val_metrics.copy()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            
        pbar.set_postfix({
            'train_dice': f'{train_metrics["dice"]:.3f}',
            'val_dice': f'{val_metrics["dice"]:.3f}'
        })
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return best_metrics

def learning_curve_analysis(
    dataset_class, image_dir, label_dir, 
    train_sizes=[10, 20, 50, 100, 200], 
    n_repeats=5,
    n_folds=5,
    device='cuda'
):
    """Analyze model performance for different training set sizes."""
    print(f"Dataset size: {len(dataset_class(image_dir, label_dir))}")
    
    results = {
        'train_sizes': train_sizes,
        'train_metrics': {size: [] for size in train_sizes},
        'val_metrics': {size: [] for size in train_sizes}
    }
    
    all_indices = list(range(len(dataset_class(image_dir, label_dir))))
    
    for repeat in range(n_repeats):
        print(f"\nRepeat {repeat + 1}/{n_repeats}")
        random.shuffle(all_indices)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED + repeat)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
            print(f"  Fold {fold + 1}/{n_folds}")
            
            val_dataset = dataset_class(
                image_dir=image_dir,
                label_dir=label_dir,
                indices=[all_indices[i] for i in val_idx],
                transforms=train_helper.get_uniform_transforms()
            )
            val_loader = get_dataloader(
                dataset=val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS
            )
            
            for n_train in train_sizes:
                available_train_idx = [all_indices[i] for i in train_idx]
                if n_train > len(available_train_idx):
                    print(f"Warning: Requested training size {n_train} is larger than available training data {len(available_train_idx)}")
                    continue
                    
                train_indices = random.sample(available_train_idx, n_train)
                
                train_dataset = dataset_class(
                    image_dir=image_dir,
                    label_dir=label_dir,
                    indices=train_indices,
                    transforms=train_helper.get_uniform_transforms()
                )
                train_loader = get_dataloader(
                    dataset=train_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=NUM_WORKERS
                )
                
                model = train_helper.initialize_model('unet').to(device)
                optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                
                metrics = train_and_evaluate(
                    model, train_loader, val_loader, optimizer, device,
                    current_size=n_train, 
                    current_repeat=repeat * n_folds + fold
                )
                
                results['train_metrics'][n_train].append({
                    'loss': metrics['train_loss'],
                    'dice': metrics['train_metrics']['dice'],
                    'precision': metrics['train_metrics']['precision'],
                    'recall': metrics['train_metrics']['recall']
                })
                
                results['val_metrics'][n_train].append({
                    'loss': metrics['val_loss'],
                    'dice': metrics['val_metrics']['dice'],
                    'precision': metrics['val_metrics']['precision'],
                    'recall': metrics['val_metrics']['recall']
                })
                
                train_dice = np.mean([m['dice'] for m in results['train_metrics'][n_train]])
                val_dice = np.mean([m['dice'] for m in results['val_metrics'][n_train]])
                train_std = np.std([m['dice'] for m in results['train_metrics'][n_train]])
                val_std = np.std([m['dice'] for m in results['val_metrics'][n_train]])
                
                print(f"    Size {n_train}: Train Dice={train_dice:.3f}±{train_std:.3f}, "
                      f"Val Dice={val_dice:.3f}±{val_std:.3f}")
    
    # Calculate final statistics
    final_results = {
        'train_sizes': train_sizes,
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_std': [],
        'val_std': []
    }
    
    for size in train_sizes:
        final_results['train_loss'].append(np.mean([m['loss'] for m in results['train_metrics'][size]]))
        final_results['val_loss'].append(np.mean([m['loss'] for m in results['val_metrics'][size]]))
        final_results['train_dice'].append(np.mean([m['dice'] for m in results['train_metrics'][size]]))
        final_results['val_dice'].append(np.mean([m['dice'] for m in results['val_metrics'][size]]))
        final_results['train_std'].append(np.std([m['dice'] for m in results['train_metrics'][size]]))
        final_results['val_std'].append(np.std([m['dice'] for m in results['val_metrics'][size]]))
    
    return final_results

def main():
    print("Starting Optimized Learning Curve Analysis Script")
    print("="*50)
    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    set_determinism(seed=RANDOM_SEED)
    
    train_sizes = [5, 10, 20, 30, 40, 50, 60, 70, 80]
    
    results = learning_curve_analysis(
        dataset_class=MicrotubuleDataset,
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        train_sizes=train_sizes,
        n_repeats=1,
        n_folds=5,
        device=DEVICE
    )
    
    np.save('learning_curve_results.npy', results)
    print("\nResults saved as 'learning_curve_results.npy'")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()