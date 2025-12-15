import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import wandb
import random
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    OneOf,
)
import datetime
import platform
from dataset import MicrotubuleDataset

# ================== Configuration Variables ==================
EXPERIMENT_NAME = 'swinunetr_finetuning'
# Data paths
IMAGE_DIR = 'training/subvols/image'
LABEL_DIR = 'training/labeled_sdt'
CHECKPOINT_DIR = 'checkpoints'

# Training parameters
BATCH_SIZE = 1
NUM_EPOCHS = 200
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0
RANDOM_SEED = 42
PATIENCE = 80

# Model parameters
SPATIAL_DIMS = 3
IN_CHANNELS = 1
OUT_CHANNELS = 1
FEATURE_SIZE = 96
DROP_RATE = 0.3
ATTN_DROP_RATE = 0.3
DROPOUT_PATH_RATE = 0.3

# Device configuration
DEVICE = torch.device('cuda')
print(f"Using device: {DEVICE}")

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

hyperparameters = {
    'experiment_name': EXPERIMENT_NAME,
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'feature_size': FEATURE_SIZE,
    'drop_rate': DROP_RATE,
    'attn_drop_rate': ATTN_DROP_RATE,
    'dropout_path_rate': DROPOUT_PATH_RATE,
    'device': str(DEVICE),
    'torch_version': torch.__version__,
    'python_version': platform.python_version(),
    'training_start_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

def compute_segmentation_metrics(outputs, labels, threshold=0.2):
    """
    Compute comprehensive segmentation metrics from SDT predictions.
    """
    # Convert SDTs to binary masks
    pred_masks = (outputs <= threshold).float()
    true_masks = (labels <= threshold).float()
    
    # Compute basic counts for metrics
    true_positives = (pred_masks * true_masks).sum()
    true_negatives = ((1 - pred_masks) * (1 - true_masks)).sum()
    false_positives = (pred_masks * (1 - true_masks)).sum()
    false_negatives = ((1 - pred_masks) * true_masks).sum()
    
    # Add small epsilon to avoid division by zero
    eps = 1e-6
    
    # Compute precision and recall
    precision = (true_positives + eps) / (true_positives + false_positives + eps)
    recall = (true_positives + eps) / (true_positives + false_negatives + eps)
    
    # Compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    
    # Compute accuracy
    total_pixels = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total_pixels
    
    # Compute balanced accuracy
    sensitivity = recall
    specificity = (true_negatives + eps) / (true_negatives + false_positives + eps)
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # Compute dice
    dice = (2 * true_positives + eps) / (2 * true_positives + false_positives + false_negatives + eps)
    
    return {
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'accuracy': accuracy.item(),
        'balanced_accuracy': balanced_accuracy.item(),
        'true_positives': true_positives.item(),
        'true_negatives': true_negatives.item(),
        'false_positives': false_positives.item(),
        'false_negatives': false_negatives.item()
    }

def format_metrics(metrics, prefix=''):
    """Format metrics for pretty printing"""
    metric_strings = []
    for key in ['loss', 'dice', 'precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy']:
        if key in metrics:
            metric_strings.append(f"{key.capitalize()}: {metrics[key]:.4f}")
    return f"{prefix} - " + ", ".join(metric_strings)

def save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, checkpoint_dir, filename):
    """Helper function to save model checkpoints"""
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
    """Helper function to log final training metrics"""
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
    wandb.log(final_metrics)


# ================== Model Definition ==================
class TanhSwinUNETR(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, feature_size=48, 
                 drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0):
        super().__init__()
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate
        )
    
    def forward(self, x):
        x = self.model(x)
        return torch.tanh(x)

def initialize_model(pretrained=True):
    """Initialize SwinUNETR with pretrained weights"""
    model = TanhSwinUNETR(
        img_size=(96, 96, 96),
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        feature_size=FEATURE_SIZE,
        drop_rate=DROP_RATE,
        attn_drop_rate=ATTN_DROP_RATE,
        dropout_path_rate=DROPOUT_PATH_RATE
    )
    
    if pretrained:
        try:
            from monai.apps import download_and_extract
            resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt"
            weightsfile = download_and_extract(resource)
            
            pretrained_weights = torch.load(weightsfile, map_location='cuda')
            model_dict = model.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_weights.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            model_dict.update(pretrained_dict)
            model.model.load_state_dict(model_dict, strict=False)
            print("Loaded pretrained weights")
            
            # Freeze encoder layers
            for name, param in model.model.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = False
                    print(f"Froze {name}")
                    
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Initializing with random weights")

    # Move model to CUDA and wrap with DataParallel if multiple GPUs available
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model = model.to(DEVICE)
    
    return model

# ================== Training Functions ==================
def get_transforms():
    return Compose([
        OneOf([
            Compose([]),  # No flips
            Compose([RandFlipd(keys=['image', 'label'], prob=1.0, spatial_axis=0)]),
            Compose([RandFlipd(keys=['image', 'label'], prob=1.0, spatial_axis=1)]),
            Compose([RandFlipd(keys=['image', 'label'], prob=1.0, spatial_axis=2)]),
        ], weights=[1/4] * 4),
        
        OneOf([
            Compose([]),  # No rotation
            Compose([RandRotate90d(keys=['image', 'label'], prob=1.0, max_k=1, spatial_axes=(0, 1))]),
            Compose([RandRotate90d(keys=['image', 'label'], prob=1.0, max_k=1, spatial_axes=(1, 2))]),
            Compose([RandRotate90d(keys=['image', 'label'], prob=1.0, max_k=1, spatial_axes=(0, 2))]),
        ], weights=[1/4] * 4)
    ])

def setup_training(model):
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only include unfrozen parameters
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': LEARNING_RATE * 0.1},
        {'params': decoder_params, 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    return optimizer, scheduler

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    wandb.init(project="microtubule-segmentation", name=EXPERIMENT_NAME, config=hyperparameters)
    best_val_loss = float('inf')
    best_val_dice = 0.0
    patience_counter = 0
    training_start_time = datetime.datetime.now()
    epoch_times = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = datetime.datetime.now()
        
        # Training phase
        model.train()
        train_loss = 0
        train_metrics = {
            'dice': 0, 'precision': 0, 'recall': 0, 'f1': 0,
            'accuracy': 0, 'balanced_accuracy': 0
        }
        
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Compute and accumulate metrics
            batch_metrics = compute_segmentation_metrics(outputs.detach(), labels)
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key]
        
        avg_train_loss = train_loss / len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        train_metrics['loss'] = avg_train_loss
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = {
            'dice': 0, 'precision': 0, 'recall': 0, 'f1': 0,
            'accuracy': 0, 'balanced_accuracy': 0
        }
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                batch_metrics = compute_segmentation_metrics(outputs, labels)
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]
        
        avg_val_loss = val_loss / len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        val_metrics['loss'] = avg_val_loss
        
        # Calculate epoch time
        epoch_duration = datetime.datetime.now() - epoch_start_time
        epoch_times.append(epoch_duration.total_seconds())
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Logging
        wandb.log({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_dice': train_metrics['dice'],
            'val_dice': val_metrics['dice'],
            'train_precision': train_metrics['precision'],
            'val_precision': val_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_recall': val_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'val_f1': val_metrics['f1'],
            'train_accuracy': train_metrics['accuracy'],
            'val_accuracy': val_metrics['accuracy'],
            'train_balanced_accuracy': train_metrics['balanced_accuracy'],
            'val_balanced_accuracy': val_metrics['balanced_accuracy'],
            'epoch': epoch + 1,
            'epoch_time': epoch_duration.total_seconds(),
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Print progress
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
        print(format_metrics(train_metrics, 'Train'))
        print(format_metrics(val_metrics, 'Val  '), end=' ')
        
        # Save best models
        if val_metrics['dice'] > best_val_dice:
            print('* New best model (Dice) ', end=' ')
            best_val_dice = val_metrics['dice']
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, avg_val_loss,
                val_metrics, CHECKPOINT_DIR, 'best_dice_model.pt'
            )
        
        if avg_val_loss < best_val_loss:
            print('* New best model (Loss) ', end=' ')
            best_val_loss = avg_val_loss
            save_checkpoint(
                model, optimizer, epoch, avg_val_loss,
                val_metrics, CHECKPOINT_DIR, 'best_loss_model.pt'
            )
        else:
            patience_counter += 1
            print()
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Log final metrics
    training_duration = datetime.datetime.now() - training_start_time
    log_final_metrics(
        training_duration, epoch_times, epoch + 1,
        best_val_loss, best_val_dice, CHECKPOINT_DIR
    )
    
    wandb.finish()

# ================== Main Function ==================
def main():
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize model
    model = initialize_model(pretrained=True)
    model = model.to(DEVICE)
    
    # Print parameter summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Create datasets
    transforms = get_transforms()
    
    dataset = MicrotubuleDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR
    )
    
    # Split indices
    dataset_length = len(dataset)
    val_indices = list(range(0, dataset_length, 5))
    train_indices = [i for i in range(dataset_length) if i not in val_indices]
    
    # Create train/val datasets
    train_dataset = MicrotubuleDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        indices=train_indices,
        transforms=transforms
    )
    
    val_dataset = MicrotubuleDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        indices=val_indices,
        transforms=transforms
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
    
    # Setup training
    optimizer, scheduler = setup_training(model)
    
    # Loss function
    criterion = nn.L1Loss()
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

if __name__ == "__main__":
    main()