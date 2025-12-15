import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import monai  # For pretrained models and medical imaging utilities

class PretrainedUNet3D(nn.Module):
    def __init__(self, pretrained_path=None, num_classes=1, freeze_encoder=True):
        super().__init__()
        # Initialize with MONAI's pretrained UNet
        self.model = monai.networks.nets.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            num_res_units=2
        )
        
        if pretrained_path:
            # Load pretrained weights
            if pretrained_path == 'medical':
                # Use MONAI's pretrained medical imaging model
                pretrained = monai.networks.nets.UNet.from_pretrained(
                    'spleen_ct_segmentation',
                    spatial_dims=3
                )
                # Copy weights from pretrained model
                self.model.load_state_dict(pretrained.state_dict())
            else:
                # Load custom pretrained weights
                self.model.load_state_dict(torch.load(pretrained_path))
        
        if freeze_encoder:
            # Freeze encoder layers
            for name, param in self.model.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder layers for full fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)

def fine_tune_model(model, train_loader, val_loader, config):
    """
    Fine-tune a pretrained model with progressive unfreezing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Training configuration
    num_epochs = config.get('num_epochs', 100)
    initial_lr = config.get('initial_lr', 1e-4)
    unfreeze_epoch = config.get('unfreeze_epoch', 30)  # When to unfreeze encoder
    
    # Initialize loss function - combine MSE for SDT with edge detection loss
    criterion = lambda pred, target: (
        F.mse_loss(pred, target) + 
        0.5 * F.l1_loss(pred, target) +  # Better edge preservation
        0.1 * edge_loss(pred, target)     # Custom edge detection loss
    )
    
    # First phase: Train only decoder
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Unfreeze encoder at specified epoch
        if epoch == unfreeze_epoch:
            print("Unfreezing encoder layers...")
            model.unfreeze_encoder()
            # Reinitialize optimizer with all parameters
            optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr * 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2
            )
        
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        val_loss /= len(val_loader)
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'best_model_finetuned.pth')

def edge_loss(pred, target):
    """
    Custom loss function to preserve edges in SDT
    """
    # Sobel filters for 3D edge detection
    sobel_x = torch.tensor([[[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]]]).to(pred.device)
    sobel_y = torch.tensor([[[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]]]).to(pred.device)
    
    # Compute gradients
    pred_grad_x = F.conv3d(pred, sobel_x.view(1, 1, 3, 3, 1), padding=1)
    pred_grad_y = F.conv3d(pred, sobel_y.view(1, 1, 3, 3, 1), padding=1)
    target_grad_x = F.conv3d(target, sobel_x.view(1, 1, 3, 3, 1), padding=1)
    target_grad_y = F.conv3d(target, sobel_y.view(1, 1, 3, 3, 1), padding=1)
    
    return F.mse_loss(pred_grad_x, target_grad_x) + F.mse_loss(pred_grad_y, target_grad_y)

# Example usage
def main():
    # Load datasets
    train_dataset = MicrotubuleDataset('path/to/train/data')
    val_dataset = MicrotubuleDataset('path/to/val/data')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize pretrained model
    model = PretrainedUNet3D(
        pretrained_path='medical',  # Use MONAI's pretrained model
        freeze_encoder=True  # Start with frozen encoder
    )
    
    # Configure fine-tuning
    config = {
        'num_epochs': 100,
        'initial_lr': 1e-4,
        'unfreeze_epoch': 30
    }
    
    # Fine-tune model
    fine_tune_model(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main()