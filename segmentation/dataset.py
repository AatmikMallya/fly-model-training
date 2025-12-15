import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class MicrotubuleDataset(Dataset):
    def __init__(self, image_dir, label_dir, indices=None, transforms=None):
        """
        Initialize the dataset and load all data into RAM.
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        
        # Precompute fixed crop indices (100 -> 96)
        self.start_idx = 2  # (100 - 96) // 2
        self.end_idx = 98   # start_idx + 96
        
        # Gather all valid image-label pairs
        label_files = {p.stem: p for p in self.label_dir.glob("*.npy")}
        all_pairs = [
            (img_path, label_files[img_path.stem])
            for img_path in sorted(self.image_dir.glob("*.npy"))
            if img_path.stem in label_files
        ]
        
        if not all_pairs:
            raise RuntimeError("No valid image-label pairs found")
        
        # Apply indices filtering if provided
        if indices is not None:
            all_pairs = [all_pairs[i] for i in indices]
            # print(all_pairs)
            # 2045541
        
        self.valid_pairs = all_pairs
        
        # Pre-load all data into memory
        self.data = []
        for img_path, lbl_path in self.valid_pairs:
            image = np.load(img_path).astype(np.float32)[
                self.start_idx:self.end_idx,
                self.start_idx:self.end_idx,
                self.start_idx:self.end_idx
            ]
            label = np.load(lbl_path).astype(np.float32)[
                self.start_idx:self.end_idx,
                self.start_idx:self.end_idx,
                self.start_idx:self.end_idx
            ]
            
            # Add channel dimension
            image = image[None]  # shape: (1, 96, 96, 96)
            label = label[None]
            
            image_t = torch.from_numpy(image)
            label_t = torch.from_numpy(label)
            
            self.data.append((image_t, label_t))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        
        # Apply transforms if available
        if self.transforms:
            data = self.transforms({'image': image, 'label': label})
            image, label = data['image'], data['label']
        
        return image, label
