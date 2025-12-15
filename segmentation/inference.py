# inference.py
# ================== Imports ==================
import numpy as np
import argparse
import os
import importlib
from time import time
import torch
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet

np.set_printoptions(precision=5, suppress=True)

# Local imports
def import_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

home_dir = '/home/am3833/project/fly/segmentation'
voxel_utils = import_module('voxel_utils', f'{home_dir}/util_files/voxel_utils.py')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '~/.config/gcloud/application_default_credentials.json'

class MicrotubuleInferenceDataset(Dataset):
    def __init__(self, gray_data: np.ndarray):
        assert len(gray_data.shape) == 4 and gray_data.shape[1:] == (96, 96, 96), \
            f"Expected shape (N, 96, 96, 96), got {gray_data.shape}"
        
        # Store as float32 to avoid repeated conversions
        if gray_data.dtype != np.float32:
            self.gray_data = gray_data.astype(np.float32, copy=False)
        else:
            self.gray_data = gray_data
            
    def __len__(self):
        return len(self.gray_data)
    
    def __getitem__(self, idx):
        volume = self.gray_data[idx]
        return torch.from_numpy(volume).unsqueeze(0)

def initialize_model():
    """
    Initialize a UNet model with predefined configurations.
    
    Returns:
        UNet: Configured UNet model instance
    """
    model_params = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "channels": (32, 64, 128, 256),
        "kernel_size": 3,
        "num_res_units": 6,
        "strides": (2,2,2),
        "dropout": 0,
        "norm": ("INSTANCE", {"affine": True})
    }
    
    model = UNet(**model_params)
    print("Initialized UNet model")
    return model

def load_trained_model(checkpoint_path, device):
    """Load and prepare model for inference"""
    model = initialize_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best val loss: {checkpoint['val_loss']:.4f}")
    
    return model.to(device)

def preprocess_data(gray_data, seg_data):
    """Preprocess grayscale data with segmentation mask"""
    if gray_data.dtype != np.float32:
        gray_data = gray_data.astype(np.float32, copy=False)
    gray_data *= (1/255)  # Normalize to [0,1]
    np.multiply(gray_data, seg_data, out=gray_data)  # Apply segmentation mask
    return gray_data

def run_inference(model, gray_data, batch_size, device):
    """Run model inference on preprocessed data"""
    dataset = MicrotubuleInferenceDataset(gray_data)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=device != 'cpu'
    )
    
    n_samples = len(gray_data)
    # Pre-allocate output arrays
    raw_predictions = np.empty((n_samples, 96, 96, 96), dtype=np.float32)
    
    print(f"Running inference on {n_samples} neuron subvolumes...")
    
    with torch.no_grad():
        start_idx = 0
        for batch_idx, batch in enumerate(dataloader):
            batch_size = batch.size(0)
            batch = batch.to(device)
            
            # Get predictions
            pred = torch.sigmoid(model(batch))
            pred_np = pred.cpu().numpy()
            
            # Store in pre-allocated arrays
            end_idx = start_idx + batch_size
            raw_predictions[start_idx:end_idx] = pred_np.squeeze(1)
            
            start_idx = end_idx
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {start_idx}/{n_samples} subvolumes")
    
    return raw_predictions

def estimate_optimal_batch_size(sample_shape, model, device='cuda'):
    if device == 'cpu':
        return 32  # Default for CPU
        
    import torch.cuda as cuda
    
    total_memory = cuda.get_device_properties(0).total_memory
    free_memory = cuda.mem_get_info()[0]
    
    input_size = np.prod(sample_shape) * 4  # float32 size
    forward_pass_memory = input_size * 4    # UNET memory multiplier
    output_size = input_size
    
    memory_per_sample = (input_size + forward_pass_memory + output_size) * 1.2
    usable_memory = free_memory * 0.9
    max_batch_size = int(usable_memory / memory_per_sample)
    
    return max(1, max_batch_size)  # Just ensure at least batch size 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing boxes and seg_data from preprocessing')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save prediction results')
    parser.add_argument('--checkpoint_path', type=str, 
                        default='best_model/best_final_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--bodyId', type=int, required=True,
                        help='Body ID of the neuron being processed')
    parser.add_argument('--max_batch_size', type=int, default=None,
                        help='Maximum batch size to use (optional)')
    args = parser.parse_args()

    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Load preprocessing results
        boxes = np.load(f"{args.input_dir}/boxes_{args.bodyId}.npy")
        seg_data = np.load(f"{args.input_dir}/seg_data_{args.bodyId}.npy")
        
        print(f"Loaded {len(boxes)} boxes for bodyId {args.bodyId}")
        
        # Get and preprocess grayscale data
        print("Getting grayscale data...")
        gray_data = voxel_utils.get_subvols_batched(boxes, 'grayscale_clahe')
        gray_data = preprocess_data(gray_data, seg_data)
        
        # Load model and estimate batch size
        print("Loading model...")
        model = load_trained_model(args.checkpoint_path, device)
        
        if args.max_batch_size is None and device == 'cuda':
            batch_size = estimate_optimal_batch_size((96,96,96), model, device)
            print(f"Estimated optimal batch size: {batch_size}")
        else:
            batch_size = args.max_batch_size or 32
            print(f"Using specified batch size: {batch_size}")
        
        print("Running inference...")
        start_time = time()
        raw_predictions = run_inference(
            model, gray_data, batch_size, device
        )
        print(f"Inference completed in {time() - start_time:.2f} seconds")
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(f"{args.output_dir}/raw_predictions_{args.bodyId}.npy", raw_predictions)
        
        print(f"Saved predictions for bodyId {args.bodyId}")
        
    except Exception as e:
        print(f"Error processing bodyId {args.bodyId}: {str(e)}")
        raise


