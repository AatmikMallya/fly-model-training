# paths.py
"""
Centralized path configuration for the microtubule segmentation pipeline.

============================================================================
HOW TO SET UP FOR A NEW SYSTEM
============================================================================

1. Update BASE_DIR to point to your segmentation directory
2. Ensure Google Cloud credentials are set up (see GOOGLE_CREDENTIALS below)
3. For training: create training data directories under BASE_DIR
4. For inference: update batch_pipeline.sh with your paths

============================================================================
"""
import os

# ============================================================================
# CONFIGURE THESE TWO PATHS FOR YOUR ENVIRONMENT
# ============================================================================

# Base directory containing the segmentation code and data
# Example: /home/your_username/project/fly/segmentation
BASE_DIR = '/home/am3833/project/fly/segmentation'

# Google Cloud credentials for tensorstore access to Hemibrain data
# After running `gcloud auth application-default login`, this is typically:
#   ~/.config/gcloud/application_default_credentials.json
# On some clusters, you may need to copy this file to your project directory
GOOGLE_CREDENTIALS = '~/.config/gcloud/application_default_credentials.json'

# ============================================================================
# DERIVED PATHS (auto-computed from BASE_DIR, usually no changes needed)
# ============================================================================

# Parent directory (one level up from segmentation/)
PARENT_DIR = os.path.dirname(BASE_DIR)

# Training data directories (relative to BASE_DIR)
TRAINING_IMAGE_DIR = os.path.join(BASE_DIR, 'training/subvols/image')
TRAINING_LABEL_DIR = os.path.join(BASE_DIR, 'training/labeled_binary_large')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

# Inference pipeline directories (sibling directories to segmentation/)
PREPROCESS_OUTPUT_DIR = os.path.join(PARENT_DIR, 'preprocess_output')
INFERENCE_OUTPUT_DIR = os.path.join(PARENT_DIR, 'unet_output')
FINAL_OUTPUT_DIR = os.path.join(PARENT_DIR, 'final_output')

# Model checkpoint for inference
MODEL_CHECKPOINT = os.path.join(PARENT_DIR, 'best_final_model.pt')

# Utility modules directory
UTIL_DIR = os.path.join(BASE_DIR, 'util_files')
