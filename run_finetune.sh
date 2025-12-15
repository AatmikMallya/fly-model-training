#!/bin/bash
#SBATCH -t 23:00:00
#SBATCH --mem=32G
#SBATCH -p scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH -J unet_pretrain
#SBATCH -o %j.out
#SBATCH -e %j.err

# Change to the correct directory
cd /home/am3833/fly

module load Python/3.10.8-GCCcore-12.2.0

# Activate the virtual environment
source /home/am3833/jupyterlab_venv/bin/activate

cd segmentation
python3 unet_pretrain.py
# python3 learning_curve_analysis.py
