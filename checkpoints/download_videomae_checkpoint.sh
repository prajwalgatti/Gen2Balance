#!/bin/bash
# Usage: bash download_videomae_checkpoint.sh

# Code to download the VideoMAE checkpoint from Google Drive (https://drive.google.com/file/d/1tEhLyskjb755TJ65ptsrafUG2llSwQE1/view?usp=sharing)
# In case link is broken, find checkpoint here: https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md  [Kinetics-400, ViT-B, 1600 epochs, 16x5x3 frames, pretrained checkpoint]

# Check if gdown is installed, if not, install it
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found, installing it now..."
    pip install gdown
fi

# Download the checkpoint using gdown
# Filename matches MODEL_PATH in the stage-1 SLURM scripts (checkpoints/mae_pretrain_vit_base.pth)
FILE_ID="1tEhLyskjb755TJ65ptsrafUG2llSwQE1"
OUTPUT_PATH="./mae_pretrain_vit_base.pth"
gdown $FILE_ID --output $OUTPUT_PATH
echo "Checkpoint downloaded successfully to $OUTPUT_PATH"