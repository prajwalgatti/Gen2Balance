#!/bin/bash
#SBATCH --job-name=wan_gen
#SBATCH --output=logs/video_gen_%A_%a.out
#SBATCH --error=logs/video_gen_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=7:00:00
#SBATCH --array=0-137        # set to ceil(num_prompts / VIDEOS_PER_JOB) - 1

# Each array task generates a contiguous slice of the manifest.
VIDEOS_PER_JOB=56
START_IDX=$((SLURM_ARRAY_TASK_ID * VIDEOS_PER_JOB))
END_IDX=$((START_IDX + VIDEOS_PER_JOB))

mkdir -p logs
nvidia-smi --list-gpus

# Activate your environment here (conda/venv/container), then run:
bash generate.sh "${START_IDX}" "${END_IDX}"
