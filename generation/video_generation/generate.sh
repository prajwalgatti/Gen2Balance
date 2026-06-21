#!/bin/bash
# Generate a slice of the prompt manifest with Wan 2.1.
# Usage: bash generate.sh START_IDX END_IDX [SECONDS_PER_VIDEO] [BATCH_SIZE]

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 START_IDX END_IDX [SECONDS_PER_VIDEO] [BATCH_SIZE]"
    exit 1
fi

START_IDX=$1
END_IDX=$2
SECONDS_PER_VIDEO=${3:-4}
BATCH_SIZE=${4:-4}

# --- Edit these paths ---
# Model: HF id (auto-downloaded) or a local path to Wan2.1-T2V-14B-Diffusers.
MODEL_PATH="Wan-AI/Wan2.1-T2V-14B-Diffusers"
# Manifest produced by build_manifest.py:
PROMPTS_FILE="video_prompts.json"
# Output directory for the generated <label_id>_<sample_idx>.mp4 files:
OUTPUT_DIR="./generated_videos"

echo "Generating manifest indices ${START_IDX}..${END_IDX} (batch ${BATCH_SIZE}, ${SECONDS_PER_VIDEO}s) on CUDA ${CUDA_VISIBLE_DEVICES}"

time python generate_videos.py \
    --start_idx="${START_IDX}" \
    --end_idx="${END_IDX}" \
    --batch_size="${BATCH_SIZE}" \
    --output_dir="${OUTPUT_DIR}" \
    --model_path="${MODEL_PATH}" \
    --prompts_file="${PROMPTS_FILE}" \
    --seconds="${SECONDS_PER_VIDEO}"

echo "Done for indices ${START_IDX} to ${END_IDX}"
