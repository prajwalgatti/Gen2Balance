#!/bin/bash
#SBATCH --job-name=K100LT_eval
#SBATCH --output=output/K100LT/stage2/eval/logs/%j.out
#SBATCH --error=output/K100LT/stage2/eval/logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

# Path to save checkpoints
OUTPUT_DIR='output/K100LT/stage2/eval/'
# Path to annotation files (train.csv/test.csv)
DATA_PATH='data/annotations/K100-LT/real'
# Path to K100LT stage 2 checkpoint for evaluation
MODEL_PATH='output/K100LT/stage2/checkpoint-19.pth' # Adjust this path to your actual checkpoint

# Root directory containing the downloaded/generated videos (see README, Data setup).
# Annotation CSV paths are relative to this root.
DATA_ROOT='data'

# Set OpenMP threads
export OMP_NUM_THREADS=1

# Generate a random port between 20000 and 60000
export MASTER_PORT=$(shuf -i 20000-60000 -n 1)

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Output Dir: $OUTPUT_DIR"
echo "Starting training at: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "RAM: $(($(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024))GB"
echo "Randomly selected Master Port: $MASTER_PORT"
echo ""

# print GPU information
nvidia-smi --list-gpus

# Load environment
source ~/miniconda3/bin/activate
conda activate gen2bal
cd ~/projects/Gen2Balance/ # Adjust this path to your actual Gen2Balance project directory

torchrun --nproc_per_node=1 \
    --master_port ${MASTER_PORT} run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --data_set Kinetics-100 \
    --nb_classes 100 \
    --batch_size 150 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 50 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_sample 2 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --dist_eval \
    --eval

echo "Evaluation completed at: $(date)"