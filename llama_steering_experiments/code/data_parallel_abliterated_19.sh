#!/bin/bash
#SBATCH --job-name=abl-19            # Job name
#SBATCH --output=/fsx/ubuntu/users/dikhulla/cot-reasoning/jobs/abl_19.out  # Stdout log
#SBATCH --error=/fsx/ubuntu/users/dikhulla/cot-reasoning/jobs/abl_19.err   # Stderr log
#SBATCH --partition=ml-p4d-24xlarge-us-east-2a       # Partition with GPU
#SBATCH --ntasks=1                         # One task
#SBATCH --cpus-per-task=4                  # CPUs per task
#SBATCH --mem=64G                          # Memory
#SBATCH --time=24:00:00                    # Runtime
#SBATCH --nodelist=ip-10-4-120-93

echo "Starting job on $(hostname)"
echo "Initializing Conda..."
pwd
echo "GPUs available to this job:"
nvidia-smi --query-gpu=name,index --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

source /fsx/ubuntu/miniconda3/etc/profile.d/conda.sh

conda activate task-vector

echo "Conda environment activated."
echo "Current Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python path: $(which python)"

# Print GPU info
echo "Checking GPU availability..."
nvidia-smi || echo "No GPU detected!"

# Run the script
echo "Running data_parallel_abliterated_19"
python /fsx/ubuntu/users/dikhulla/cot-reasoning/data_parallel_abliterated_19.py

echo "Job finished!"
