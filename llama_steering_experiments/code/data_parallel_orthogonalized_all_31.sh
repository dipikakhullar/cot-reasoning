#!/bin/bash
#SBATCH --job-name=all-31          # Job name
#SBATCH --output=/fsx/ubuntu/users/dikhulla/cot-reasoning/jobs/orthogonalized_all_31.out  # Stdout log
#SBATCH --error=/fsx/ubuntu/users/dikhulla/cot-reasoning/jobs/orthogonalized_all_31.err   # Stderr log
#SBATCH --partition=ml-p4d-24xlarge-us-east-2a       # Partition with GPU
#SBATCH --ntasks=1                         # One task
#SBATCH --cpus-per-task=4                  # CPUs per task
#SBATCH --mem=64G                          # Memory
#SBATCH --time=24:00:00                    # Runtime
#SBATCH --nodelist=ip-10-4-118-13

echo "Starting job on $(hostname)"
echo "Initializing Conda..."
pwd
echo "GPUs available to this job:"
nvidia-smi --query-gpu=name,index --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


# Manually source Conda before activating environment
# source /fsx/ubuntu/miniconda3/condabin/conda
source /fsx/ubuntu/miniconda3/etc/profile.d/conda.sh

conda activate task-vector

echo "Conda environment activated."
echo "Current Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python path: $(which python)"

# Print GPU info
echo "Checking GPU availability..."
nvidia-smi || echo "No GPU detected!"

# Run the script
echo "Running data_parallel_orthogonalized_all_31"
python -u /fsx/ubuntu/users/dikhulla/cot-reasoning/data_parallel_orthogonalized_all_31.py

echo "Job finished!"
