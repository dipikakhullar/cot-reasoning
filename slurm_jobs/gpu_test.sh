#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --output=gpu_test.out
#SBATCH --error=gpu_test.err
#SBATCH --time=00:10:00
#SBATCH --partition=ml-p4d-24xlarge-us-east-2a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "Running on node: $(hostname)"
echo "Job started at: $(date)"

echo "GPU info:"
nvidia-smi

echo "Testing PyTorch CUDA availability..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Current device:', torch.cuda.current_device())"

echo "Job finished at: $(date)"
