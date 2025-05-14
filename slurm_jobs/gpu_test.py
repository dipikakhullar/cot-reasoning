# Add this before creating LLM instances to verify GPU connectivity
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    for j in range(torch.cuda.device_count()):
        if i != j:
            can_access = torch.cuda.can_device_access_peer(i, j)
            print(f"GPU {i} can access GPU {j}: {can_access}")
        