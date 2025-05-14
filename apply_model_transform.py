import numpy as np
import torch.nn.functional as F
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import os

# --- Configuration Variables ---
# MODEL_NAME = "Qwen/QwQ-32B-Preview"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

STEERING_VECTOR_FILE_PATH = "steering_vectors_Wait.pt" # Path to your .pt file
STEERING_VECTOR_LAYER_IDX_TO_USE = 10 # Layer index from which to get the steering vector
TRANSFORM_NAME = "orthogonalization"  # Descriptive name for the transformation

# Output configuration
LOCAL_MODEL_SAVE_DIR = "my_models" # Local directory to save models
HUGGINGFACE_USERNAME = "dipikakhullar" # Your Hugging Face username

# Determine device map (adjust if you have specific GPU requirements)
# Example: "cuda:0", "auto", {"": "cuda:0"}
DEVICE_MAP = "cuda:0" 
# --- End Configuration Variables ---

# --- Derived Names ---
MODEL_NAME_SUFFIX = MODEL_NAME.split('/')[-1]
TRANSFORMED_MODEL_NAME = f"{MODEL_NAME_SUFFIX}-{TRANSFORM_NAME}-all-layers-steered-by-layer-{STEERING_VECTOR_LAYER_IDX_TO_USE}"
LOCAL_SAVE_PATH = os.path.join(LOCAL_MODEL_SAVE_DIR, TRANSFORMED_MODEL_NAME)
HUGGINGFACE_REPO_ID = f"{HUGGINGFACE_USERNAME}/{TRANSFORMED_MODEL_NAME}"
# --- End Derived Names ---

def get_orthogonalized_matrix(W, direction):
    """
    Orthogonalize weight matrix W w.r.t. a direction vector.
    W: torch.Tensor of shape [out_dim, in_dim]
    direction: torch.Tensor of shape [out_dim]
    """
    direction = direction / direction.norm()  # normalize to a unit vector
    projection = torch.ger(direction, direction)  # take outer product
    return W - torch.mm(projection, W)  # subtract project part


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda:3",
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
    ),
)
model.eval()

steering_vectors = torch.load(STEERING_VECTOR_FILE_PATH)

# Sort by absolute mean of the steering vectors
activation_scored = sorted(
    [
        (layer_idx, vec / vec.norm()) for layer_idx, vec in steering_vectors.items()
        if isinstance(vec, torch.Tensor)
    ],
    key=lambda x: abs(x[1].float().mean().item()),  # Convert to float32 for better precision
    reverse=True,
)

# Display top-k layers with strongest steering signal
top_k = 50
# print("Top-k layers by abs mean activation:")
for i, (layer_id, vec) in enumerate(activation_scored[:top_k]):
    print(f"#{i}: Layer {layer_id:<2}      | Mean = {abs(vec.float().mean()):.4f}      | Norm = {vec.float().norm():.2f}")


# Apply to all layers
for i, layer in enumerate(model.model.layers):
    device = layer.self_attn.o_proj.weight.device
    print(device)
    steering_vector = steering_vectors[STEERING_VECTOR_LAYER_IDX_TO_USE].to(device)

    # Convert to float16 before modification
    W_o_proj = layer.self_attn.o_proj.weight.data.to(device = device, dtype=torch.float16)
    W_down_proj = layer.mlp.down_proj.weight.data.to(device = device, dtype=torch.float16)
    
    # Apply orthogonalization
    W_o_proj_orth = get_orthogonalized_matrix(W_o_proj, steering_vector)
    W_down_proj_orth = get_orthogonalized_matrix(W_down_proj, steering_vector)

    # Restore original dtype
    layer.self_attn.o_proj.weight.data = W_o_proj_orth.to(layer.self_attn.o_proj.weight.dtype)
    layer.mlp.down_proj.weight.data = W_down_proj_orth.to(layer.mlp.down_proj.weight.dtype)

    print(f"Orthogonalized layer {i} using layer {STEERING_VECTOR_LAYER_IDX_TO_USE}'s vector")

model.save_pretrained(LOCAL_SAVE_PATH)
tokenizer.save_pretrained(LOCAL_SAVE_PATH)
print("Saved locally.")

login(token="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")

model.push_to_hub(HUGGINGFACE_REPO_ID)
tokenizer.push_to_hub(HUGGINGFACE_REPO_ID)
print(f"--- Transformation Complete ---")

