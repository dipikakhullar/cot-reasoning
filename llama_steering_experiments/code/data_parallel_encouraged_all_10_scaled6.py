import os
import multiprocessing as mp
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import torch
import os
import sys
sys.stdout.reconfigure(line_buffering=True)


STEP_BY_STEP_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}.\n"

def generate_steering_prefill(row):
    """
    Returns the user content (first in message_pairs) + step-by-step suffix.
    """
    messages = row["message_pairs"]
    if len(messages) < 2:
        return None  # Return None if there aren't enough messages
    
    # messages[0] is the user message
    return messages[0]["content"] + STEP_BY_STEP_SUFFIX

def compute_steering_vector(activations_dict):
    steering_vectors = {}
    for layer in activations_dict["backtracking"]:
        backtracking_activations = [torch.tensor(a) if isinstance(a, np.ndarray) else a for a in activations_dict["backtracking"][layer]]
        non_backtracking_activations = [torch.tensor(a) if isinstance(a, np.ndarray) else a for a in activations_dict["non_backtracking"][layer]]

        backtracking_activations = torch.stack(backtracking_activations)  
        non_backtracking_activations = torch.stack(non_backtracking_activations)  

        mean_backtracking = backtracking_activations.mean(dim=0)  
        mean_non_backtracking = non_backtracking_activations.mean(dim=0)  

        steering_vectors[layer] = mean_backtracking - mean_non_backtracking  

    return steering_vectors

def set_steering_params(vector, scale=1.0):
    """Set global steering parameters."""
    global _STEERING_VECTOR, _STEERING_SCALE
    _STEERING_VECTOR = vector
    _STEERING_SCALE = scale

def reset_steering_params():
    """Reset global steering parameters."""
    global _STEERING_VECTOR, _STEERING_SCALE
    _STEERING_VECTOR = None
    _STEERING_SCALE = 1.0

def simple_steering_hook(module, input, output):
    # print("INSIDE HOOK", flush=True)

    """
    A simple hook that applies the global steering vector.
    This hook has the exact signature PyTorch expects.
    """
    global _STEERING_VECTOR, _STEERING_SCALE
    # print(f"[HOOK] Called on module: {module.__class__.__name__}")
    # print(f"[HOOK] Vector is None? {_STEERING_VECTOR is None}")
    
    if _STEERING_VECTOR is None:
        return output  # No steering to apply
    
    if isinstance(output, tuple):
        output_tensor = output[0]
        device = output_tensor.device
        steering_vector_device = _STEERING_VECTOR.to(device)
        # print("INSIDE THE FORWARD HOOK")
        # print("INSIDE HOOK HELLO", flush=True)

        return (output_tensor + _STEERING_SCALE * steering_vector_device,) + output[1:]
    else:
        return output + _STEERING_SCALE * _STEERING_VECTOR.to(output.device)


def generate_text(model, tokenizer, input_text, steering_vector=None, scale=1.0, max_length=2000, layer_index=5):
    """
    Generates text with optional steering vector applied.
    """
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    hook = None
    
    try:
        if steering_vector is not None:
            # layer = model.model.layers[layer_index]
            layer = model.model.layers[layer_index]

            
            # Set global steering parameters
            set_steering_params(steering_vector, scale)
            
            # Register the simple hook
            hook = layer.register_forward_hook(simple_steering_hook)
        
        # Generate output
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_length=max_length,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # recommended for deterministic outputs
                )
            # output_tokens = model.generate(**inputs, max_length=max_length, use_cache=True)
            
        return tokenizer.decode(output_tokens[0], skip_special_tokens=True, skip_prompt=True)
    
    finally:
        # Clean up
        if hook is not None:
            hook.remove()
        reset_steering_params()


def generate_text_batch(model, tokenizer, input_texts, steering_vector=None, scale=1.0, max_length=2000, layer_index=5):
    """
    Batched generation with optional steering vector hook at one layer.
    Assumes all prompts in the batch share the same steering configuration.
    """
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    hook = None

    try:
        if steering_vector is not None:
            layer = model.model.layers[layer_index]
            set_steering_params(steering_vector, scale)
            hook = layer.register_forward_hook(simple_steering_hook)

        with torch.no_grad():
            output_tokens = model.generate(**inputs, max_length=max_length, use_cache=True)

        decoded_outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        return decoded_outputs

    finally:
        if hook is not None:
            hook.remove()
        reset_steering_params()




def run_on_gpu(gpu_id, model_name, steering_vector_path, df, output_path, layer_idx):
    print("ID: ", gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[GPU {gpu_id}] Starting process on device {torch.cuda.current_device()}", flush=True)

    torch.manual_seed(42)

    # Global variables to store steering information
    _STEERING_VECTOR = None
    _STEERING_SCALE = 1.0

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    model.eval()

    print("Available devices:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}", flush=True)

    print("Model is on:")
    for name, param in model.named_parameters():
        if param.device.type == "cuda":
            print(f"{name} is on {param.device}", flush=True)


    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: allocated {allocated:.2f} GB, reserved {reserved:.2f} GB")



    # Loop through data
    start_time = time.time()
    batch_size = 1
    batch = []
    results =  []
    for idx, row in df.iterrows():
        if idx % 20 == 0:
            elapsed = time.time() - start_time
            print(f"{idx} examples processed in {elapsed:.2f}s", flush = True)

        input_text = row["steering_prefill"]
        uuid = row["uuid"]

        batch.append((uuid, input_text))
        if len(batch) >= batch_size:
            uuids, texts = zip(*batch)

            # Generate in batches
            # print("baseline_outputs batched")
            baseline_outputs = generate_text_batch(model, tokenizer, texts, steering_vector=None, layer_index=layer_idx, max_length=8000)
            # print(baseline_outputs)
            # Store all
            for i in range(len(batch)):
                results.append({
                    "uuid": uuids[i],
                    "input_text": texts[i],
                    "encouraged_all_10_8k_scale.6": baseline_outputs[i],
                })
            # Reset batch
            batch = []

            if idx % 5 == 0:
                # Save
                df_temp = pd.DataFrame(results)
                print("writing",  flush = True)
                df_temp.to_parquet(f"{output_path}/gpu{gpu_id}_enc_all_10_8k_{idx}.parquet", index=False)
                print("done", f"{output_path}/gpu{gpu_id}_enc_all_10_8k_{idx}.parquet",  flush = True)



def launch_all_processes():
    model_name = "dipikakhullar/DeepSeek-R1-Distill-Llama-8B-encouraged-all-10-scaled.6"
    df_path = "openr1_math_with_cot_cues.pkl"
    steering_vector_path = "steering_vectors_Wait.pt"
    layer_idx = 19
    num_gpus = torch.cuda.device_count()
    output_path = "encouraged_all_10_8k_scale.6"
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_pickle(df_path)

    df["steering_prefill"] = df.apply(generate_steering_prefill, axis=1)

    df = df.iloc[1000:2000]
    chunk_size = len(df) // num_gpus
    processes = []

    for i in range(num_gpus):
        start = i * chunk_size
        end = None if i == num_gpus - 1 else (i + 1) * chunk_size
        df_chunk = df.iloc[start:end]
        p = mp.Process(target=run_on_gpu, args=(i, model_name, steering_vector_path, df_chunk, output_path, layer_idx))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    launch_all_processes()
