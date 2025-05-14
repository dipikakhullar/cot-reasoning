import os
import multiprocessing as mp
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import torch
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


STEP_BY_STEP_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}.\n"

def get_existing_uuids(output_path):
    existing_uuids = set()
    for pf in Path(output_path).glob("*.parquet"):  # Match all .parquet files in the folder
        try:
            df_part = pd.read_parquet(pf)
            existing_uuids.update(df_part["uuid"].tolist())
        except Exception as e:
            print(f"Failed to read {pf}: {e}")
    return existing_uuids


def generate_steering_prefill(row):
    """
    Returns the user content (first in message_pairs) + step-by-step suffix.
    """
    messages = row["message_pairs"]
    if len(messages) < 2:
        return None  # Return None if there aren't enough messages
    
    # messages[0] is the user message
    return messages[0]["content"] + STEP_BY_STEP_SUFFIX


def generate_text_batch(model, tokenizer, input_texts, steering_vector=None, scale=1.0, max_length=2000, layer_index=5):
    """
    Batched generation with optional steering vector hook at one layer.
    Assumes all prompts in the batch share the same steering configuration.
    """
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    hook = None
    with torch.no_grad():
        output_tokens = model.generate(**inputs, 
                                        max_length=max_length, 
                                        use_cache=True,
                                        eos_token_id=tokenizer.eos_token_id,  # Stop at EOT
                                        pad_token_id=tokenizer.eos_token_id,  # Use EOT as padding
                                        early_stopping=True  # End generation when EOT is produced
                                        )

    decoded_outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
    return decoded_outputs


def run_on_gpu(gpu_id, model_name, steering_vector_path, df, output_path, layer_idx):
    print("ID: ", gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[GPU {gpu_id}] Starting process on device {torch.cuda.current_device()}", flush=True)

    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    )
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
        if idx % 2 == 0:
            elapsed = time.time() - start_time
            print(f"{idx} examples processed in {elapsed:.2f}s", flush = True)

        input_text = row["steering_prefill"]
        uuid = row["uuid"]

        batch.append((uuid, input_text))
        if len(batch) >= batch_size:
            uuids, texts = zip(*batch)

            # Generate in batches
            print("baseline_outputs batched")
            baseline_outputs = generate_text_batch(model, tokenizer, texts, steering_vector=None, layer_index=layer_idx, max_length=16000)

            # Store all
            for i in range(len(batch)):
                results.append({
                    "uuid": uuids[i],
                    "input_text": texts[i],
                    "baseline_llama_16k": baseline_outputs[i],
                })

            # Reset batch
            batch = []

            # Save
        if idx % 5 == 0:
            df_temp = pd.DataFrame(results)
            print("writing",  flush = True)
            df_temp.to_parquet(f"{output_path}/gpu{gpu_id}_baseline_llama_16k_{idx}.parquet", index=False)
            print("done", f"{output_path}/gpu{gpu_id}_baseline_llama_16k_{idx}.parquet",  flush = True)



def launch_all_processes():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    df_path = "openr1_math_with_cot_cues.pkl"
    steering_vector_path = "steering_vectors_Wait.pt"
    layer_idx = 19
    num_gpus = torch.cuda.device_count()
    output_path = "baseline_llama_16k"
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_pickle(df_path)

    df["steering_prefill"] = df.apply(generate_steering_prefill, axis=1)

    existing_uuids = get_existing_uuids(output_path)
    df = df[~df["uuid"].isin(existing_uuids)].reset_index(drop=True)

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
