import os
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up paths and parameters
OUTPUT_DIR = "qwq_math_results_suffixed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "qwq_math_results.parquet")
CHECKPOINT_INTERVAL = 10
MAX_SAMPLES = 1000

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load existing results if available
if os.path.exists(OUTPUT_FILE):
    existing_results = pd.read_parquet(OUTPUT_FILE)
    processed_uuids = set(existing_results["uuid"])  # Keep track of processed UUIDs
    start_index = len(existing_results)  # Start from the next batch
    print(f"Resuming from index {start_index}, {len(existing_results)} samples already processed.")
else:
    existing_results = pd.DataFrame()
    processed_uuids = set()
    start_index = 0  # Start from the beginning

# Load the dataset
print("Loading the OpenR1-Math-220k dataset...")
dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")
print(f"Dataset loaded with {len(dataset)} examples")

# Load model and tokenizer (with low precision to save memory)
print("Loading QwQ-32B model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B-Preview", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/QwQ-32B-Preview",
    torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
    device_map="auto",           # Automatically use available GPUs
    trust_remote_code=True,
)
print("Model and tokenizer loaded")

# Initialize results list
results = []

# Process the next MAX_SAMPLES samples
for i, sample in enumerate(tqdm(dataset.select(range(start_index, start_index + MAX_SAMPLES)))):
    uuid = sample["uuid"]

    # Skip already processed UUIDs
    if uuid in processed_uuids:
        continue  

    problem = sample["problem"]
    problem += "Please reason step by step, and put your final answer within \\boxed{}."

    # Generate response
    inputs = tokenizer(problem, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=1024,
        temperature=0.6,  # Introduces some randomness
        top_k=40,         # Restricts token choices to top 40
        top_p=0.95,       # Filters out unlikely tokens
        min_p=0.1,        # Optional: Ensures tokens have a minimum probability
        do_sample=True,   # Enables stochastic sampling
    )

    # Decode the output and extract the model's message
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model_response = response[len(problem):].strip()  # Extract just the model's reply

    # Store the result
    results.append({
        "uuid": uuid,
        "problem": problem,
        "model_response": model_response
    })

    # Save results at checkpoint intervals
    if (i + 1) % CHECKPOINT_INTERVAL == 0 or i == MAX_SAMPLES - 1:
        new_df = pd.DataFrame(results)

        # Append to existing results
        combined_df = pd.concat([existing_results, new_df], ignore_index=True)
        combined_df.to_parquet(OUTPUT_FILE)
        print(f"Checkpoint saved after {i + 1} new samples (Total: {len(combined_df)} samples).")

        # Also save a CSV backup
        backup_file = os.path.join(OUTPUT_DIR, f"qwq_math_results_checkpoint_{start_index + i + 1}.csv")
        combined_df.to_csv(backup_file, index=False)
        print(f"Backup saved to {backup_file}")

print(f"Processing complete. Processed {len(results)} new samples.")
print(f"Results saved to {OUTPUT_FILE}")
