import numpy as np
from datasets import load_dataset
import numpy as np
import torch.nn.functional as F
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from backtracking_data_sampling import *
print("calculating steering vector")
df_exploded = pd.read_pickle("openr1_math_with_cot_cues.pkl")
print(len(df_exploded))
# MODEL_NAME = "Qwen/QwQ-32B-Preview"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda:0",
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
    ),
)
model.eval()


relevant_cues = [" Wait", "Wait"]

STEP_BY_STEP_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
)

activations_data = []

count = 0
# Iterate over each row in the DataFrame
for index, row in df_exploded.iterrows():
    # if row["uuid"] in unique_uuids_evan:
    activations_data.extend(process_samples(row, tokenizer, relevant_cues, model))
    count +=1 
    if count > 750:
        break


print(len(activations_data))

# We take the first 10 from each prefill
def collect_relevant_activations(activations_data, tokenizer, max_per_uuid=10):
    """
    Collects activations from each layer for backtracking and non-backtracking sentences.
    
    Returns:
        activations_dict = {
            "backtracking": [{layer_1: activations, layer_2: activations, ...}],
            "non_backtracking": [{layer_1: activations, layer_2: activations, ...}]
        }
    """
    activations_dict = {"backtracking": defaultdict(list), "non_backtracking": defaultdict(list)}
    
    # Track how many activations are collected per UUID
    backtracking_counts = defaultdict(int)
    non_backtracking_counts = defaultdict(int)
    
    for data in activations_data:
        cue_start = data["cue_start"]
        determining_token_index = data["determining_token_index"]
        cot_cue_value = data["cot_cue_value"]
        cot_cue_token_id = data["cot_cue_token_id"]
        tokens_of_interest = data["tokens_of_interest"]
        relevant_probabilities = data["relevant_probabilities"]
        activations = data["activations"]
        sentence_start = data["sentence_start"]
        uuid = data["uuid"]

        if cot_cue_value:
            cue_probability = relevant_probabilities[cot_cue_token_id]
            # print(cot_cue_value, cot_cue_token_id, cue_probability.item())
            if cue_probability.item() > 0.5 and backtracking_counts[uuid] < max_per_uuid:
                for layer, activation in activations.items():
                    activations_dict["backtracking"][layer].append(activation[0]) 

        else:
            sum_of_relevant_probabilities = sum(relevant_probabilities[i] for i in relevant_probabilities)
            if sum_of_relevant_probabilities < .001 and non_backtracking_counts[uuid] < max_per_uuid:
                for layer, activation in activations.items():
                    activations_dict["non_backtracking"][layer].append(activation[0])  # Activation at token index 1

    return activations_dict

relevant_activations = collect_relevant_activations(activations_data, tokenizer, max_per_uuid=20)




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

# Compute the steering vectors
steering_vectors = compute_steering_vector(relevant_activations)
# Save to disk
torch.save(steering_vectors, "steering_vectors_Wait_prev_index.pt")



from collections import defaultdict, OrderedDict
import torch

#norm_scores will reflect the true strength of the backtracking signal at each layer — not just 1s.


# Use the activations_dict returned by your function
back = relevant_activations["backtracking"]
non_back = relevant_activations["non_backtracking"]

steering_vectors = {}
mean_scores = {}
norm_scores = {}

for layer in back.keys():
    back_acts = torch.tensor(back[layer])        # [N, d_model]
    non_back_acts = torch.tensor(non_back[layer])  # [M, d_model]

    back_mean = back_acts.mean(dim=0)
    non_back_mean = non_back_acts.mean(dim=0)

    raw_vec = back_mean - non_back_mean
    # vec = raw_vec / vec.norm()
    vec = raw_vec / raw_vec.norm()


    steering_vectors[layer] = vec
    mean_scores[layer] = abs(vec.mean().item())
    # norm_scores[layer] = vec.norm().item()
    norm_scores[layer] = raw_vec.norm().item()  # <-- this line changed

# Rank layers by each metric
ranked_by_mean = OrderedDict(sorted(mean_scores.items(), key=lambda x: x[1], reverse=True))
ranked_by_norm = OrderedDict(sorted(norm_scores.items(), key=lambda x: x[1], reverse=True))

# Show top ranked layers
print(f"{'Rank':<5} {'By abs(mean)':<20} {'Score':<10} || {'By norm':<20} {'Score'}")
print("=" * 70)

for i, ((l1, m_score), (l2, n_score)) in enumerate(zip(ranked_by_mean.items(), ranked_by_norm.items())):
    print(f"{i+1:<5} Layer {l1:<13} {m_score:<10.6f} || Layer {l2:<13} {n_score:.6f}")


