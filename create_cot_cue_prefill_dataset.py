import numpy as np
from datasets import load_dataset

import json
import numpy as np
import pandas as pd
import torch.nn.functional as F
from backtracking_data_sampling import *
# Define CoT cues
COT_CUES = {
    "is_backtracking": [" Wait", " No", " Nope", " Actually", " Hold", " Hang", " Oops", " Sorry"],
}

# Load OpenR1-Math-220k Dataset
dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")

# Convert dataset to Pandas DataFrame
df = dataset.to_pandas()
df["cot_cue_key"], df["cot_cue_value"] = zip(*df["messages"].map(find_cot_cue))

df_filtered = df.dropna(subset=["cot_cue_key", "cot_cue_value"], how="all")

# Assuming df["messages"] is a list of lists where each sublist contains multiple message exchanges
df_filtered["message_pairs"] = df_filtered["messages"].apply(split_messages_into_pairs)

# Explode the DataFrame so each row contains exactly one (user, assistant) pair
df_exploded = df_filtered.explode("message_pairs", ignore_index=True)
len(df_exploded)

# Define function to generate prefill message
def generate_prefill(row):
    messages = row["message_pairs"]
    if len(messages) < 2:
        return None  # Return None if there aren't enough messages
    # print(messages[1]["content"])
    return messages[0]["content"] + STEP_BY_STEP_SUFFIX + messages[1]["content"]


# Display a sample of processed data
df_exploded[["solution", "cot_cue_key", "cot_cue_value"]].head(10)
df_exploded["prefill"] = df_exploded.apply(generate_prefill, axis=1)



df_exploded.to_pickle("openr1_math_with_cot_cues.pkl")