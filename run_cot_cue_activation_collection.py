import json
import re
import uuid

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from backtrack_data_sampling import *

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
# model = AutoModelForCausalLM.from_pretrained(
#     "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
#     device_map="auto",
#     quantization_config=BitsAndBytesConfig(
#         load_in_8bit=True,
#     ),
# )
# model.eval()

MODEL_NAME = "Qwen/QwQ-32B-Preview"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda:2",
    trust_remote_code=True
)
model.eval()


df_exploded = pd.read_pickle("openr1_math_with_cot_cues.pkl")

# relevant_cues = [" Wait"]
relevant_cues = ['Wait', 'No', 'Nope', 'Actually', 'Hold', 'Hang', 'Oops', 'Sorry']

count = 0
activations_data = []

count = 0
# Iterate over each row in the DataFrame
for index, row in df_exploded.iterrows():
    activations_data.extend(process_samples(row, tokenizer, relevant_cues, model))
    count +=1 
    if count > 10:
        break