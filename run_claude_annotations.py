import os
import json
import base64
import datetime
import time
from io import BytesIO
import re 

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.config import Config
from bs4 import BeautifulSoup
from joblib import Parallel, delayed

# Initialize Bedrock client
bedrock = boto3.client(service_name='bedrock-runtime')
model_id = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
system_prompt = """You are an expert in evaluating mathematical data and understanding mathematical equivalence"""

def create_claude_body(messages = [
                         {"role": "user", "content": "Hello!"}
                        ], 
                       system=system_prompt,
                       token_count=150, 
                       temp=0, 
                       topP=1,
                       topK=350, 
                       stop_sequence=["Human"]):
    """
    Simple function for creating a body for Anthropic Claude models for the Messages API.
    https://docs.anthropic.com/claude/reference/messages_post
    """
    body = {
        "messages": messages,
        "max_tokens": token_count,
        "system":system,
        "temperature": temp,
        "anthropic_version":"",
        "top_k": topK,
        "top_p": topP,
        "stop_sequences": stop_sequence
    }
    return body


def get_claude_response(messages="", 
                        system = "",
                        token_count=4096, 
                        temp=0,
                        topP=1, 
                        topK=0, 
                        stop_sequence=["Human:"], 
                        model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'):
    """
    Simple function for calling Claude via boto3 and the invoke_model API. 
    """
    body = create_claude_body(messages=messages, 
                              system = system,
                              token_count=token_count, 
                              temp=temp,
                              topP=topP, 
                              topK=topK, 
                              stop_sequence=stop_sequence)
    response = bedrock.invoke_model(modelId=model_id, body=json.dumps(body))
    response = json.loads(response['body'].read().decode('utf-8'))
    return response["content"][0]["text"]


# def extract_json_values(response):
#     """
#     This function parses the JSON response from Claude, extracts the correctness and explanation,
#     and returns them in a dictionary to be added as new columns in the DataFrame.
#     """
#     print(response)
#     try:
#         # Parse the response assuming it's a JSON-formatted string
#         response_data = json.loads(response)
        
#         # Extract the 'all_layers_correct' and 'baseline_correct' from the response
#         all_layers_correct = response_data.get("all_layers_correct", False)
#         baseline_correct = response_data.get("baseline_correct", False)
        
#         # Extract the explanation from the response
#         explanation = response_data.get("explanation", "")
        
#         # Return the values as a dictionary
#         return all_layers_correct, baseline_correct, explanation
    
#     except json.JSONDecodeError as e:
#         print(f"Error parsing JSON: {e}")
#         return False, False, "Error parsing JSON"

import re

def extract_json_values(response):
    """
    This function parses the response using string matching (regex) to extract the correctness
    and explanation fields, and returns them to be added as new columns in the DataFrame.
    """
    if not response or response.strip() == "":
        print("Empty or invalid response received.")
        return False, False, "Invalid response"
    
    print(f"Response: {response}")  # Log the response for debugging

    # Define regular expressions for extracting the values
    all_layers_correct_match = re.search(r'"all_layers_correct":\s*(true|false)', response)
    baseline_correct_match = re.search(r'"baseline_correct":\s*(true|false)', response)
    explanation_match = re.search(r'"explanation":\s*"([^"]+)"', response)
    
    # Extract the values using the matches
    all_layers_correct = all_layers_correct_match.group(1) == "true" if all_layers_correct_match else False
    baseline_correct = baseline_correct_match.group(1) == "true" if baseline_correct_match else False
    explanation = explanation_match.group(1) if explanation_match else "No explanation provided"

    return all_layers_correct, baseline_correct, explanation



# Updated process_row to incorporate the new columns
def process_row(row):
    """Function to process each row, call Claude, and validate response."""
    # Extract the answers directly from the DataFrame row
    all_layers_answer = row["all_layers_answer"]
    baseline_answer = row["baseline_answer"]
    gold = row["answer"]
    
    # Prepare the message for Claude
    sample_data = json.dumps({
        "all_layers_answer": all_layers_answer, 
        "baseline_answer": baseline_answer, 
        "gold": gold
    })
    messages = [{"role": "user", "content": sample_data}]
    
    # Call Claude and extract response
    response = get_claude_response(messages=messages, system=mega_prompt, model_id=model_id)      # Extract the correctness and explanation
    all_layers_correct, baseline_correct, explanation = extract_json_values(response)
        
    return response, all_layers_correct, baseline_correct, explanation

    
# In the processing loop, now assign these new columns as well
def process_and_add_columns(df):
    # Process each row, and add new columns
    results = Parallel(n_jobs=n_jobs, backend="threading")(delayed(process_row)(row) for _, row in df.iterrows())
    
    # Extract results and assign them to the DataFrame
    df[["claude_response_raw", "all_layers_correct", "baseline_correct", "explanation"]] = pd.DataFrame(results, index=df.index)




mega_prompt = """

You are an AI classifier that identifies whether two texts are mathematically equivalent. Given the following data:

- `all_layers_answer`: The answer produced by the model after processing all layers.
- `baseline_answer`: The answer produced by the baseline model.
- `gold`: The correct ground truth answer.

For each entry, compare `all_layers_answer` and `baseline_answer` to the `gold` answer to determine if they are mathematically equivalent.

Return a JSON object with the following structure:

```json
{
    "all_layers_correct": true/false,
    "baseline_correct": true/false
}
```

Where:

- `all_layers_correct`: true if `all_layers_answer` is mathematically equivalent to `gold`, false otherwise.
- `baseline_correct`: true if `baseline_answer` is mathematically equivalent to `gold`, false otherwise.

Only output a json and nothing else.

### Example 1:

#### Input:
- `all_layers_answer`: "45°"
- `baseline_answer`: ""
- `gold`: "45°"

Expected output:
```json
{
    "all_layers_correct": true,
    "baseline_correct": false,
    "explanation": The angle symbols are consistent between the `all_layers_answer` and `gold`, so it is considered correct. Since the `baseline_answer` is empty, it is considered incorrect.

}
```

### Example 2:

#### Input:
- `all_layers_answer`: "1"
- `baseline_answer`: ""
- `gold`: "8.5, 8, 6.5, 6, 5.5, 4.5"

Expected output:
```json
{
    "all_layers_correct": false,
    "baseline_correct": false,
    "explanation": The `all_layers_answer` of "1" is clearly different from the list of values in `gold`, and the empty `baseline_answer` is not equivalent either. Therefore, both answers are incorrect.

}
```

"""
parquet_file_path = "bedrock_sample.parquet"
df = pd.read_parquet(parquet_file_path, engine="pyarrow")
n_jobs = 32

# Split DataFrame into two batches for parallel processing
mid_idx = len(df) // 2
df_batch_1 = df.iloc[:mid_idx]
df_batch_2 = df.iloc[mid_idx:]

# Process first batch
print("Processing batch 1...")
results_1 = Parallel(n_jobs=n_jobs, backend="threading")(delayed(process_row)(row) for _, row in df_batch_1.iterrows())

# Sleep to avoid throttling (if applicable)
sleep_time = 15  # Adjust based on AWS Bedrock rate limits
print(f"Sleeping for {sleep_time} seconds...")
time.sleep(sleep_time)

# Process second batch
print("Processing batch 2...")
results_2 = Parallel(n_jobs=n_jobs, backend="threading")(delayed(process_row)(row) for _, row in df_batch_2.iterrows())

# Combine results
results = results_1 + results_2
print("Processing complete!")

# Assign results to new columns
df[["claude_response_raw", "all_layers_correct", "baseline_correct", "explanation"]] = pd.DataFrame(results, index=df.index)

# Saving the processed DataFrame back to a local Parquet file
output_file_path = "processed_bedrock_sample.parquet"
df.to_parquet(output_file_path, engine="pyarrow")
print(f"Saved processed file to {output_file_path}")


