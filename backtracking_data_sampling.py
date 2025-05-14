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
import gc

# Define CoT cues
COT_CUES = {
    "is_backtracking": [
        " Wait",
        " No",
        " Nope",
        " Actually",
        " Hold",
        " Hang",
        " Oops",
        " Sorry",
    ],
}
# examples = backtracking_samples
STEP_BY_STEP_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
)

# Define function to generate prefill message
def generate_prefill(row):
    messages = row["message_pairs"]
    if len(messages) < 2:
        return None  # Return None if there aren't enough messages
    
    return messages[0]["content"] + STEP_BY_STEP_SUFFIX + extract_tagged_content(messages[1]["content"], tag="think")[0] 



def extract_tagged_content(text, tag="think"):
    """
    Extracts content inside the specified <tag> while preserving the tags in the output.
    Returns (tagged_content_with_tags, start_index, end_index), or (None, None, None) if no match.
    """
    match = re.search(fr"(<{tag}>.*?</{tag}>)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1), match.start(), match.end()  # Return full match including tags
    return None, None, None



def find_cot_cue(messages):
    """
    Finds the first CoT reasoning cue within all <think> content.
    Iterates over all assistant messages and checks each one's <think> content separately.
    """
    for message in messages:
        if message["role"] == "assistant":
            think_content = extract_tagged_content(message["content"])[0]
            if think_content:  # Ensure we have extracted content
                for key, values in COT_CUES.items():
                    for value in values:
                        # Use regex to match value with optional preceding whitespace/newlines
                        pattern = r"[\s\n]*" + re.escape(
                            value
                        )  # Allow leading spaces/newlines
                        if re.search(pattern, think_content, re.IGNORECASE):
                            return key, value  # Return the first matched key-value pair
    return None, None  # Return None if no match is found


def split_messages_into_pairs(messages):
    """Splits a list of messages into non-overlapping pairs of (user, assistant)."""
    if len(messages) > 2:
        print(len(messages))
    return [
        messages[i : i + 2]
        for i in range(0, len(messages), 2)
        if len(messages[i : i + 2]) == 2
    ]


def extract_tagged_content(text, tag="think"):
    """
    Extracts content inside specified <tag> tags and returns its start and end indices.
    Returns (content, start_index, end_index), or (None, None, None) if no match.
    """
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return (
            match.group(1).strip(),
            match.start(1),
            match.end(1),
        )  # Content and its position in original text
    return None, None, None


def find_cue_indices(tokenizer, input_text, cues, token_ids_of_interest, tag="think"):
    # Extract the text inside the <think> tag and its position in input_text        
    tagged_text, tag_start, tag_end = extract_tagged_content(input_text, tag=tag)

    # Tokenize while keeping track of character mapping
    encoding = tokenizer(input_text, return_tensors="pt", return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    if cues is None or not cues:
        sentences = nltk_sentence_tokenize_message(input_text)
        if len(sentences) > 1:  # Ensure a second sentence exists
            second_sentence_start = input_text.find(sentences[1])  # Get character start index
            token_index = encoding.char_to_token(second_sentence_start)
            if token_index is not None:
                return tokens, [(token_index, token_index, None)]
        return tokens, [(20, 20, None)]  # Added None for no specific cue
    
    cues_char_span = []
    
    # Iterate over all cues in the list
    for cue in cues:
        # Store the tuple: (token_start, token_end, actual_cue)
        # regex_pattern = r"(\n{0,2})\s*" + re.escape(cue.strip()) + r"\s*"

        # In find_cue_indices:
        regex_pattern = r"\b" + re.escape(cue.strip()) + r"\b"  # Changed from using \n{0,2}
        # regex_pattern = r"\b" + cue + r"\b"  # Changed from using \n{0,2}

        cue_char_span = [m.span() for m in re.finditer(regex_pattern, input_text)]
        cue_char_span = [(start, end - 1, cue) for start, end in cue_char_span]
        cues_char_span.extend(cue_char_span)

    # Keep only matches that fall within <think> boundaries
    valid_cue_spans = [
        (start, end, cue) for start, end, cue in cues_char_span if tag_start <= start < tag_end
    ]

    # Convert character indices to token indices
    cue_token_indices = [
        (encoding.char_to_token(start_idx), encoding.char_to_token(end_idx), cue)
        for start_idx, end_idx, cue in valid_cue_spans
    ]


    # After converting character indices to token indices:
    # Only keeps token spans where at least one of the tokens matches an expected ID

    filtered_cue_token_indices = []
    for start_idx, end_idx, cue in cue_token_indices:
        if start_idx is None or end_idx is None:
            continue  # Skip invalid mappings
        
        # Get actual token IDs for this span
        if start_idx < len(encoding['input_ids'][0]) and end_idx < len(encoding['input_ids'][0]):
            span_token_ids = encoding['input_ids'][0][start_idx:end_idx+1].tolist()
            
            # Check if any of these token IDs match our expected ones for this cue
            # expected_ids = cue_to_token_ids[cue]
            # print(span_token_ids, "looking in", token_ids_of_interest)
            if any(token_id in span_token_ids for token_id in token_ids_of_interest):
                filtered_cue_token_indices.append((start_idx, end_idx, cue))
                # print(f"Matched token IDs for cue '{cue}': {span_token_ids}")
            else:
                print(f"Skipping span with non-matching token IDs: {span_token_ids}")


    # Post-process to ensure proper matching for each cue
    adjusted_cue_indices = []
    for start_idx, end_idx, cue in  filtered_cue_token_indices:
        if start_idx is None or end_idx is None:
            continue  # Skip invalid mappings
        
        # Check if this is a valid token range
        if start_idx >= len(tokens) or end_idx >= len(tokens) or start_idx > end_idx:
            print(f"Skipping invalid token range: {start_idx}-{end_idx}")
            continue
        
        # Get the actual tokens for this span
        span_tokens = tokens[start_idx:end_idx+1]
        token_text = tokenizer.convert_tokens_to_string(span_tokens)
        # print(f"Token span {start_idx}-{end_idx}: {span_tokens} → '{token_text}'")
        
        # Get the core part of the cue (without any whitespace)
        core_cue = cue.strip()
        core_cue_id= tokenizer(
                cue,
                padding=True,
                truncation=True,
                return_offsets_mapping=True,
                add_special_tokens= False
            )['input_ids'][-1]
        # Create a mapping from token index to actual cleaned content
        token_contents = {}
        for i, token in enumerate(span_tokens):
            # Clean the token by removing tokenizer-specific prefixes
            clean_token = token.replace('Ġ', '').replace('▁', '')
            # Remove any whitespace characters
            clean_token = clean_token.strip()
            if clean_token:  # Only consider non-empty tokens
                token_contents[start_idx + i] = clean_token
        
        # Find the first token that contains actual content (not just whitespace)
        for token_idx, content in token_contents.items():
            if content and any(char.isalnum() for char in content):
                # Check if this content is part of the core cue
                if content.lower() in core_cue.lower() or core_cue.lower().startswith(content.lower()):
                    # print(f"Found core content at token {token_idx}: '{content}'")
                    adjusted_cue_indices.append((token_idx, end_idx, cue, core_cue_id))
                    break
        else:
            # If we couldn't find a token with matching content, use the original span
            print(f"Using original span: {start_idx}-{end_idx}")
            adjusted_cue_indices.append((start_idx, end_idx, cue, core_cue_id))

    # print(adjusted_cue_indices)
    return tokens, adjusted_cue_indices


def nltk_sentence_tokenize_message(message):
    """Tokenize a full message into sentences."""
    return nltk.sent_tokenize(message)


def get_sentence_start_end_indices_new(message, tokenizer, tag=None):
    """
    Tokenizes the full message but filters only sentences inside a specified tag.
    If no tag is provided, returns all sentences with their corresponding token indices.
    """

    # If a tag is provided, extract content inside the tag
    if tag:
        tagged_text, tag_start, tag_end = extract_tagged_content(message, tag=tag)
    else:
        tagged_text, tag_start, tag_end = (
            message,
            0,
            len(message),
        )  # Default to full message if no tag is given

    if not tagged_text:  # If no content found in tag, return empty
        return [], []

    # Tokenize the **full message** while keeping character offsets
    encoding = tokenizer(
        message,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
    )
    offsets = (
        encoding["offset_mapping"].squeeze(0).tolist()
    )  # List of (char_start, char_end)

    # Tokenize full message into sentences (to get original character-level offsets)
    sentences = nltk_sentence_tokenize_message(message)

    # Get start-end positions of sentences in full message
    char_ranges = [
        (message.find(sent), message.find(sent) + len(sent)) for sent in sentences
    ]

    # Convert character indices to token indices using char_to_token
    sentence_token_ranges = []
    filtered_sentences = []

    for i, (char_start, char_end) in enumerate(char_ranges):
        # Ensure sentence is inside the <think> tag OR process all sentences if no tag is given
        if tag is None or (tag_start <= char_start < tag_end):
            token_start = encoding.char_to_token(char_start)
            token_end = encoding.char_to_token(
                char_end - 1
            )  # Use last character's token index

            if token_start is not None and token_end is not None:
                sentence_token_ranges.append((token_start, token_end))
                filtered_sentences.append(sentences[i])

    return filtered_sentences, sentence_token_ranges


from contextlib import ExitStack, contextmanager


@contextmanager
def residuals_for_layers(model, layers):
    cached_outputs = {}

    def make_layer_hook(layer: int):
        def hook(_model: torch.nn.Module, _args, _kwargs, output: torch.Tensor):
            cached_outputs[layer] = output[0].to("cpu")
            return output

        return hook

    with ExitStack() as hook_stack:
        for layer in layers:
            hook_stack.enter_context(
                model.model.layers[layer].register_forward_hook(
                    make_layer_hook(layer), with_kwargs=True
                )
            )
        yield cached_outputs
    return


# Function to get all activations from the model
def get_all_model_outputs(model, inputs):
    with torch.no_grad(), residuals_for_layers(model, range(32)) as hidden_states:
        # Get both hidden states and logits
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        # hidden_states = outputs.hidden_states
        logits = outputs.logits[0]  # First batch item

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu()

    # Extract all activations and make sure they're 2D (tokens × embedding)
    all_activations = {}
    for layer_idx in range(32):  # range(1, len(hidden_states)):
        # Get the layer's activations and remove batch dimension
        layer_activations = hidden_states[layer_idx][0].cpu().numpy()
        all_activations[layer_idx] = layer_activations

    # print(f"Total layers in hidden_states: {len(hidden_states)}")
    # print(f"Extracted layers: {list(all_activations.keys())}")
    # print(f"Activation shape for first layer: {all_activations[1].shape}")

    return probs, all_activations


# Function to get activations at specific indices
def get_activations_at(indices, all_activations):
    """Extracts activations for specific indices from all activations."""
    if not indices or not all_activations:
        return {}

    # Extract activations at specific indices
    specific_activations = {
        layer_idx: all_activations[layer_idx][indices] for layer_idx in all_activations
    }

    return specific_activations


def get_token_probabilities(
    probs,
    cue_index_in_tokenized_inputs,
    tokens_of_interest
):
    """
    Determine if a sentence is backtracking based on pattern matching and token probabilities.

    """
    # Look at probability of cue token at the position *before* sentence start
    preceding_idx = cue_index_in_tokenized_inputs - 1

    # {2360: probability, 14144: probability}
    relevant_probabilities = {i:probs[preceding_idx, i]  for i in list(tokens_of_interest)}

    # Find probability of cue in the top predictions before sentence start
    return relevant_probabilities


# **Process backtracking and non-backtracking samples**
def process_samples(samples, tokenizer, relevant_cues, model):
    tokens_of_interest = set()
    for cue in relevant_cues:
        tokenized_inputs= tokenizer(
                cue,
                padding=True,
                truncation=True,
                return_offsets_mapping=True,
                add_special_tokens= False
            )
        tokens_of_interest.add(tokenized_inputs['input_ids'][-1])
    # print(tokens_of_interest)

    activations_data = []
    message = samples["prefill"]
    input_device = model.hf_device_map.get("model.embed_tokens", "cuda:0")
    tokenized_inputs = tokenizer(
        message,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
    ).to(input_device)
    # filtered sentences only within specific tag, indices relative to entire input
    sentences, sentence_token_indices = get_sentence_start_end_indices_new(
        message, tokenizer, tag="think"
    )
    # print(sentence_token_indices)
    tokens, cue_token_indices = find_cue_indices(
        tokenizer, message, list(relevant_cues), tokens_of_interest, tag="think"
    )
    print(f"Found {len(cue_token_indices)} potential cue indices")




    torch.cuda.empty_cache()
    gc.collect()


    model_probs, all_activations = get_all_model_outputs(model, tokenized_inputs)

    sentence_info_map = []
    non_cue_sentence_count = 0
    cue_sentence_count = 0
    for i in range(len(sentences)):
        sentence_cue_indices = []
        sentence = sentences[i]
        sent_token_start, sent_token_end = sentence_token_indices[i]
        # print(f"Sentence {i}: Token range {sent_token_start}-{sent_token_end}")
        determining_token_start_index = sent_token_start
        cot_cue_value = None
        cot_cue_token_id = None

        for cue_start, cue_end, found_cue, cue_token_id, in cue_token_indices:
            # print(f"Cue span: {cue_start}-{cue_end}, tokens: {tokens[cue_start:cue_end+1]}")
            # print(f"  Checking if cue {cue_start}-{cue_end} is in sentence {sent_token_start}-{sent_token_end}")

            if cue_start >= sent_token_start and cue_end <= sent_token_end:
                cot_cue_token_id = cue_token_id
                cot_cue_value = found_cue  # Use the actual cue that was found
                # print("cot_cue_value found: ", cot_cue_value)
                sentence_cue_indices.append([cue_start, cue_end])
                determining_token_start_index = cue_start
                break
        if cot_cue_value == None:
            non_cue_sentence_count += 1
        if non_cue_sentence_count > 10 and cot_cue_value == None:
            continue

        relevant_probabilities = get_token_probabilities(
            model_probs,
            determining_token_start_index,
            tokens_of_interest
        )
        sentence_activations = get_activations_at(
            list(range(determining_token_start_index -1, determining_token_start_index + 2)),
            all_activations)
        activations_data.append(
            {
                "uuid": samples["uuid"],
                "prefill": message,
                "sentence": sentence,
                "cot_cue_value": cot_cue_value,
                "cot_cue_token_id": cot_cue_token_id,
                "relevant_probabilities": relevant_probabilities,
                "char_offset": message.find(sentence),
                "determining_token_index": determining_token_start_index,
                "cue_start": sentence_cue_indices,
                "sentence_start": sent_token_start,
                "sentence_start_index": sent_token_start,
                "activations": sentence_activations,
                "tokens_of_interest": tokens_of_interest
            }
        )
    return activations_data

