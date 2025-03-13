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
    messages = row["messages"]
    if len(messages) < 2:
        return None  # Return None if there aren't enough messages
    
    return messages[0]["content"] + STEP_BY_STEP_SUFFIX + extract_tagged_content(messages[1]["content"], tag="think")[0] 


# def extract_think_content(text):
#     """
#     Extracts all content inside <think> tags.
#     If multiple <think> tags exist, concatenates them.
#     Returns None if no <think> tag is found.
#     """
#     matches = re.findall(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
#     return (
#         " ".join(matches).strip() if matches else None
#     )  # Join multiple matches if needed


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
            think_content = extract_tagged_content(message["content"])
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


def find_cue_indices(tokenizer, input_text, cue, tag="think"):
    # Extract the text inside the <think> tag and its position in input_text
    tagged_text, tag_start, tag_end = extract_tagged_content(input_text, tag=tag)

    # print(tag_start, tag_end )
    # Tokenize while keeping track of character mapping
    encoding = tokenizer(input_text, return_tensors="pt", return_offsets_mapping=True)
    # print("ENCODING: ", encoding)
    # Convert token IDs to readable tokens
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    if cue == None:
        sentences = nltk_sentence_tokenize_message(input_text)
        if len(sentences) > 1:  # Ensure a second sentence exists
            second_sentence_start = input_text.find(
                sentences[1]
            )  # Get character start index
            token_index = encoding.char_to_token(second_sentence_start)
            if token_index is not None:
                # print(len(tokens), [(token_index, token_index)])
                return tokens, [(token_index, token_index)]
        return tokens, [(20, 20)]

    # Find all occurrences of "Wait" in the original text
    # regex_pattern = r'\b' + re.escape(cue) + r'\b'
    regex_pattern = r"(\n{0,2})\s*" + re.escape(cue.strip()) + r"\s*"

    # The span() method of a regex match object returns tuple (start, end). start is inclusive, but end is exclusive.
    cue_char_span = [m.span() for m in re.finditer(regex_pattern, input_text)]
    cue_char_span = [(start, end - 1) for start, end in cue_char_span]
    print("cue_char_indices", cue_char_span)

    # Keep only matches that fall within <think> boundaries
    valid_cue_spans = [
        (start, end) for start, end in cue_char_span if tag_start <= start < tag_end
    ]

    # Get the token indices corresponding to each occurrence of "Wait"
    # wait_token_indices = [encoding.char_to_token(idx) for idx in wait_char_indices]
    cue_token_indices = [
        (encoding.char_to_token(start_idx), encoding.char_to_token(end_idx))
        for start_idx, end_idx in valid_cue_spans
    ]

    """
    The function currently maps cue_char_span (character-level indices) to token indices, but sometimes the start token index corresponds to an unwanted token (e.g., '.ĊĊ' instead of 'Wait'). This happens because tokenization might:
    
    Split tokens improperly, especially when dealing with newlines (\n, Ċ tokens).
    Attach special characters (like Ċ, ▁, or punctuation) to the start of tokens.
    Thus, we need to check if the first token in cue_token_indices is actually part of the cue. If it's not, we should use the second index instead"""

    # Post-process to ensure proper matching
    adjusted_cue_indices = []
    for start_idx, end_idx in cue_token_indices:
        if start_idx is None or end_idx is None:
            continue  # Skip invalid mappings

        first_token = tokens[start_idx] if start_idx < len(tokens) else ""
        expected_cue = cue.lstrip()  # Remove any leading spaces/newlines

        # If the first token is not part of the cue, use the second token
        if not first_token.lower().startswith(expected_cue.lower()):
            adjusted_cue_indices.append((end_idx, end_idx))  # Use second token index
        else:
            adjusted_cue_indices.append((start_idx, end_idx))  # Keep original match

    return tokens, adjusted_cue_indices

    # return tokens, cue_token_indices


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
    # with torch.no_grad():
    #     outputs = model(**inputs, output_hidden_states=True)
    #     hidden_states = outputs.hidden_states

    with torch.no_grad(), residuals_for_layers(model, range(32)) as hidden_states:
        # Get both hidden states and logits
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        # hidden_states = outputs.hidden_states
        logits = outputs.logits[0]  # First batch item

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

    # Extract all activations and make sure they're 2D (tokens × embedding)
    all_activations = {}
    for layer_idx in range(32):  # range(1, len(hidden_states)):
        # Get the layer's activations and remove batch dimension
        layer_activations = hidden_states[layer_idx][0].cpu().numpy()
        all_activations[layer_idx] = layer_activations

    print(f"Total layers in hidden_states: {len(hidden_states)}")
    print(f"Extracted layers: {list(all_activations.keys())}")
    print(f"Activation shape for first layer: {all_activations[1].shape}")

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


def determine_backtracking_status(
    sentence,
    backtracking_cue,
    sentence_start_token_index,
    probs,
    cue_index_in_tokenized_inputs,
    cue_token_id,
):
    """
    Determine if a sentence is backtracking based on pattern matching and token probabilities.

    """
    # check if cue is present in start of sentence
    # regex_pattern = r"^\n{0,2}\b" + re.escape(backtracking_cue) + r"\b"
    regex_pattern = r"^\n{0,2}\b" + re.escape(backtracking_cue.strip()) + r"\b"
    # regex_pattern = backtracking_cue
    match = re.search(regex_pattern, sentence)
    sentence_start_contains_cue = bool(match)

    # Look at probability of cue token at the position *before* sentence start
    preceding_idx = cue_index_in_tokenized_inputs - 1
    # print(sentence)
    # print(f"Shape of probs: {probs.shape}, Preceding Index: {preceding_idx}, Cue Token ID: {cue_token_id}")

    # Show top predicted tokens at the position before sentence start
    top_tokens = torch.topk(probs[preceding_idx, :], 5)
    # if sentence_start_contains_cue:
    #     print(f"Top tokens at position before sentence start ({preceding_idx}):")
    # for i, (prob, tok_id) in enumerate(zip(top_tokens.values, top_tokens.indices)):
    #     token = tokenizer.decode([tok_id.item()])
    #     if sentence_start_contains_cue:
    #         print(f"  {i+1}. {token} (ID: {tok_id.item()}, prob: {prob.item():.4f})")

    # Find probability of cue in the top predictions before sentence start
    return top_tokens

    # Look for any token that might be part of the cue in the top tokens
    # for prob, tok_id in zip(top_tokens.values, top_tokens.indices):
    #     if tok_id == cue_token_id:
    #         cue_prob = prob.item()


# **Process backtracking and non-backtracking samples**
def process_samples(samples, tokenizer, model):
    # print("process_samples", samples)
    activations_data = []
    # Extract message and cue
    message = samples["prefill"]
    # message = (
    #     samples["messages"][0]["content"]
    #     + STEP_BY_STEP_SUFFIX
    #     + samples["messages"][1]["content"]
    # )
    backtracking_cue = samples["cot_cue_value"]
    print(backtracking_cue)
    # Tokenize full message (with offset mapping)
    # returns token ids, attention mask, attention mask doesn't matter here, it would be all 1s because it's trying to pay equal attention to everything
    tokenized_inputs = tokenizer(
        message,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
    ).to("cuda")
    # filtered sentences only within specific tag, indices relative to entire input
    sentences, sentence_token_indices = get_sentence_start_end_indices_new(
        message, tokenizer, tag="think"
    )
    # print(sentence_token_indices)
    tokens, cue_token_indices = find_cue_indices(
        tokenizer, message, backtracking_cue, tag="think"
    )
    # print(cue_token_indices)
    # print("cue_token_indices", cue_token_indices)
    # print("sentence_token_indices", sentence_token_indices)
    # inputs are tokenized inputs
    model_probs, all_activations = get_all_model_outputs(model, tokenized_inputs)

    sentence_info_map = []
    for i in range(len(sentences)):
        # print("EXAMINING SENTENCE ", i)
        sentence_cue_indices = []
        sentence = sentences[i]
        sent_token_start, sent_token_end = sentence_token_indices[i]
        cue_token_id = 14524
        determining_token_start_index = sent_token_start
        for cue_start, cue_end in cue_token_indices:
            # print(cue_start, cue_end)
            if cue_start >= sent_token_start and cue_end <= sent_token_end:
                sentence_cue_indices.append([cue_start, cue_end])
                determining_token_start_index = cue_start
                break

        # sentence, backtracking_cue, sentence_start_token_index, probs, cue_index_in_tokenized_inputs
        top_tokens = determine_backtracking_status(
            sentence,
            backtracking_cue,
            sent_token_start,
            model_probs,
            determining_token_start_index,
            cue_token_id,
        )
        sentence_activations = get_activations_at(
            list(
                range(determining_token_start_index - 1, determining_token_start_index + 2)
            ),
            all_activations,
        )
        activations_data.append(
            {
                "uuid": samples["uuid"],
                "full_message": message,
                "sentence": sentence,
                # "is_backtracking": is_backtracking,
                # "cue_probability": cue_prob,
                "top_tokens": top_tokens,
                "char_offset": message.find(sentence),
                "determining_token_index": determining_token_start_index,
                "cue_start": sentence_cue_indices,
                "sentence_start": sent_token_start,
                # "indices_to_use": indices_to_use,
                "sentence_start_index": sent_token_start,
                "activations": sentence_activations,
                "cot_cue_value": samples["cot_cue_value"],
            }
        )
    return activations_data