
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import FixedLocator
import re
import nltk
from nltk.tokenize import sent_tokenize


def find_cue_indices(tokenizer, input_text, cue):
     # Tokenize while keeping track of character mapping
    encoding = tokenizer(input_text, return_tensors="pt", return_offsets_mapping=True)
    # print("ENCODING: ", encoding)
    # Convert token IDs to readable tokens
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    
    if cue == None:
        sentences = sent_tokenize(input_text)
        if len(sentences) > 1:  # Ensure a second sentence exists
            second_sentence_start = input_text.find(sentences[1])  # Get character start index
            token_index = encoding.char_to_token(second_sentence_start)
            if token_index is not None:
                print(len(tokens), [(token_index, token_index)])
                return tokens, [(token_index, token_index)]
        return tokens, [(20,20)]

   
    # Find all occurrences of "Wait" in the original text
    # wait_char_indices = [m.start() for m in re.finditer(r'\bLet me reconsider\b', input_text)]
    regex_pattern = r'\b' + re.escape(cue) + r'\b'
    #The span() method of a regex match object returns tuple (start, end). start is inclusive, but end is exclusive.
    cue_char_span = [m.span() for m in re.finditer(regex_pattern, input_text)]
    cue_char_span = [(start, end-1) for start, end in cue_char_span]
    print("cue_char_indices", cue_char_span)
    
    # Get the token indices corresponding to each occurrence of "Wait"
    # wait_token_indices = [encoding.char_to_token(idx) for idx in wait_char_indices]
    cue_token_indices = [(encoding.char_to_token(start_idx), encoding.char_to_token(end_idx)) for start_idx, end_idx in cue_char_span]
    return tokens, cue_token_indices

def clean_tokens(tokens):
    """
    Cleans tokenizer output to make it more readable **without changing the token count**.
    
    1. Removes the 'Ġ' prefix (space marker) **but keeps words separate**.
    2. Replaces 'Ċ' (newline marker) with a visible symbol.
    3. Keeps all tokens intact without merging subwords.
    
    Parameters:
    tokens -- List of tokens from the tokenizer
    
    Returns:
    list -- Cleaned tokens for better readability **without modifying indices**.
    """
    cleaned_tokens = []
    
    for token in tokens:
        # Replace space markers but keep separate tokens
        cleaned = token.replace('Ġ', '')  # Remove space marker, but keep as a separate token
        cleaned = cleaned.replace('Ċ', '↵')  # Replace newlines with a visual marker

        cleaned_tokens.append(cleaned)  # Keep same token positions
    
    return cleaned_tokens


def remove_unwanted_tokens(tokens, attn_matrix, focus_idx_start, focus_idx_end):
    """
    Removes unwanted tokens and updates the focus index accordingly.
    Returns the new attention matrix, reduced token list, and updated focus index.
    """
    # print("attn_matrix", len(attn_matrix))
    print("original token in cue", tokens[focus_idx_start])
    unwanted_tokens = {"", " ", "Ġ", "G", '"', "'", ",", ".", "!", "?", "<pad>", "<s>", "</s>"}

    # Identify valid tokens
    valid_indices = [i for i, token in enumerate(tokens) if token not in unwanted_tokens]
    
    # Create a mapping: old index → new index
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}

    # Update focus index
    new_focus_idx_start = index_mapping.get(focus_idx_start, None)  # None if focus token got removed
    new_focus_idx_end = index_mapping.get(focus_idx_end, None)  # None if focus token got removed

    # Reduce the attention matrix & tokens list
    # reduced_attn_matrix = attn_matrix[np.ix_(valid_indices, valid_indices)]
    reduced_tokens = [tokens[i] for i in valid_indices]
    print("new token in cue", reduced_tokens[new_focus_idx_start])


    # Move attn_matrix to CPU and convert to NumPy if it's a tensor
    if isinstance(attn_matrix, torch.Tensor):
        attn_matrix = attn_matrix.to(torch.float32).cpu().numpy()  # Convert to float32 before NumPy

        
    # **Handle Multi-Head Attention Case**
    if len(attn_matrix.shape) == 3:  
        # print("here") # (num_heads, seq_len, seq_len)
        num_heads = attn_matrix.shape[0]
        reduced_attn_matrix = np.zeros((num_heads, len(valid_indices), len(valid_indices)))

        for head in range(num_heads):
            reduced_attn_matrix[head] = attn_matrix[head][np.ix_(valid_indices, valid_indices)]
    
    # **Handle Single-Head Attention Case**
    elif len(attn_matrix.shape) == 2:  # (seq_len, seq_len)
        reduced_attn_matrix = attn_matrix[np.ix_(valid_indices, valid_indices)]

    else:
        raise ValueError(f"Unexpected attention matrix shape: {attn_matrix.shape}")

    print(f"Reduced attention shape: {reduced_attn_matrix.shape}")
    
    return reduced_attn_matrix, reduced_tokens, new_focus_idx_start, new_focus_idx_end


def plot_attention_grid_v3(attentions, tokens, focus_idx_start, focus_idx_end, window_size=10, 
                        reasoning_cue=None, save_path=None, grid_rows=6, grid_cols=7, title="No Title"):
    """
    Plot attention maps for all heads in a grid layout with enhanced visualization.
    
    Args:
        attentions: Attention tensor of shape [num_heads, seq_len, seq_len]
        tokens: List of tokens
        focus_idx_start: Start index of focus region
        focus_idx_end: End index of focus region
        window_size: Size of window around focus region
        reasoning_cue: Cue being highlighted
        save_path: Path to save the figure
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        title: Title for the entire figure
    """
    # Extract dimensions
    num_heads, seq_len, _ = attentions.shape
    print(f"Input attention shape: {attentions.shape}")
    print(f"Focus indices: {focus_idx_start}, {focus_idx_end}")

    # Determine window bounds with better handling of edge cases
    if reasoning_cue is None:
        start_idx = max(focus_idx_start, 0)
    else:
        start_idx = max(focus_idx_start - window_size, 0)
    
    # Ensure end_idx does not exceed valid range
    end_idx = min(focus_idx_end + window_size + 1, len(tokens))
    
    # If start_idx >= end_idx, force an expansion
    if start_idx >= end_idx - 1:
        print(f"Warning: start_idx ({start_idx}) is too close to end_idx ({end_idx})! Expanding window.")
        if end_idx < len(tokens) - 5:
            end_idx = min(end_idx + 5, len(tokens))
        else:
            start_idx = max(start_idx - 5, 0)
    
    print(f"Window indices: start {start_idx}, end {end_idx}")

    # Extract the window tokens and determine relative focus positions
    window_tokens = tokens[start_idx:end_idx]
    focus_start_rel = max(0, focus_idx_start - start_idx)
    focus_end_rel = min(end_idx - start_idx - 1, focus_idx_end - start_idx)
    
    print(f"Relative focus position: {focus_start_rel}, {focus_end_rel}")
    print(f"Window size: {len(window_tokens)} tokens")
    
    # Create figure (removed space for shared colorbar since we're using individual ones)
    fig = plt.figure(figsize=(grid_cols * 3.5, grid_rows * 3 + 0.5))
    
    # Create GridSpec without extra space for colorbar
    gs = gridspec.GridSpec(grid_rows, grid_cols, figure=fig,
                          wspace=0.3, hspace=0.3)  # Increased spacing for individual colorbars
    
    # Define a custom colormap for better visualization
    cmap = plt.cm.get_cmap('Blues')  # Base colormap
    
    # Track the heatmaps to create a shared colorbar
    heatmaps = []
    
    # Plot each head's attention map
    for head_idx in range(min(num_heads, grid_rows * grid_cols)):
        # Calculate grid position
        row = head_idx // grid_cols
        col = head_idx % grid_cols
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        # Get this head's attention matrix and apply windowing
        head_attn = attentions[head_idx]
        windowed_attn = head_attn[start_idx:end_idx, start_idx:end_idx]
        
        # Calculate min/max for this specific head
        head_min = windowed_attn.min()
        head_max = windowed_attn.max()
        
        # Determine whether to show labels based on token count
        should_label = len(window_tokens) <= 15
        
        # Plot heatmap with individual color scaling
        if should_label:
            hm = sns.heatmap(windowed_attn, cmap=cmap, annot=False, 
                          xticklabels=window_tokens, yticklabels=window_tokens, 
                          ax=ax, vmin=head_min, vmax=head_max, cbar=True,
                          cbar_kws={'shrink': 0.8, 'label': ''})
        else:
            # For many tokens, show a sample of ticks (every nth token)
            n = max(1, len(window_tokens) // 5)  # Show about 5 ticks
            sample_indices = list(range(0, len(window_tokens), n))
            
            # Ensure the focus tokens are among the ticks
            if focus_start_rel not in sample_indices and focus_start_rel < len(window_tokens):
                sample_indices.append(focus_start_rel)
            if focus_end_rel not in sample_indices and focus_end_rel < len(window_tokens):
                sample_indices.append(focus_end_rel)
            
            sample_indices = sorted(sample_indices)
            sample_tokens = [window_tokens[i] for i in sample_indices]
            
            hm = sns.heatmap(windowed_attn, cmap=cmap, annot=False, 
                          xticklabels=sample_tokens if len(sample_tokens) <= 10 else False,
                          yticklabels=sample_tokens if len(sample_tokens) <= 10 else False,
                          ax=ax, vmin=head_min, vmax=head_max, cbar=True,
                          cbar_kws={'shrink': 0.8, 'label': ''})
            
            # Set tick positions to match the sampled tokens
            if len(sample_tokens) <= 10:
                ax.set_xticks([i for i in sample_indices])
                ax.set_yticks([i for i in sample_indices])
        
        heatmaps.append(hm)
        
        # Highlight focus area if reasoning_cue is provided
        if reasoning_cue is not None and focus_start_rel <= focus_end_rel:
            # Print debug information
            # print(f"Original focus positions: {focus_start_rel}, {focus_end_rel}")
            
            # In heatmap coordinates, cell positions are at the integer indices
            # Rectangles need to start at the integer position and have integer width/height
            rect_x = focus_start_rel
            rect_y = focus_start_rel
            width = focus_end_rel - focus_start_rel + 1
            height = focus_end_rel - focus_start_rel + 1
            
            # print(f"Rectangle: x={rect_x}, y={rect_y}, w={width}, h={height}")
            
            # Column highlight (vertical)
            column_rect = plt.Rectangle((rect_x, 0), width, len(window_tokens),
                                      linewidth=1.0, edgecolor='red', facecolor='none', alpha=0.7)
            # Row highlight (horizontal)
            row_rect = plt.Rectangle((0, rect_y), len(window_tokens), height,
                                    linewidth=1.0, edgecolor='red', facecolor='none', alpha=0.7)
            # Focus area intersection
            focus_rect = plt.Rectangle((rect_x, rect_y), width, height,
                                      linewidth=1.5, edgecolor='darkred', facecolor='none',
                                      linestyle='-', alpha=0.9)
            
            ax.add_patch(column_rect)
            ax.add_patch(row_rect)
            ax.add_patch(focus_rect)
        
        # Set title for this head
        ax.set_title(f"Head {head_idx}", fontsize=10, pad=4)
        
        # Adjust tick parameters for clarity
        ax.tick_params(axis='both', which='major', labelsize=7, length=0)
        
        # Rotate x-axis labels for readability
        if should_label or (len(sample_tokens) <= 10 and len(sample_tokens) > 0):
            plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)
    
    # No need for shared colorbar since we're using individual ones
    
    # Add overall title
    plt.suptitle(title, fontsize=16, y=0.98)
    
    # Add axis labels for the entire figure
    fig.text(0.5, 0.01, "Tokens Attending", ha='center', fontsize=14)
    fig.text(0.01, 0.5, "Tokens Attended To", va='center', rotation='vertical', fontsize=14)
    
    # Adjust layout - don't need to reserve space for shared colorbar
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.96])
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    
    # Explicitly display the figure in Jupyter
    plt.figure(fig.number)
    plt.show()
    
    return fig



def plot_attention_map_windowed(attn_matrix, tokens, focus_idx_start, focus_idx_end, window_size=10, reasoning_cue=None, save_path=None):
    """
    Plots the full attention heatmap where both axes contain the same tokens.
    """
    # print(attn_matrix)
    if reasoning_cue == None:
        start_idx = max(focus_idx_start, 0)

    else:
        start_idx = max(focus_idx_start - window_size, 0)
    end_idx = min(focus_idx_end + window_size + 1, len(tokens))

    print(len(attn_matrix), len(attn_matrix[0]))

    attn_matrix = attn_matrix[start_idx:end_idx, start_idx:end_idx]  # Extract windowed attention
    tokens = tokens[start_idx:end_idx]

    plt.figure(figsize=(10, 10))  # Adjust figure size
    ax = sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="Blues", annot=False)


    # # Get relative positions of focus indices in the windowed view
    focus_start_rel = focus_idx_start - start_idx
    focus_end_rel = focus_idx_end - start_idx

     # Create rectangle to highlight focus area - properly aligned with cell boundaries
    rect_x = focus_start_rel
    rect_y = focus_start_rel
    width = focus_end_rel - focus_start_rel + 1
    height = focus_end_rel - focus_start_rel + 1
    
    # Column highlight (vertical)
    column_rect = plt.Rectangle((rect_x, 0), width, len(tokens), 
                              linewidth=.5, edgecolor='red', facecolor='none')
    # Row highlight (horizontal)
    row_rect = plt.Rectangle((0, rect_y), len(tokens), height, 
                            linewidth=.5, edgecolor='red', facecolor='none')
    
    # Focus area intersection
    focus_rect = plt.Rectangle((rect_x, rect_y), width, height, 
                              linewidth=.5, edgecolor='darkred', facecolor='none', linestyle='--')


    if reasoning_cue != None:
        # Add rectangles to the plot
        ax.add_patch(column_rect)
        ax.add_patch(row_rect)
        ax.add_patch(focus_rect)
 
    
    plt.title("Full Attention Map")
    plt.xlabel("Tokens Attended To")
    plt.ylabel("Tokens Attending")
    # plt.xticks(rotation=90)
    # plt.yticks(rotation=0)

    plt.xticks(rotation=90, fontsize=6)  # Reduce font size for x-ticks
    plt.yticks(rotation=0, fontsize=6)  # Reduce font size for y-ticks

    # Save the figure if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directories if they don’t exist
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save with high quality

    plt.show()


def plot_full_attention_map(attn_matrix, tokens, save_path, title):
    """
    Plots the full attention heatmap where both axes contain the same tokens.
    """
    # attn_matrix = attn_matrix.mean(axis=0)  # Average over all heads (Shape: seq_len x seq_len)

    plt.figure(figsize=(100, 100))  # Adjust figure size
    # sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="Blues", annot=False)
    
    ax = sns.heatmap(
        attn_matrix, 
        xticklabels=tokens, 
        yticklabels=tokens, 
        cmap="Blues", 
        annot=False, 
        vmin=0, vmax=max_attn_value,  # Fix min and max
        cbar_kws={
            "shrink": 0.5,  # Make color bar thinner
            "aspect": 10,   # Adjust aspect ratio
            "ticks": np.linspace(0, max_attn_value, num=6)  # Ensure max value is a tick
        }
    )

    plt.title(title)
    plt.xlabel("Tokens Attended To")
    plt.ylabel("Tokens Attending")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches="tight")
        print(f"Saved attention map to {save_path}")

    plt.show()
