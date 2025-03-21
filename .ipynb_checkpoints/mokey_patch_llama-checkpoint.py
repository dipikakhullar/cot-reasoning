import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig, Cache, FlashAttentionKwargs, Unpack, apply_rotary_pos_emb, repeat_kv
import transformers.models.llama.modeling_llama as modeling_llama

# From: transformers/models/llama/modeling_llama.py

def my_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    construct_dormant: bool,               # @psando: if True, dormant masks are saved to module.dormant_masks
    zero_dormant: bool,                    # @psando: if True, we zero out the output of dormant heads
    use_double_sink_def: bool,             # @psando: if False, we use the first token (single sink) definition
    threshold_avg_weight: Optional[float], # @psando
    threshold_value_norm: Optional[float], # @psando
    zero_dormant_randomly: bool,           # @psando
    zero_dormant_randomly_prob: float,     # @psando
    layers_to_exclude: list = [],          # @psando
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    if construct_dormant: # @psando: create a dormant_mask boolean tensor of shape (B, N_head)
        assert not torch.isnan(attn_weights).any(), "attn_weights contains nan values and should not"
        # print(f"{attn_weights.shape=}")   # (B, N_head, S, S)
        # print(f"{value_states.shape=}")   # (B, N_head, S, D)
        # print(f"{attention_mask.shape=}") # (B, 1, 1, S)
        # print(f"{module.pad_idxs=}")      # (B,)

        if module.layer_idx in layers_to_exclude:
            # where each entry is False
            dormant_mask = torch.zeros_like(attn_weights[:,:,0,0], dtype=torch.bool)
        elif zero_dormant_randomly: 
            # where an entry is True with probability zero_dormant_randomly_prob
            dormant_mask = torch.rand_like(attn_weights[:,:,0,0]) < zero_dormant_randomly_prob
        else: 
            # where each entry is True if the head is dormant
            attn_output = torch.matmul(attn_weights, value_states) # (B, N_head, S, D)
            norm_per_token = attn_output.norm(dim=-1) # (B, N_head, S)
            
            # padding token norms do not matter, so we set them to nan to be ignored by nanmean
            if hasattr(module, 'pad_idxs'): # saved in evaluate_attention_heads_drop.py bc lm-eval-harness does not use padding attention mask in loglikelihood evals
                for b in range(norm_per_token.shape[0]):
                    pad_idx = module.pad_idxs[b]
                    norm_per_token[b, :, pad_idx:] = torch.nan
            else: # get padding indices from the attention mask (padding tokens are where mask is -inf)
                for b in range(norm_per_token.shape[0]):
                    pad_idx = torch.sum(~(attention_mask[b,0,-1,:] == torch.finfo(attention_mask.dtype).min), dim=-1).item()
                    norm_per_token[b, :, pad_idx:] = torch.nan
                
            avg_norm_per_head = norm_per_token.nanmean(dim=-1) # (B, N_head)
            assert not torch.isnan(avg_norm_per_head).any(), "avg_norm_per_head contains nan values and should not"

            # compute average across all heads in layer
            layer_context = avg_norm_per_head.mean(dim=1) # (B,)

            # head output defn: avg_norm_per_head < threshold_avg_weight * layer_context
            relative_avg_norm_per_head = (avg_norm_per_head / layer_context[:, None]) # (B, N_head)
            dormant_mask = relative_avg_norm_per_head < threshold_avg_weight # (B, N_head)

        if dormant_mask.any():
            module.dormant_masks.append(dormant_mask.cpu())
            # Heads that are dropped will not change the original hidden state at those head dimensions
            # so we only execute attn_weights @ value_states for heads that are not dropped
            # i.e. attn_output will be the same as hidden_states for dropped heads
            #                  will be the result of attn_weights @ value_states for non-dropped heads
            # 3. Compute the attention output 
            # start with zeros, replace with hidden_states for dropped heads
            if zero_dormant:
                attn_output = torch.zeros_like(value_states, device=value_states.device)
                # index into attn_weights and value_states to only include heads that are not dropped
                attn_output_not_dropped = torch.matmul(attn_weights[~dormant_mask], value_states[~dormant_mask])
                # update attn_output with the result of attn_weights @ value_states for non-dropped heads
                attn_output[~dormant_mask] = attn_output_not_dropped
            # else: attn_output has already been computed above so we use it without modification
        else:
            module.dormant_masks.append(torch.zeros((attn_weights.shape[0], attn_weights.shape[1]), dtype=bool))
            attn_output = torch.matmul(attn_weights, value_states)
    else:
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class MyLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int,
                 save_value_states=False, 
                 save_pre_output_proj_hidden_states=False, 
                 save_post_output_proj_hidden_states=False,
                 construct_dormant=False,
                 zero_dormant=False,
                 use_double_sink_def=False,
                 threshold_avg_weight=None,
                 threshold_value_norm=None,
                 zero_dormant_randomly=False,
                 zero_dormant_randomly_prob=None,
                 layers_to_exclude=[]):
        super().__init__(config=config, layer_idx=layer_idx)

        # @psando
        self.save_value_states = save_value_states
        self.save_pre_output_proj_hidden_states = save_pre_output_proj_hidden_states
        self.save_post_output_proj_hidden_states = save_post_output_proj_hidden_states
        self.value_states = None
        self.pre_output_proj_hidden_states = None
        self.post_output_proj_hidden_states = None

        self.construct_dormant = construct_dormant
        self.zero_dormant = zero_dormant
        self.use_double_sink_def = use_double_sink_def
        self.threshold_avg_weight = threshold_avg_weight
        self.threshold_value_norm = threshold_value_norm
        self.zero_dormant_randomly = zero_dormant_randomly
        self.zero_dormant_randomly_prob = zero_dormant_randomly_prob
        self.layers_to_exclude = layers_to_exclude

        self.dormant_masks = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if self.save_value_states:           # @psando
            self.value_states = value_states # @psando

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = my_eager_attention_forward # @psando

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            construct_dormant=self.construct_dormant,
            zero_dormant=self.zero_dormant,
            use_double_sink_def=self.use_double_sink_def,
            threshold_avg_weight=self.threshold_avg_weight,
            threshold_value_norm=self.threshold_value_norm,
            zero_dormant_randomly=self.zero_dormant_randomly,
            zero_dormant_randomly_prob=self.zero_dormant_randomly_prob,
            layers_to_exclude=self.layers_to_exclude,
            **kwargs,
        )
        if self.save_pre_output_proj_hidden_states:                         # @psando
            self.pre_output_proj_hidden_states = attn_output.transpose(1,2) # @psando
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if self.save_post_output_proj_hidden_states:          # @psando
            self.post_output_proj_hidden_states = attn_output # @psando
        return attn_output, attn_weights
    
def patch_llama(save_value_states=False, 
                save_pre_output_proj_hidden_states=False, 
                save_post_output_proj_hidden_states=False,
                construct_dormant=False,
                zero_dormant=False,
                use_double_sink_def=False,
                threshold_avg_weight=None,
                threshold_value_norm=None,
                zero_dormant_randomly=False,
                zero_dormant_randomly_prob=None,
                layers_to_exclude=[]):
    class PatchedLlamaAttention(MyLlamaAttention):
        def __init__(self, config: LlamaConfig, layer_idx: int, *args, **kwargs):
            assert config._attn_implementation == 'eager', "Only eager attention is supported because access to intermediate tensors like attention weights is needed.\n" \
                   "Please set `AutoModelForCausalLM.from_pretrained(..., attn_implementation='eager')` when initializing the model."
            super().__init__(
                config,
                layer_idx=layer_idx,
                save_value_states=save_value_states,
                save_pre_output_proj_hidden_states=save_pre_output_proj_hidden_states,
                save_post_output_proj_hidden_states=save_post_output_proj_hidden_states,
                construct_dormant=construct_dormant,
                zero_dormant=zero_dormant,
                use_double_sink_def=use_double_sink_def,
                threshold_avg_weight=threshold_avg_weight,
                threshold_value_norm=threshold_value_norm,
                zero_dormant_randomly=zero_dormant_randomly,
                zero_dormant_randomly_prob=zero_dormant_randomly_prob,
                layers_to_exclude=layers_to_exclude,
                *args,
                **kwargs
            )

    modeling_llama.LlamaAttention = PatchedLlamaAttention
