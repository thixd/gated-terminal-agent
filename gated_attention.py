#!/usr/bin/env python3
"""
Phase 2 scaffold: residual gated attention for Hugging Face Qwen2 models.

This keeps the first implementation deliberately narrow:
- targets Qwen2 decoder self-attention
- follows the eager attention path for clarity
- preserves baseline behavior at initialization with alpha=0
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Cache,
    Qwen2Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)


class GatedQwen2Attention(Qwen2Attention):
    """
    Qwen2 attention with a residual head-wise gate.

    Gate formula:
        gate = 1 + alpha * (2 * sigmoid(gate_logits) - 1)

    With gate_proj initialized to zero and alpha=0, the module behaves exactly
    like the baseline attention layer.
    """

    def __init__(
        self,
        config,
        layer_idx: Optional[int] = None,
        alpha_init: float = 0.0,
        learnable_alpha: bool = True,
    ):
        super().__init__(config=config, layer_idx=layer_idx)
        self.gate_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        nn.init.zeros_(self.gate_proj.weight)

        alpha = torch.tensor(float(alpha_init), dtype=torch.float32)
        if learnable_alpha:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # Residual gate initialization avoids "initialization shock" when patching
        # a pretrained model. A raw multiplicative gate would inject random noise
        # into the learned attention output and can destabilize training badly.
        #
        # We instead center sigmoid to [-1, 1] and anchor the gate around 1.0:
        #   residual_gate = 1 + alpha * (2 * sigmoid(gate_logits) - 1)
        #
        # With gate_proj zero-initialized and alpha=0.0, the multiplier is exactly
        # 1.0 at step 0, so the patched model is behaviorally identical to the
        # baseline model. RL can then increase alpha smoothly only if the gate
        # proves useful, reducing catastrophic forgetting risk.
        #
        # hidden_states -> [B, T, H] -> [B, H, T, 1]
        gate_logits = self.gate_proj(hidden_states)
        gate = torch.sigmoid(gate_logits).permute(0, 2, 1).unsqueeze(-1)
        residual_gate = 1.0 + self.alpha.to(attn_output.dtype) * (2.0 * gate - 1.0)
        attn_output = attn_output * residual_gate

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def convert_qwen2_attention_layer(
    layer: Qwen2Attention,
    alpha_init: float = 0.0,
    learnable_alpha: bool = True,
) -> GatedQwen2Attention:
    """
    Replace an existing Qwen2 attention module with a gated variant.
    """
    gated = GatedQwen2Attention(
        config=layer.config,
        layer_idx=layer.layer_idx,
        alpha_init=alpha_init,
        learnable_alpha=learnable_alpha,
    )

    # Copy shared projections and rotary embedding state.
    gated.load_state_dict(layer.state_dict(), strict=False)
    return gated


def patch_qwen2_with_gated_attention(
    model: nn.Module,
    alpha_init: float = 0.0,
    learnable_alpha: bool = True,
    force_eager: bool = True,
) -> nn.Module:
    """
    Swap Qwen2 decoder self-attention modules in-place.

    Args:
        model: A loaded Hugging Face Qwen2 model.
        alpha_init: Residual gate strength at initialization. Use 0.0 for
            baseline-equivalent behavior.
        learnable_alpha: Whether alpha is trainable.
        force_eager: Force attention implementation to eager on the config so
            the custom attention module is used consistently.
    """
    if force_eager and hasattr(model, "config"):
        model.config._attn_implementation = "eager"

    decoder_layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        decoder_layers = model.model.layers
    elif hasattr(model, "layers"):
        decoder_layers = model.layers

    if decoder_layers is None:
        raise ValueError("Could not find Qwen2 decoder layers on the provided model.")

    for layer in decoder_layers:
        layer.self_attn = convert_qwen2_attention_layer(
            layer.self_attn,
            alpha_init=alpha_init,
            learnable_alpha=learnable_alpha,
        )

    return model
