"""
Subquadratic attention combining a small sliding window with a topk component + linear attention

For each layer:
- We first compute (softmax) attention over sliding windows and preserve a few topk keys
- We then compute standard linear attention to "fill in" the earlier parts
- We combine to model the entire sequence
"""

from typing import List, Tuple, Optional, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache

from .linear_attention import (
    LolcatsLinearAttention,
    LinearAttentionState,
    softmax_attention,
)


def keyformer_mask(
    attn_weights, recent_tokens, key_tokens, tau_init, accumulation_method="keyformer"
):
    # attn_weights (BS, head, query, keys)
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]
    padding_length = 0

    if accumulation_method is "keyformer":
        scoring_fn = nn.functional.gumbel_softmax
        accumulation_fn = torch.sum
    elif accumulation_method is "h2o":
        scoring_fn = nn.functional.softmax
        accumulation_fn = torch.sum

    offset = torch.finfo(attn_weights.dtype).min
    tmp_attn = scoring_fn(attn_weights, dim=-1, tau=tau_init, hard=False).to(
        dtype_attn_weights
    )

    accumulated_score = accumulation_fn(
        tmp_attn[:, :, padding_length : recent_tokens + key_tokens + padding_length, :],
        dim=-2,
    )  # (samples, head, keys)
    accumulated_score[:, :, recent_tokens + key_tokens + padding_length :] = 0
    accumulated_score[:, :, :padding_length] = 0

    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom[
        :,
        :,
        padding_length : recent_tokens + key_tokens + padding_length,
        padding_length : recent_tokens + key_tokens + padding_length,
    ] = True

    for token_index in range(recent_tokens + key_tokens + padding_length, seq_length):
        tmp_attn_index = scoring_fn(
            attn_weights[:, :, token_index, :], dim=-1, tau=tau_init, hard=False
        ).to(dtype_attn_weights)

        _, tmp_topk_index = accumulated_score[:, :, : token_index - recent_tokens].topk(
            k=key_tokens - 1, dim=-1
        )

        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        mask_bottom_index = zeros_index.scatter(
            -1, tmp_topk_index, True
        )  # (samples, head, keys)
        mask_bottom_index[:, :, token_index] = True

        mask_bottom[:, :, token_index, :] = mask_bottom_index
        accumulated_score += tmp_attn_index
        accumulated_score = accumulated_score * mask_bottom_index

    return mask_bottom


# ----------------------
# Sliding window helpers
# ----------------------
def get_masks(
    window_size: int,
    q_len: int,
    k_len: int,
    device: torch.device,
    a_pred: torch.Tensor,
    key_tokens: int = None,
    tau_init: float = 1.0,
    accumulation_method: str = "keyformer",
) -> tuple[torch.Tensor]:
    """
    Return masks for softmax and linear attention terms using keyformer_mask
    -> 1 is include, 0 is ignore
    """
    kwargs = {"device": device, "dtype": int}
    causal_mask = torch.ones((q_len, k_len), **kwargs).tril(k_len - q_len)
    linear_mask = torch.ones((q_len, k_len), **kwargs).tril(k_len - q_len - window_size)
    window_mask = causal_mask - linear_mask

    # Use keyformer_mask on attention weights
    keyformer_window_mask = keyformer_mask(
        a_pred, window_size, key_tokens, tau_init, accumulation_method
    )
    # Convert keyformer mask to match expected output format
    # keyformer_window_mask is (B, H, Q, K) boolean
    new_window = keyformer_window_mask.to(torch.int)
    # linear mask is the complement of window mask (within causal region)
    B, H, Q, K = a_pred.shape
    base_causal = causal_mask[None, None, ...].repeat(B, H, 1, 1).bool().to(device)
    new_linear = (base_causal & (~keyformer_window_mask)).to(torch.int)

    return new_window, new_linear


def hybrid_attention_quadratic(
    q: torch.Tensor,
    k: torch.Tensor,
    f_q: torch.Tensor,
    f_k: torch.Tensor,
    v: torch.Tensor,
    window_factor: torch.Tensor,
    linear_factor: torch.Tensor,
    window_size: int,
    kv_state: torch.Tensor = None,
    k_state: torch.Tensor = None,
    eps: float = 1e-12,
    mask_value: float = -1e8,
    recent_tokens: int = None,
    key_tokens: int = None,
    tau_init: float = 1.0,
    accumulation_method: str = "keyformer",
):
    """
    Hybrid attention combining sliding window and linear attentions with keyformer masking

    Recent Extra Args:
        recent_tokens: Number of recent tokens for keyformer window
        key_tokens: Number of key tokens to keep with keyformer
        tau_init: Temperature parameter for keyformer gumbel softmax
        accumulation_method: Method for accumulating attention scores
    """

    # 1. Sliding window (softmax attention), grab initial vals
    a_sm = torch.einsum("bhmd,bhnd->bhmn", q.float(), k.float()) * (k.shape[-1] ** -0.5)

    mask_window, mask_linear = get_masks(
        window_size,
        q.shape[-2],
        k.shape[-2],
        q.device,
        a_pred=a_sm,
        key_tokens=key_tokens,
        tau_init=tau_init,
        accumulation_method=accumulation_method,
    )

    a_sm = a_sm.masked_fill(~mask_window.bool(), mask_value)
    # torch.softmax(a_sm, dim=-1), but we account for the max when combining
    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
    a_sm = window_factor * torch.exp(a_sm - a_sm_max)
    sum_sm = a_sm.sum(dim=-1, keepdim=True)

    # 2. Under window (linear attention)
    a_ln = torch.einsum("bhmd,bhnd->bhmn", f_q.float(), f_k.float())
    a_ln = linear_factor * a_ln.masked_fill(~mask_linear.bool(), 0)
    sum_ln = a_ln.sum(dim=-1, keepdim=True)

    # 3. Combine
    a = ((a_sm + a_ln) / (sum_sm + sum_ln)).to(q.dtype)  # Save attention weights
    # Allow outputs to also depend on prior kv_state and k_state
    y = torch.einsum("bhmn,bhnd->bhmd", a_sm + a_ln, v.float())
    if kv_state is not None:  # Combine with prior kv_state and k_state
        y += linear_factor * torch.einsum(
            "bhld,bhdf->bhlf", f_q.float(), kv_state.float()
        )
        sum_ln += (
            linear_factor
            * torch.einsum("bhld,bhnd->bhl", f_q.float(), k_state.float())[..., None]
        )
    y = (y / (sum_sm + sum_ln)).to(q.dtype)
    return y, a  # attention weights only for the last chunk


# ---------------------
# Attention layer class
# ---------------------
class LolcatsLinearSlidingWindowRandomMask(LolcatsLinearAttention):
    """
    Lolcats attention combining sliding window and linear attention
    """

    def __init__(
        self,
        window_size: int = 32,
        key_token_cache_budget: int = 32,
        accumulation_method: str = "keyformer",
        decode_window_size: int = None,
        affine_attention_factors: bool = False,
        init_window_factor: float = 0,
        train_window_factor: bool = True,
        state_grad_enabled: bool = False,
        keyformer_tau_init: float = 1.0,
        **kwargs,
    ):
        self.window_size = window_size
        self.key_token_cache_budget = key_token_cache_budget
        self.accumulation_method = accumulation_method
        self.decode_window_size = (
            decode_window_size if decode_window_size is not None else window_size
        )
        self.window_kwargs = {"dimension": 2, "size": window_size, "step": 1}
        super().__init__(**kwargs)
        self.attention_type = kwargs["attention_type"]  #  'hedgehog_llama_window_tk'
        # Determine how we compute attentions
        self.quadratic_attention = hybrid_attention_quadratic
        self.attention_type = kwargs[
            "attention_type"
        ]  # 'hedgehog_long_llama_window_topk'
        # Learnable factor for combining attentions
        self.affine_attention_factors = affine_attention_factors
        device, dtype = self.q_proj.weight.device, self.q_proj.weight.dtype
        if train_window_factor:
            self.window_factors = nn.Parameter(
                init_window_factor
                * torch.ones(1, self.num_heads, 1, 1, device=device, dtype=dtype)
            )
        else:
            self.register_buffer(
                "window_factors",
                init_window_factor
                * torch.ones(1, self.num_heads, 1, 1, device=device, dtype=dtype),
            )
        # Whether we use original flash attention 2 inference (use during attention transfer)
        self.base_inference = False
        self.state_grad_enabled = state_grad_enabled
        self.window_factor = self.window_factors  # legacy naming support
        # Keyformer mask temperature parameter
        self.keyformer_tau_init = keyformer_tau_init

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with the option to compute attention weights multiple ways
        if self.train_attention is True
        -> Consistent with HuggingFace Transformers for easy use with their pretrained models
        """
        b, l, _ = hidden_states.size()
        q, k, v, kv_seq_len = self.process_qkv(
            hidden_states, attention_mask, position_ids, past_key_value
        )
        f_q, f_k = self.feature_map_q(q), self.feature_map_k(
            k
        )  # Have to do after repeat for grouped-query attn if we use same fmap

        attn_weights = None
        past_key_value.window_size = self.decode_window_size
        y_true = None
        if f_q.shape[2] == 1 and kv_seq_len > 1 and not self.training:  # Generating

            raise NotImplementedError(
                "This hybrid sliding window + linear attention does not support autoregressive generation with cache eviction policies at the moment"
            )

        elif not self.training:  # Eval mode, full sequence
            y_true, attn_weights = self.quadratic_attention(
                q,
                k,
                f_q,
                f_k,
                v,
                window_factor=self.window_factor,
                linear_factor=1.0,
                window_size=self.window_size,
                kv_state=None,
                k_state=None,
                eps=1e-6,
                recent_tokens=self.window_size,
                key_tokens=self.key_token_cache_budget,
                tau_init=self.keyformer_tau_init,
                accumulation_method=self.accumulation_method,
            )

        # Concatenate heads and apply output projection
        y_true = y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
        y_true = self.o_proj(y_true)

        return y_true, attn_weights, past_key_value
