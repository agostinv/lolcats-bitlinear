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


# ----------------------
# Sliding window helpers
# ----------------------
def get_masks(
    window_size: int,
    q_len: int,
    k_len: int,
    device: torch.device,
    topk: int = None,
    a_pred: torch.Tensor = None,
) -> tuple[torch.Tensor]:
    """
    Return masks for softmax and linear attention terms
    -> 1 is include, 0 is ignore
    """
    kwargs = {"device": device, "dtype": int}
    causal_mask = torch.ones((q_len, k_len), **kwargs).tril(k_len - q_len)
    linear_mask = torch.ones((q_len, k_len), **kwargs).tril(k_len - q_len - window_size)
    window_mask = causal_mask - linear_mask

    # If no topk or no prediction scores provided, return default masks
    if topk is None or a_pred is None:
        # Return softmax mask (window), linear attention mask
        # -> shapes broadcast over (b, h, q_len, k_len)
        return window_mask[None, None, ...], linear_mask[None, None, ...]

    # We'll select per-row keys from the linear region using the provided
    # a_pred scores (expected shape like (b, h, q_len, k_len) or broadcastable).
    # The number of selected keys increases per block of rows. The block size
    # is provided in `topk` and the starting kept keys is half the block size.
    topk_block = int(topk)
    start_k = max(1, topk_block // 2)

    assert len(a_pred.size()) == 4, "a_pred should be (b,h,q,k)"
    scores = a_pred  # shape (b,h,q,k)
    bsize, nheads = scores.shape[0], scores.shape[1]

    # Prepare masks expanded to (b,h,q,k)
    new_window = window_mask[None, None, ...].expand(bsize, nheads, -1, -1).clone()
    new_linear = linear_mask[None, None, ...].expand(bsize, nheads, -1, -1).clone()

    # Zero out everything outside the linear region so they don't affect ranking
    lm = linear_mask[None, None, ...].to(dtype=scores.dtype)
    masked_scores = scores * lm

    assert q_len > window_size, "Query length must be larger than window size"
    assert k_len > window_size, "Key length must be larger than window size"

    num_rows_linear = q_len - window_size
    num_blocks = num_rows_linear // topk_block

    for block_idx in range(num_blocks):
        start_row = window_size + block_idx * topk_block
        end_row = start_row + topk_block

        # block_lm: (block, k_len) indicating available linear positions per row
        block_lm = linear_mask[start_row:end_row, :].to(device=scores.device)

        # scores for this block: (b,h,block,k_len)
        scores_block = masked_scores[:, :, start_row:end_row, :]

        # mask_unavail: (1,1,block,k_len) with -inf for unavailable, 0 for available
        neginf = float("-inf")
        mask_unavail = torch.where(
            block_lm == 1,
            torch.tensor(0.0, device=scores.device, dtype=scores.dtype),
            torch.tensor(neginf, device=scores.device, dtype=scores.dtype),
        )
        mask_unavail = mask_unavail.unsqueeze(0).unsqueeze(0)

        # Compute keep_k for this block (start_k guaranteed >=1)
        keep_k = int(start_k * (block_idx + 1))
        assert keep_k > 0, "Must keep at least one key per row beyond sliding window"

        # Add mask_unavail so unavailable positions are ignored by topk
        scores_block = scores_block + mask_unavail

        # topk over last dim -> (b,h,block,keep_k)
        _, top_idx = torch.topk(
            scores_block, keep_k, dim=-1, largest=True, sorted=False
        )

        # Build binary mask and scatter into (b,h,block,k_len)
        b_s, h_s, block_sz = top_idx.shape[0], top_idx.shape[1], top_idx.shape[2]
        row_mask = torch.zeros(
            (b_s, h_s, block_sz, k_len), dtype=new_window.dtype, device=scores.device
        )
        row_mask.scatter_(-1, top_idx, 1)

        # Apply block mask to new_window/new_linear
        new_window[:, :, start_row:end_row, :] = (
            new_window[:, :, start_row:end_row, :] | row_mask
        )
        new_linear[:, :, start_row:end_row, :] = new_linear[
            :, :, start_row:end_row, :
        ] & (1 - row_mask)

    # Return masks shaped for broadcasting over (b, h, q_len, k_len)
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
    topk_block_size: int = None,
    a_pred: torch.Tensor = None,
    kv_state: torch.Tensor = None,
    k_state: torch.Tensor = None,
    eps: float = 1e-12,
    mask_value: float = -1e8,
):
    """
    Hybrid attention combining sliding window and linear attentions
    """

    mask_window, mask_linear = get_masks(
        window_size,
        q.shape[-2],
        k.shape[-2],
        q.device,
        topk_block_size,
        a_pred,
    )

    # 1. Sliding window (softmax attention)
    a_sm = torch.einsum("bhmd,bhnd->bhmn", q.float(), k.float()) * (k.shape[-1] ** -0.5)
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
class LolcatsLinearSlidingWindowTopkBlockwise(LolcatsLinearAttention):
    """
    Lolcats attention combining sliding window and linear attention
    """

    def __init__(
        self,
        window_size: int = 32,
        topk_block_size: int = 16,
        decode_window_size: int = None,
        affine_attention_factors: bool = False,
        init_window_factor: float = 0,
        train_window_factor: bool = True,
        state_grad_enabled: bool = False,
        **kwargs,
    ):
        self.window_size = window_size
        self.topk_block_size = topk_block_size
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

        if self.train_attention:
            # 1. Compute "ground-truth" attention output and weights
            with torch.no_grad():
                _y_true, a_true = softmax_attention(q, k, v)[:2]
                y_true = (
                    _y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                )
                y_true = self.o_proj(y_true)

            # 2. Compute "predicted" attention outputs
            # compute attn weights under sliding window
            window_factors = F.sigmoid(self.window_factors)
            linear_factors = 1 - window_factors if self.affine_attention_factors else 1
            y_pred, a_pred = self.quadratic_attention(
                q,
                k,
                f_q,
                f_k,
                v,
                window_factors,
                linear_factors,
                window_size=self.window_size,
                topk_block_size=self.topk_block_size,
            )
            attn_weights = ((a_pred, a_true), (y_pred, _y_true))
        else:
            attn_weights = None
            # attention_mask = None  # For now this is always True
            if past_key_value is None:  # Regular training
                window_factors = F.sigmoid(self.window_factors)
                linear_factors = (
                    1 - window_factors if self.affine_attention_factors else 1
                )
                y_true, a_pred = self.quadratic_attention(
                    q,
                    k,
                    f_q,
                    f_k,
                    v,
                    window_factors,
                    linear_factors,
                    window_size=self.window_size,
                )
                attn_weights = a_pred
            else:
                past_key_value.window_size = self.decode_window_size
                if (
                    f_q.shape[2] == 1 and kv_seq_len > 1 and not self.training
                ):  # Generating
                    assert use_cache is True
                    _kv = past_key_value.update_for_decoding(
                        k, v, self.layer_idx, self.feature_map_k, dtype=q.dtype
                    )
                    k_cache, v_cache, f_kv_state, f_k_state = _kv

                    # Sliding window + linear attention decode
                    window_factors = F.sigmoid(self.window_factors)
                    linear_factors = (
                        1 - window_factors if self.affine_attention_factors else 1
                    )

                    # Softmax attention terms
                    a_sm = torch.einsum(
                        "bhmd,bhnd->bhmn", q.float(), k_cache.float()
                    ) * (k.shape[-1] ** -0.5)
                    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
                    a_sm = window_factors * torch.exp(a_sm - a_sm_max)
                    sum_sm = a_sm.sum(dim=-1, keepdim=True)

                    # Combine with linear attention terms
                    y_true = torch.einsum(
                        "bhmn,bhnd->bhmd", a_sm, v_cache.float()
                    ) + linear_factors * torch.einsum(
                        "bhlf,bhfd->bhld", f_q.float(), f_kv_state.float()
                    )
                    sum_ln = (
                        linear_factors
                        * torch.einsum(
                            "bhld,bhnd->bhl", f_q.float(), f_k_state.float()
                        )[..., None]
                    )
                    y_true = (y_true / (sum_sm + sum_ln)).to(q.dtype)

                else:  # Stateful training
                    try:
                        kv_state = past_key_value.kv_states[self.layer_idx]
                        k_state = past_key_value.k_states[self.layer_idx]
                    except IndexError:
                        kv_state, k_state = None, None
                    window_factors = F.sigmoid(self.window_factors)
                    linear_factors = (
                        1 - window_factors if self.affine_attention_factors else 1
                    )
                    y_true, _ = self.quadratic_attention(
                        q,
                        k,
                        f_q,
                        f_k,
                        v,
                        window_factors,
                        linear_factors,
                        window_size=self.window_size,
                        kv_state=kv_state,
                        k_state=k_state,
                    )
                    # Save and update KV cache and states
                    # past_key_value.update(k, v.detach(), self.layer_idx,
                    #                       fmap_key_states=f_k.detach(),
                    #                       accumulate_in_fp32=True)
                    past_key_value.update(
                        k,
                        v,
                        self.layer_idx,
                        fmap_key_states=f_k,
                        accumulate_in_fp32=True,
                    )
            # Concatenate heads and apply output projection
            y_true = y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
            y_true = self.o_proj(y_true)
        return y_true, attn_weights, past_key_value


# NOTE: will be non-functional
class LinearAttentionSlidingWindowTopkBlockwiseCache(LinearAttentionState):
    """
    Class for `past_key_values`
    -> Alternative to KV cache; here we only maintain a "KV state" and "K state"
    -> Modified from transformers.cache_utils.DynamicCache (v4.36)
    """

    def __init__(self, window_size: int = 64) -> None:
        super().__init__()
        self._seen_tokens = 0  # should be `self.seen_tokens` in Transformers v4.36
        self._seen_tokens_by_layer: List[int] = []
        self.kv_states: List[torch.Tensor] = []
        self.k_states: List[torch.Tensor] = []

        # Account for sliding windows
        self.decode_kv_states: List[torch.Tensor] = []
        self.decode_k_states: List[torch.Tensor] = []
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        self.window_size = window_size

    # NOTE: Do not care about this for SW Linear attn usually because we don't engage in stateful training in this example
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: Optional[int] = None,
        cache_kwargs: Optional[any] = None,
        accumulate_in_fp32: bool = False,
        fmap_key_states: torch.Tensor = None,  # should not be None
        grad_enabled: bool = False,
        **kwargs: any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Stateful training not implemented for LinearAttentionSlidingWindowCache"
        )

    def update_for_decoding(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        feature_map_k: Callable,
        dtype: torch.dtype,
    ):
        """
        Update the decoding KV and K states, and KV cache, during decodeing
        """
        with torch.no_grad():
            k_cache = self.k_cache[layer_idx]
            v_cache = self.v_cache[layer_idx]

            if k_cache.shape[-2] < self.window_size:  # build window-size cache
                self.k_cache[layer_idx] = torch.cat([k_cache, keys], dim=-2)
                self.v_cache[layer_idx] = torch.cat([v_cache, values], dim=-2)
            else:
                k_state = feature_map_k(k_cache[:, :, :1, :])
                v_state = v_cache[:, :, :1, :]
                kv_state = torch.einsum(
                    "bhlf,bhld->bhfd", k_state.float(), v_state.float()
                ).to(
                    dtype
                )  # b, h, f, d
                self.decode_kv_states[layer_idx] += kv_state
                self.decode_k_states[layer_idx] += k_state

                self.k_cache[layer_idx] = torch.cat(
                    [k_cache[:, :, 1:, :], keys], dim=-2
                )
                self.v_cache[layer_idx] = torch.cat(
                    [v_cache[:, :, 1:, :], values], dim=-2
                )

            if layer_idx == 0:
                self._seen_tokens += keys.shape[-2]
            self._seen_tokens_by_layer[layer_idx] += keys.shape[-2]
            return (
                self.k_cache[layer_idx],
                self.v_cache[layer_idx],
                self.decode_kv_states[layer_idx],
                self.decode_k_states[layer_idx],
            )
