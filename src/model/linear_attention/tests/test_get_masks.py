import os
import sys
import importlib
import torch
import math

# Weird import issue, relative import is having problem identifying parent package but __init__.py is clearly present
### from ..linear_window_attention_topk_linear import get_masks


# ----------------------
# Sliding window helpers
# ----------------------
def get_masks_terraced(
    window_size: int, q_len: int, k_len: int, device: torch.device
) -> tuple[torch.Tensor]:
    """
    Return masks for softmax and linear attention terms
    -> 1 is include, 0 is ignore
    """
    kwargs = {"device": device, "dtype": int}
    l = window_size
    m = math.ceil(max(q_len, k_len) / window_size)
    # Creates an n x n mask where n = window_size^2
    mask = torch.block_diag(
        *[
            torch.ones(
                (l, l),
            )
        ]
        * m
    )
    mask += torch.roll(mask, -l, -1)  # this adds the terracing
    if mask.shape[0] > q_len:
        mask = mask[-q_len:]
    if mask.shape[1] > k_len:
        mask = mask[:, -k_len:]
    # Return softmax mask (window), linear attention mask
    mask = mask[None, None, ...]  # b, h, q_len, k_len
    return torch.tril(mask).to(**kwargs), torch.tril(1 - mask).to(**kwargs)


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

    # Ensure device placement (use the provided a_pred)
    ap = a_pred.to(device)

    B, H, Q, K = ap.shape
    k_select = topk

    # Build base masks (q,k) -> (1,1,q,k) and repeat them to (B,H,Q,K)
    # Use `repeat` to explicitly tile the (1,1,q,k) masks across batch and head dims
    base_window = window_mask[None, None, ...].repeat(B, H, 1, 1).bool().to(device)
    base_linear = linear_mask[None, None, ...].repeat(B, H, 1, 1).bool().to(device)
    base_causal = causal_mask[None, None, ...].repeat(B, H, 1, 1).bool().to(device)

    # Replace values within eps of zero with -inf so they are ignored by topk selection
    eps = 1e-6
    ap_masked = ap.clone()
    ap_masked[ap_masked.abs() <= eps] = float("-inf")

    # Exclude positions outside the causal (lower-triangular) region
    # since those are invalid for attention.
    ap_masked = ap_masked.masked_fill(~base_causal, float("-inf"))

    # Exclude positions already included in the base sliding window from top-k selection
    # (they will be included anyway), so mask them out by setting -inf.
    ap_masked = ap_masked.masked_fill(base_window, float("-inf"))

    # Get top-k indices (may include -inf entries if fewer than k non-zeros)
    values, indices = torch.topk(ap_masked, k=k_select, dim=-1)

    # Build boolean mask for selected topk non-zero positions using vectorized scatter
    topk_mask = torch.zeros(B, H, Q, K, dtype=torch.bool, device=device)
    # `values` and `indices` have shape (B, H, Q, k_select)
    valid = values > float("-inf")
    # Scatter the valid flags into the key dimension at the top-k indices
    # `scatter_` will write `True` at positions where valid is True
    topk_mask.scatter_(-1, indices, valid)

    # Ensure top-k mask does not contain non-causal positions (safety):
    # expand the causal mask to (B,H,Q,K) and AND it with topk_mask
    topk_mask = topk_mask & base_causal

    # Combine base masks with topk mask. Cast back to int dtype to match callers
    new_window = (base_window | topk_mask).to(torch.int)
    new_linear = (base_linear & (~topk_mask)).to(torch.int)

    return new_window, new_linear


def test_get_masks_topk_moves_indices():

    # small synthetic setup
    B, H = 1, 1
    q_len = 8
    k_len = 8
    window_size = 1
    topk = 2

    a_pred = torch.Tensor(
        [
            [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.1, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.2, 0.2, 0.4, 0.0, 0.0, 0.0],
                    [0.1, 0.2, 0.1, 0.1, 0.2, 0.3, 0.0, 0.0],
                    [0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.0],
                    [0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1],
                ]
            ]
        ]
    )

    device = torch.device("cpu")

    # Also test terraced masks for comparison
    print("====================")
    print("Terraced masks:")
    terraced_window_mask, terraced_linear_mask = get_masks_terraced(
        window_size, q_len, k_len, device
    )
    print(terraced_window_mask[0, 0])
    print(terraced_linear_mask[0, 0])
    print("====================")

    # Get masks with top-k
    window_mask, linear_mask = get_masks(
        window_size, q_len, k_len, device, topk=topk, a_pred=a_pred
    )

    # masks are returned as (b, h, q, k) tensors of int (0/1)
    assert window_mask.shape == (B, H, q_len, k_len)
    assert linear_mask.shape == (B, H, q_len, k_len)

    print(window_mask[0, 0])

    # For rows >= window_size, the positions we set in a_pred should be 1 in window_mask and 0 in linear_mask
    assert window_mask[0, 0, 0, 0] == 1

    assert window_mask[0, 0, 1, 0] == 1
    assert window_mask[0, 0, 1, 1] == 1

    assert window_mask[0, 0, 2, 0] == 1
    assert window_mask[0, 0, 2, 1] == 1
    assert window_mask[0, 0, 2, 2] == 1

    assert window_mask[0, 0, 3, 1] == 1
    assert window_mask[0, 0, 3, 2] == 1
    assert window_mask[0, 0, 3, 3] == 1

    assert window_mask[0, 0, 4, 2] == 1
    assert window_mask[0, 0, 4, 3] == 1
    assert window_mask[0, 0, 4, 4] == 1

    assert window_mask[0, 0, 5, 1] == 1
    assert window_mask[0, 0, 5, 4] == 1
    assert window_mask[0, 0, 5, 5] == 1

    assert window_mask[0, 0, 6, 0] == 1
    assert window_mask[0, 0, 6, 4] == 1
    assert window_mask[0, 0, 6, 6] == 1

    assert window_mask[0, 0, 7, 1] == 1
    assert window_mask[0, 0, 7, 6] == 1
    assert window_mask[0, 0, 7, 7] == 1

    assert torch.equal(
        linear_mask,
        (
            (torch.ones_like(window_mask) & ~window_mask)
            & ~torch.triu(torch.ones_like(window_mask), diagonal=1)
        ),
    )
    assert torch.equal(
        linear_mask + window_mask, torch.tril(torch.ones_like(window_mask))
    )


if __name__ == "__main__":
    test_get_masks_topk_moves_indices()
    print("OK")
