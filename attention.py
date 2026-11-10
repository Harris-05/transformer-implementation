"""Inference-only dense and tiled scaled dot-product attention.

Implements stable row normalization and online softmax explicitly. Inputs are
finite Q/K/V tensors of shape [batch, heads, tokens, head_dim]. FP16/BF16 inputs
accumulate in FP32; FP64 precision is preserved. See ATTENTION.md for equations.
"""

import math

import torch


def _prepare(q, k, v, causal, query_offset):
    if any(x.ndim != 4 or min(x.shape) < 1 for x in (q, k, v)):
        raise ValueError('Expected nonempty [batch, heads, tokens, head_dim] tensors.')
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError('Attention requires a supported floating-point dtype.')
    if len({x.dtype for x in (q, k, v)}) != 1 or len({x.device for x in (q, k, v)}) != 1:
        raise ValueError('Q, K and V must share dtype and device.')
    if q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1]:
        raise ValueError('Incompatible batch, K/V sequence/head, or Q/K head dimensions.')
    if q.shape[1] % k.shape[1]:
        raise ValueError('Query heads must be an integer multiple of KV heads.')
    if type(causal) is not bool or type(query_offset) is not int:
        raise ValueError('causal must be bool and query_offset must be an integer.')
    accumulation = torch.float64 if q.dtype == torch.float64 else torch.float32
    return q.to(accumulation), k.to(accumulation), v.to(accumulation)


def _mask(scores, q_start, k_start, query_offset):
    rows = torch.arange(q_start, q_start + scores.shape[-2], device=scores.device) + query_offset
    columns = torch.arange(k_start, k_start + scores.shape[-1], device=scores.device)
    return scores.masked_fill(columns[None, :] > rows[:, None], float('-inf'))


@torch.inference_mode()
def reference_attention(q, k, v, *, causal=True, query_offset=0):
    """Dense reference with scaling, causal masking and stable explicit softmax.

    Query row i has absolute position i+query_offset. A fully masked row returns
    zeros. Noncausal mode allows all keys. No dropout or attention bias is applied.
    """
    dtype = q.dtype
    q, k, v = _prepare(q, k, v, causal, query_offset)
    groups = q.shape[1] // k.shape[1]
    if groups != 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    if causal:
        scores = _mask(scores, 0, 0, query_offset)
    maximum = scores.amax(dim=-1, keepdim=True)
    safe_maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    weights = torch.exp(scores - safe_maximum)
    denominator = weights.sum(dim=-1, keepdim=True)
    weights = weights / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return (weights @ v).to(dtype)


@torch.inference_mode()
def tiled_attention(q, k, v, *, causal=True, query_offset=0, block_q=64, block_k=64):
    """Blocked attention using running maxima, sums and weighted-value numerators.

    The largest score tensor is a query/key tile, never the complete QK matrix
    unless the chosen tile itself covers the whole input. K/V head expansion for
    GQA happens per tile. Returned dtype equals Q.dtype. No backward pass supplied.
    """
    if any(type(size) is not int or size < 1 for size in (block_q, block_k)):
        raise ValueError('Tile sizes must be positive integers.')
    dtype = q.dtype
    q, k, v = _prepare(q, k, v, causal, query_offset)
    batch, heads, queries, dimension = q.shape
    keys, value_dimension = k.shape[2], v.shape[-1]
    groups = heads // k.shape[1]
    scale = 1.0 / math.sqrt(dimension)
    result = torch.empty((batch, heads, queries, value_dimension), dtype=q.dtype, device=q.device)
    for q_start in range(0, queries, block_q):
        q_end = min(q_start + block_q, queries)
        query_tile = q[:, :, q_start:q_end]
        row_shape = (batch, heads, q_end - q_start, 1)
        maximum = torch.full(row_shape, float('-inf'), dtype=q.dtype, device=q.device)
        denominator = torch.zeros(row_shape, dtype=q.dtype, device=q.device)
        numerator = torch.zeros((*row_shape[:-1], value_dimension), dtype=q.dtype, device=q.device)
        for k_start in range(0, keys, block_k):
            # Every later key tile is future-only for this complete query tile.
            if causal and k_start > q_end - 1 + query_offset:
                break
            k_end = min(k_start + block_k, keys)
            key_tile, value_tile = k[:, :, k_start:k_end], v[:, :, k_start:k_end]
            if groups != 1:
                key_tile = key_tile.repeat_interleave(groups, dim=1)
                value_tile = value_tile.repeat_interleave(groups, dim=1)
            scores = (query_tile @ key_tile.transpose(-1, -2)) * scale
            if causal:
                scores = _mask(scores, q_start, k_start, query_offset)
            new_maximum = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
            # Initially/all-masked rows have maximum=-inf; avoid -inf - -inf.
            safe_maximum = torch.where(torch.isfinite(new_maximum), new_maximum,
                                       torch.zeros_like(new_maximum))
            rescale = torch.exp(maximum - safe_maximum)
            weights = torch.exp(scores - safe_maximum)
            denominator = rescale * denominator + weights.sum(dim=-1, keepdim=True)
            numerator = rescale * numerator + weights @ value_tile
            maximum = new_maximum
        result[:, :, q_start:q_end] = numerator / torch.where(
            denominator > 0, denominator, torch.ones_like(denominator))
    return result.to(dtype)
