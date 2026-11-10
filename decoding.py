"""Explicit batch-one greedy generation for the pinned Qwen3 model.

No generate() calls: forward passes, positions, masks and cache ownership live
here. See KV_CACHE.md for timing definitions and supported input restrictions.
"""

from dataclasses import dataclass, field
import time

import torch
from transformers import DynamicCache


@dataclass
class DecodeResult:
    token_ids: list[int]
    stopped_on_eos: bool
    decode_steps: int
    cache_tokens: int
    cache_bytes: int
    prefill_seconds: float | None = None
    decode_seconds: float | None = None
    total_seconds: float | None = None
    initial_allocated_gib: float | None = None
    peak_allocated_gib: float | None = None
    peak_reserved_gib: float | None = None
    trace: list[dict] = field(default_factory=list)


def first_divergence(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def cache_size(cache):
    if cache is None:
        return 0, 0
    size = sum(tensor.numel() * tensor.element_size() for layer in cache.layers
               for tensor in (layer.keys, layer.values) if tensor is not None)
    return int(cache.get_seq_length()), size


@torch.inference_mode()
def greedy_decode(model, input_ids, *, max_new_tokens, use_cache,
                  eos_token_id=None, attention_mask=None, trace=False, measure=False):
    """Return generated tokens including EOS, with a fresh private cache per call.

    Supports one unpadded, nonempty long-tensor sequence. EOS=None deliberately
    generates the full budget, for fixed-work benchmarks. Timing excludes input
    tokenization, buffer allocation, model loading and final CPU token transfer.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Expected one nonempty unpadded sequence with shape [1, length].")
    if input_ids.dtype != torch.long:
        raise ValueError("Token IDs must have dtype torch.long.")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer.")
    if attention_mask is not None and (attention_mask.shape != input_ids.shape
            or attention_mask.device != input_ids.device or not bool(torch.all(attention_mask == 1))):
        raise ValueError("Only an all-ones mask matching the unpadded input is supported.")
    if model.training:
        raise ValueError("Call model.eval() before deterministic decoding.")
    prompt_length = input_ids.shape[1]
    capacity = prompt_length + max_new_tokens
    if capacity > model.config.max_position_embeddings:
        raise ValueError("Prompt plus generation budget exceeds model context; no truncation is performed.")
    eos_ids = [] if eos_token_id is None else ([eos_token_id] if type(eos_token_id) is int else list(eos_token_id))
    if any(type(i) is not int or not 0 <= i < model.config.vocab_size for i in eos_ids):
        raise ValueError("EOS IDs must be vocabulary token integers.")

    device = input_ids.device
    gpu = measure and device.type == "cuda"
    initial_memory = None
    if gpu:
        torch.cuda.synchronize(device)
        initial_memory = torch.cuda.memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
    # Identical token, position and mask buffers for both paths, never mutate input.
    sequence = torch.empty((1, capacity), dtype=torch.long, device=device)
    sequence[:, :prompt_length] = input_ids
    positions = torch.arange(capacity, device=device).unsqueeze(0)
    mask = torch.ones_like(sequence)
    eos = torch.tensor(eos_ids, dtype=torch.long, device=device)
    cache = DynamicCache(config=model.config) if use_cache else None
    steps, stopped, records = 0, False, []

    def synchronize():
        if gpu:
            torch.cuda.synchronize(device)

    synchronize()
    started = time.perf_counter() if measure else 0.0
    prefill_end = started
    for step in range(max_new_tokens):
        end = prompt_length + step
        start = end - 1 if use_cache and step else 0
        outputs = model(input_ids=sequence[:, start:end],
            attention_mask=mask[:, :end], position_ids=positions[:, start:end],
            past_key_values=cache, use_cache=use_cache, logits_to_keep=1)
        if use_cache:
            cache = outputs.past_key_values
            if cache is None or int(cache.get_seq_length()) != end:
                raise RuntimeError("Cache did not grow to the processed prefix length.")
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        sequence[:, end] = next_token
        steps = step + 1
        stopped = bool(torch.isin(next_token, eos).any()) if eos_ids else False
        if trace:
            records.append({"step": step, "input_tokens": end - start,
                "mask_tokens": end, "position_start": start, "position_end": end - 1,
                "cache_tokens": int(cache.get_seq_length()) if cache is not None else 0})
        if step == 0:
            synchronize()
            prefill_end = time.perf_counter() if measure else started
        if stopped:
            break
    synchronize()
    finished = time.perf_counter() if measure and steps > 1 else prefill_end
    cache_tokens, cache_bytes = cache_size(cache)
    return DecodeResult(
        token_ids=sequence[0, prompt_length:prompt_length + steps].tolist(),
        stopped_on_eos=stopped, decode_steps=steps - 1,
        cache_tokens=cache_tokens, cache_bytes=cache_bytes,
        prefill_seconds=prefill_end - started if measure else None,
        decode_seconds=finished - prefill_end if measure else None,
        total_seconds=finished - started if measure else None,
        initial_allocated_gib=initial_memory,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if gpu else None,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30 if gpu else None,
        trace=records)


@torch.inference_mode()
def diagnose_divergence(model, input_ids, common_tokens):
    """Compare last logits under the identical prefix, outside timed generation."""
    prefix = torch.tensor([common_tokens], dtype=torch.long, device=input_ids.device)
    full = torch.cat((input_ids, prefix), dim=1)
    positions = torch.arange(full.shape[1], device=full.device).unsqueeze(0)
    plain = model(input_ids=full, attention_mask=torch.ones_like(full),
                  position_ids=positions, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    cache = DynamicCache(config=model.config)
    length = input_ids.shape[1]
    output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        position_ids=positions[:, :length], past_key_values=cache, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    for index in range(len(common_tokens)):
        end = length + index + 1
        output = model(input_ids=full[:, end - 1:end], attention_mask=torch.ones_like(full[:, :end]),
            position_ids=positions[:, end - 1:end], past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
    cached = output.logits[0, -1].float()

    def describe(logits):
        values, indices = logits.topk(min(5, logits.numel()))
        return {"top_ids": indices.tolist(), "top_logits": values.tolist(),
                "top_two_margin": (values[0] - values[1]).item()}

    return {"prefix_tokens": full.shape[1], "uncached": describe(plain),
            "cached": describe(cached), "max_abs_logit_difference": (plain - cached).abs().max().item()}
