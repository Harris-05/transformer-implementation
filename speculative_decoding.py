"""Explicit batch-one greedy speculative decoding with separate full KV caches."""

from dataclasses import dataclass, field
import time

import torch
from transformers import DynamicCache

from decoding import cache_size


@dataclass
class SpeculativeResult:
    token_ids: list[int]
    stopped_on_eos: bool
    proposed_tokens: int
    accepted_tokens: int
    bonus_tokens: int
    rejection_rounds: int
    target_forward_calls: int
    draft_forward_calls: int
    target_cache_tokens: int
    draft_cache_tokens: int
    target_cache_bytes: int
    draft_cache_bytes: int
    total_seconds: float | None = None
    initial_allocated_gib: float | None = None
    peak_allocated_gib: float | None = None
    peak_reserved_gib: float | None = None
    trace: list[dict] = field(default_factory=list)


def crop_cache(cache, length):
    """Pinned Transformers API: negative counts remove suffix tokens."""
    old = int(cache.get_seq_length())
    if length < 0 or length > old:
        raise ValueError('Cropping cannot grow a cache or request a negative length.')
    if old > length:
        cache.crop(-(old - length))
    if int(cache.get_seq_length()) != length:
        raise RuntimeError('Cache rollback did not reach the committed prefix.')


@torch.inference_mode()
def speculative_decode(target, draft, input_ids, *, max_new_tokens,
                       draft_length=4, eos_token_id=None, trace=False, measure=False):
    """Draft candidates, batch-verify, correct the first rejection and roll back.

    Both caches describe committed tokens except the last at round boundaries.
    The first round includes the complete uncached prompt. Output includes EOS.
    Mathematical greedy equivalence is tested, not assumed for reduced precision.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1 or input_ids.dtype != torch.long:
        raise ValueError('Expected one nonempty unpadded long-tensor sequence.')
    if any(type(n) is not int or n < 1 for n in (max_new_tokens, draft_length)):
        raise ValueError('Generation budget and draft length must be positive integers.')
    if target.training or draft.training:
        raise ValueError('Both models must be in evaluation mode.')
    if target.config.vocab_size != draft.config.vocab_size:
        raise ValueError('Models must have the same output vocabulary size and verified token IDs.')
    device = input_ids.device
    if any(next(model.parameters()).device != device for model in (target, draft)):
        raise ValueError('Both models and inputs must share one device.')
    prompt_length = input_ids.shape[1]
    capacity = prompt_length + max_new_tokens
    if capacity > min(target.config.max_position_embeddings, draft.config.max_position_embeddings):
        raise ValueError('Prompt and generation budget exceed one model context; no truncation is performed.')
    eos = [] if eos_token_id is None else ([eos_token_id] if type(eos_token_id) is int else list(eos_token_id))
    if any(type(n) is not int or not 0 <= n < target.config.vocab_size for n in eos):
        raise ValueError('EOS IDs must be valid vocabulary integers.')

    gpu = measure and device.type == 'cuda'
    initial = None
    if gpu:
        torch.cuda.synchronize(device)
        initial = torch.cuda.memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
    sequence = torch.empty((1, capacity), dtype=torch.long, device=device)
    sequence[:, :prompt_length] = input_ids
    mask = torch.ones_like(sequence)
    positions = torch.arange(capacity, device=device).unsqueeze(0)
    target_cache, draft_cache = DynamicCache(config=target.config), DynamicCache(config=draft.config)
    length = prompt_length
    proposed = accepted = bonuses = rejections = target_calls = draft_calls = 0
    records, stopped = [], False

    def forward(model, cache, start, end, keep):
        return model(input_ids=sequence[:, start:end], attention_mask=mask[:, :end],
                     position_ids=positions[:, start:end], past_key_values=cache,
                     use_cache=True, logits_to_keep=keep)

    if gpu:
        torch.cuda.synchronize(device)
    started = time.perf_counter() if measure else 0.0
    while length < capacity:
        before = length
        remaining = capacity - length
        count = min(draft_length, remaining)
        draft_before = int(draft_cache.get_seq_length())
        target_before = int(target_cache.get_seq_length())
        if length != prompt_length and (draft_before != length - 1 or target_before != length - 1):
            raise RuntimeError('Draft and target caches are not synchronized at a round boundary.')
        output = forward(draft, draft_cache, draft_before, length, 1)
        draft_cache = output.past_key_values
        draft_calls += 1
        for index in range(count):
            candidate = output.logits[:, -1, :].argmax(dim=-1)
            sequence[:, before + index] = candidate
            if eos and candidate.item() in eos:
                count = index + 1
                break
            if index + 1 < count:
                output = forward(draft, draft_cache, before + index, before + index + 1, 1)
                draft_cache = output.past_key_values
                draft_calls += 1
        candidates = sequence[0, before:before + count].tolist()
        proposed += count
        # Last K+1 rows predict the K candidates and one possible target bonus.
        output = forward(target, target_cache, target_before, before + count, count + 1)
        target_cache = output.past_key_values
        target_calls += 1
        predictions = output.logits[0].argmax(dim=-1).tolist()
        if len(predictions) != count + 1:
            raise RuntimeError('Target must return the final K+1 prediction rows.')
        matching = 0
        for candidate, prediction in zip(candidates, predictions):
            if candidate != prediction:
                break
            matching += 1
            if candidate in eos:
                break
        emitted = candidates[:matching]
        rejected = matching < count
        accepted += matching
        if rejected:
            emitted.append(predictions[matching])
            rejections += 1
        elif not (emitted and emitted[-1] in eos) and matching < remaining:
            emitted.append(predictions[count])
            bonuses += 1
        if not emitted or len(emitted) > remaining:
            raise RuntimeError('A speculative round must make progress within its output budget.')
        sequence[:, before:before + len(emitted)] = torch.tensor([emitted], dtype=torch.long, device=device)
        length += len(emitted)
        desired = length - 1
        draft_after_proposal = int(draft_cache.get_seq_length())
        target_after_verification = int(target_cache.get_seq_length())
        # On full acceptance plus bonus, the final candidate has not yet entered
        # the draft cache. Process it, but leave the emitted bonus pending.
        if int(draft_cache.get_seq_length()) < desired:
            output = forward(draft, draft_cache, int(draft_cache.get_seq_length()), desired, 1)
            draft_cache = output.past_key_values
            draft_calls += 1
        crop_cache(target_cache, desired)
        crop_cache(draft_cache, desired)
        stopped = emitted[-1] in eos
        if trace:
            records.append({'prefix_tokens': before, 'committed_tokens': length,
                'proposed': count, 'accepted': matching, 'emitted': emitted, 'rejected': rejected,
                'target_cache_before': target_before, 'draft_cache_before': draft_before,
                'target_cache_after_verification': target_after_verification,
                'draft_cache_after_proposal': draft_after_proposal,
                'target_cache_after': int(target_cache.get_seq_length()),
                'draft_cache_after': int(draft_cache.get_seq_length())})
        if stopped:
            break
    if gpu:
        torch.cuda.synchronize(device)
    finished = time.perf_counter() if measure else started
    target_tokens, target_bytes = cache_size(target_cache)
    draft_tokens, draft_bytes = cache_size(draft_cache)
    return SpeculativeResult(token_ids=sequence[0, prompt_length:length].tolist(), stopped_on_eos=stopped,
        proposed_tokens=proposed, accepted_tokens=accepted, bonus_tokens=bonuses,
        rejection_rounds=rejections, target_forward_calls=target_calls, draft_forward_calls=draft_calls,
        target_cache_tokens=target_tokens, draft_cache_tokens=draft_tokens,
        target_cache_bytes=target_bytes, draft_cache_bytes=draft_bytes,
        total_seconds=finished - started if measure else None,
        initial_allocated_gib=initial,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if gpu else None,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30 if gpu else None,
        trace=records)
