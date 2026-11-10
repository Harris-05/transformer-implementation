"""Pinned draft tokenizer compatibility, development cases and paired loading."""

import hashlib
import json
from pathlib import Path

from prepare_model_inputs import load_tokenizer, sha256


def assert_compatible_tokenizers(target, draft, target_vocab_size, draft_vocab_size):
    if not target.is_fast or not draft.is_fast:
        raise ValueError('The experiment requires verified fast-tokenizer backends.')
    if target.get_vocab() != draft.get_vocab() or target_vocab_size != draft_vocab_size:
        raise ValueError('Draft and target token IDs or output vocabulary sizes differ.')
    special = {name: getattr(target, name) for name in ('bos_token_id', 'eos_token_id', 'pad_token_id')}
    if any(getattr(draft, name) != value for name, value in special.items()):
        raise ValueError('Draft and target special token IDs differ.')
    backend = json.loads(target.backend_tokenizer.to_str())
    if backend != json.loads(draft.backend_tokenizer.to_str()):
        raise ValueError('Configured tokenization backends differ despite matching vocabulary IDs.')
    if max(target.get_vocab().values()) >= target_vocab_size:
        raise ValueError('Tokenizer emits IDs outside the model vocabulary.')
    return {'compatible': True, 'token_count': len(target), 'special_token_ids': special,
            'model_vocab_size': target_vocab_size,
            'backend_sha256': hashlib.sha256(json.dumps(backend, sort_keys=True).encode()).hexdigest(),
            'prompt_template': 'target_only', 'draft_template_used': False}


def load_draft_tokenizer(root, offline=False):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    lock = json.loads((root / 'configs/draft_tokenizer_lock.json').read_text(encoding='utf-8'))
    destination = root / 'artifacts/tokenizer/qwen3-0.6b'
    missing = False
    for name, digest in lock['file_sha256'].items():
        path = destination / name
        if not path.is_file():
            missing = True
        elif sha256(path) != digest:
            raise ValueError(f'Draft tokenizer asset changed: {name}')
    if missing:
        if offline:
            raise ValueError('Draft tokenizer assets are missing; prepare once online.')
        snapshot_download(repo_id=lock['model_id'], revision=lock['revision'], token=False,
            local_dir=destination, cache_dir=root / 'artifacts/tokenizer/.hub_cache',
            allow_patterns=list(lock['file_sha256']))
        for name, digest in lock['file_sha256'].items():
            if sha256(destination / name) != digest:
                raise ValueError(f'Downloaded draft asset differs from lock: {name}')
    return AutoTokenizer.from_pretrained(destination, local_files_only=True, use_fast=True,
                                         trust_remote_code=False), lock


def prepare(root, training_output, offline=False):
    from benchmark_kv_cache import prepare_cases
    settings = json.loads((root / 'configs/speculative_config.json').read_text(encoding='utf-8'))
    if (not settings['draft_lengths'] or len(set(settings['draft_lengths'])) != len(settings['draft_lengths'])
            or any(type(k) is not int or k < 1 for k in settings['draft_lengths'])
            or type(settings['repetitions']) is not int or settings['repetitions'] < 1
            or type(settings['warmups']) is not int or settings['warmups'] < 0):
        raise ValueError('Draft lengths/repetitions/warmups are invalid.')
    if (settings['sdpa_backend'] != 'math' or settings['target_compute_dtype'] != 'float32'
            or settings['draft_dtype'] != 'float16'):
        raise ValueError('This controlled pair requires math SDPA, FP32 target compute and FP16 draft.')
    target_tokenizer, target_lock = load_tokenizer(root, offline=offline)
    draft_tokenizer, draft_lock = load_draft_tokenizer(root, offline=offline)
    if settings['draft_model_id'] != draft_lock['model_id'] or settings['draft_revision'] != draft_lock['revision']:
        raise ValueError('Draft model and tokenizer lock disagree.')
    target_config = json.loads((root / 'artifacts/tokenizer/qwen3-4b-instruct-2507/config.json').read_text())
    draft_config = json.loads((root / 'artifacts/tokenizer/qwen3-0.6b/config.json').read_text())
    compatibility = assert_compatible_tokenizers(target_tokenizer, draft_tokenizer,
        target_config['vocab_size'], draft_config['vocab_size'])
    pairs = {tuple(pair) for pair in settings['stress_pairs']}
    if len(pairs) != len(settings['stress_pairs']) or not pairs or any(len(p) != 2 for p in pairs):
        raise ValueError('Stress cases must be distinct input/output pairs.')
    case_settings = {'input_lengths': sorted({p[0] for p in pairs}),
                     'output_lengths': sorted({p[1] for p in pairs})}
    config, training, validation_hash, stress, clinical = prepare_cases(root, training_output, case_settings)
    stress = [c for c in stress if (c['input_tokens'], c['output_budget']) in pairs]
    return {'model': config, 'settings': settings, 'training': training,
            'validation_sha256': validation_hash, 'tokenizer_compatibility': compatibility,
            'draft_tokenizer_lock': draft_lock, 'target_tokenizer_lock': target_lock,
            'cases': stress + clinical}


def load_pair(prepared, training_output):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    config, settings = prepared['model'], prepared['settings']
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)
    base = AutoModelForCausalLM.from_pretrained(config['model_id'], revision=config['revision'],
        quantization_config=quantization, device_map={'': 0}, dtype=torch.float32,
        attn_implementation='sdpa', trust_remote_code=False)
    target = PeftModel.from_pretrained(base, training_output / 'adapter', is_trainable=False).eval()
    target.config.pad_token_id = config['pad_token_id']
    draft = AutoModelForCausalLM.from_pretrained(settings['draft_model_id'], revision=settings['draft_revision'],
        device_map={'': 0}, dtype=torch.float16, attn_implementation='sdpa', trust_remote_code=False).eval()
    return target, draft
