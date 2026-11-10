"""Lazy local inference; no model weights are loaded just to open the browser."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from prepare_model_inputs import build_messages, load_tokenizer, sha256
from screening import aggregate, explain_prediction


ROOT = Path(__file__).resolve().parent


class ModelService:
    def __init__(self, root=ROOT, training=None, device='auto'):
        self.root = Path(root)
        self.training = Path(training) if training else self.root / 'results/finetuning_qlora_v1'
        self.device_choice = device
        self.config = json.loads((self.root / 'configs/model_input_config.json').read_text())
        self.model = self.draft = self.tokenizer = None
        self.provenance = {}

    def status(self):
        return {'model_id': self.config['model_id'], 'loaded': self.model is not None,
                'draft_loaded': self.draft is not None, 'adapter_present': (self.training / 'adapter/adapter_model.safetensors').is_file(),
                'device_choice': self.device_choice, 'provenance': self.provenance,
                'loading_note': 'First inference downloads pinned base weights if absent. CPU FP32 needs approximately 16 GB for base weights alone; CUDA uses NF4 with FP32 compute.'}

    def load(self):
        if self.model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from benchmark_kv_cache import attention_backend
        summary = json.loads((self.training / 'training_summary.json').read_text())
        manifest = json.loads((self.training / 'training_manifest.json').read_text())
        if not summary['complete'] or summary['signature'] != manifest['signature'] or manifest['identity']['model'] != self.config:
            raise ValueError('The selected adapter metadata does not match the frozen model/prompt.')
        for name, digest in summary['adapter_sha256'].items():
            if Path(name).name != name or sha256(self.training / 'adapter' / name) != digest:
                raise ValueError(f'Selected adapter checksum differs: {name}')
        tokenizer, _ = load_tokenizer(self.root)
        device = 'cuda:0' if self.device_choice == 'cuda' or (self.device_choice == 'auto' and torch.cuda.is_available()) else 'cpu'
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Select CPU or start on a GPU host.')
        options = {'revision': self.config['revision'], 'dtype': torch.float32,
                   'attn_implementation': 'sdpa', 'trust_remote_code': False, 'device_map': {'': device}}
        if device.startswith('cuda'):
            options['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)
        torch.set_float32_matmul_precision('highest')
        with attention_backend('math'):
            base = AutoModelForCausalLM.from_pretrained(self.config['model_id'], **options)
            model = PeftModel.from_pretrained(base, self.training / 'adapter', is_trainable=False).eval()
        self.tokenizer, self.model = tokenizer, model
        self.provenance = {'model_id': self.config['model_id'], 'revision': self.config['revision'],
            'training_signature': summary['signature'], 'adapter_sha256': summary['adapter_sha256'],
            'device': device, 'precision': 'NF4 weights / FP32 compute' if device.startswith('cuda') else 'unquantized FP32',
            'attention_backend': 'math SDPA', 'prompt_version': self.config['prompt_version'],
            'protocol_note': 'Application FP32 protocol is separate from the saved FP16 classification evaluation.',
            'max_sequence_length': self.config['max_sequence_length'], 'max_new_tokens': self.config['max_new_tokens']}

    def load_draft(self):
        self.load()
        if self.draft is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM
        from speculative_inputs import load_draft_tokenizer, assert_compatible_tokenizers
        tokenizer, lock = load_draft_tokenizer(self.root)
        configuration = json.loads((self.root / 'artifacts/tokenizer/qwen3-0.6b/config.json').read_text())
        assert_compatible_tokenizers(self.tokenizer, tokenizer, self.model.config.vocab_size, configuration['vocab_size'])
        device = next(self.model.parameters()).device
        self.draft = AutoModelForCausalLM.from_pretrained(lock['model_id'], revision=lock['revision'],
            dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            device_map={'': str(device)}, attn_implementation='sdpa', trust_remote_code=False).eval()
        self.provenance['draft_model_id'] = lock['model_id']
        self.provenance['draft_revision'] = lock['revision']

    def encode(self, note, criterion):
        import torch
        prompt = build_messages({'note': note, 'criterion_type': criterion['type'], 'criterion_text': criterion['text']})
        tokens = self.tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
        if len(tokens) + self.config['max_new_tokens'] > self.config['max_sequence_length']:
            raise ValueError(f'Criterion prompt has {len(tokens)} tokens; limit is {self.config["max_sequence_length"] - self.config["max_new_tokens"]}. Shorten the record or criterion. Nothing was truncated.')
        return torch.tensor([tokens], dtype=torch.long, device=next(self.model.parameters()).device)

    def decode(self, inputs, mode, budget=None, measure=True):
        from benchmark_kv_cache import attention_backend
        from decoding import greedy_decode
        from speculative_decoding import speculative_decode
        budget = self.config['max_new_tokens'] if budget is None else budget
        with attention_backend('math'):
            if mode in ('cached', 'uncached'):
                result = asdict(greedy_decode(self.model, inputs, max_new_tokens=budget,
                    use_cache=mode == 'cached', eos_token_id=self.config['eos_token_id'], measure=measure))
                result.update(target_forward_calls=len(result['token_ids']), draft_forward_calls=0)
                return result
            if mode not in ('speculative_2', 'speculative_4') or self.draft is None:
                raise ValueError('Unknown inference mode or draft is not loaded.')
            return asdict(speculative_decode(self.model, self.draft, inputs, max_new_tokens=budget,
                draft_length=int(mode[-1]), eos_token_id=self.config['eos_token_id'], measure=measure))

    def screen(self, request, progress):
        progress('Loading the selected model. The first weight download may take several minutes.')
        self.load()
        # Validate every budget before producing any decisions.
        encoded = [self.encode(request['note'], criterion) for criterion in request['criteria']]
        results = []
        for i, (criterion, inputs) in enumerate(zip(request['criteria'], encoded)):
            progress(f'Assessing criterion {i + 1} of {len(encoded)}.')
            result = self.decode(inputs, 'cached', self.config['max_new_tokens'])
            raw = self.tokenizer.decode(result['token_ids'], skip_special_tokens=True)
            results.append({**criterion, **explain_prediction(raw, criterion['type'], request['note']),
                            'generation': result})
        return {**request, 'results': results, 'outcome': aggregate(results), 'provenance': dict(self.provenance),
                'created_at': datetime.now(timezone.utc).isoformat()}

    def benchmark(self, request, modes, progress):
        progress('Loading model(s) for a live comparison on the first entered criterion.')
        self.load()
        if any(mode.startswith('speculative_') for mode in modes):
            self.load_draft()
        inputs = self.encode(request['note'], request['criteria'][0])
        references = self.decode(inputs, 'cached', measure=False)['token_ids']
        for mode in modes:
            self.decode(inputs, mode, measure=False)
        trials = []
        for repeat in range(3):
            order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            for mode in order:
                progress(f'Repetition {repeat + 1} of 3: {mode}.')
                value = self.decode(inputs, mode)
                trials.append({'mode': mode, 'repeat': repeat, 'matches_greedy': value['token_ids'] == references, 'result': value})
        summaries = []
        base_seconds = statistics.median(t['result']['total_seconds'] for t in trials if t['mode'] == 'cached')
        for mode in modes:
            rows = [t for t in trials if t['mode'] == mode]
            equal = all(row['matches_greedy'] for row in rows)
            median = statistics.median(row['result']['total_seconds'] for row in rows)
            proposed = sum(row['result'].get('proposed_tokens', 0) for row in rows)
            summaries.append({'mode': mode, 'median_seconds': median,
                'tokens_per_second': statistics.median(len(row['result']['token_ids']) / row['result']['total_seconds'] for row in rows),
                'tokens_equal': equal, 'speedup_vs_cached': base_seconds / median if equal and all(t['matches_greedy'] for t in trials if t['mode'] == 'cached') else None,
                'acceptance_rate': sum(row['result'].get('accepted_tokens', 0) for row in rows) / proposed if proposed else None,
                'peak_allocated_gib': max((row['result']['peak_allocated_gib'] for row in rows if row['result']['peak_allocated_gib'] is not None), default=None)})
        return {'summaries': summaries, 'trials': trials, 'provenance': dict(self.provenance),
            'memory_scope': 'All loaded models remain resident in every mode. CPU GPU-memory peaks are unavailable.',
            'input_tokens': inputs.shape[1], 'repetitions': 3, 'warmups': 1,
            'reference_token_ids': references, 'criterion': request['criteria'][0]}
