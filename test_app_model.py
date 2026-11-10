import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from app_model import ModelService
from screening import validate_request


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(1, 7))

    def decode(self, ids, **kwargs):
        return 'Actual tiny-model tokens: ' + ','.join(str(i) for i in ids)


class AppModelTests(unittest.TestCase):
    def service(self):
        torch.set_num_threads(1)
        def tiny(seed):
            torch.manual_seed(seed)
            config = Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=64,pad_token_id=0,eos_token_id=63)
            config._attn_implementation='sdpa'
            return Qwen3ForCausalLM(config).double().eval()
        service = ModelService()
        service.model, service.draft, service.tokenizer = tiny(31), tiny(77), TinyTokenizer()
        service.config = {**service.config, 'max_new_tokens': 8, 'max_sequence_length': 64, 'eos_token_id': 63}
        service.provenance = {'device': 'cpu', 'precision': 'miniature FP64 test only'}
        return service

    def request(self):
        return validate_request({'note':'Age 45.', 'criteria':[{'type':'inclusion','text':'Age at least 18.'}]})

    def test_real_tiny_live_comparison_and_screening_use_the_configured_budget(self):
        service=self.service()
        result=service.benchmark(self.request(),['cached','uncached','speculative_2'],lambda _:None)
        self.assertEqual(len(result['trials']),9)
        self.assertTrue(all(row['tokens_equal'] for row in result['summaries']))
        self.assertLessEqual(len(result['reference_token_ids']),8)
        self.assertTrue(all(row['peak_allocated_gib'] is None for row in result['summaries']))
        report=service.screen(self.request(),lambda _:None)
        self.assertEqual(report['outcome']['code'],'review_required')
        self.assertFalse(report['results'][0]['prediction']['output_valid'])


if __name__=='__main__':unittest.main()
