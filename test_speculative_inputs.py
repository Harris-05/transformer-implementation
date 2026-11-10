import unittest

from tokenizers import Tokenizer, models, normalizers
from transformers import PreTrainedTokenizerFast

from speculative_inputs import assert_compatible_tokenizers


class SpeculativeInputTests(unittest.TestCase):
    def tokenizer(self, swapped=False, lowercase=False):
        vocab = {'<unk>': 0, '<pad>': 1, '<eos>': 2, 'A': 4 if swapped else 3, 'B': 3 if swapped else 4}
        backend = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
        if lowercase:
            backend.normalizer = normalizers.Lowercase()
        return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>',
                                     pad_token='<pad>', eos_token='<eos>')

    def test_tokenizer_compatibility_checks_ids_specials_and_actual_backend(self):
        report = assert_compatible_tokenizers(self.tokenizer(), self.tokenizer(), 8, 8)
        self.assertTrue(report['compatible'])
        self.assertEqual(report['token_count'], 5)
        for other in (self.tokenizer(swapped=True), self.tokenizer(lowercase=True)):
            with self.assertRaises(ValueError):
                assert_compatible_tokenizers(self.tokenizer(), other, 8, 8)
        other = self.tokenizer()
        other.eos_token = 'A'
        with self.assertRaises(ValueError):
            assert_compatible_tokenizers(self.tokenizer(), other, 8, 8)
        with self.assertRaises(ValueError):
            assert_compatible_tokenizers(self.tokenizer(), self.tokenizer(), 8, 9)


if __name__ == '__main__':
    unittest.main()
