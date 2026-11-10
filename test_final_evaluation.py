import json
import unittest

from evaluate_final import build_test_records, validate_saved_predictions


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 3]


class FinalEvaluationTests(unittest.TestCase):
    def frame(self):
        import pandas as pd
        return pd.DataFrame([{'annotation_id': 3, 'patient_id': 'test-a', 'trial_id': 'trial', 'trial_title': 'Title',
            'note': '0. Age 45.', 'criterion_type': 'inclusion', 'criterion_text': 'Age at least 18.',
            'expert_eligibility': 'included', 'expert_sentences': '[0]'}])

    def test_only_test_patients_and_all_rows_are_kept_and_no_truncation(self):
        config = {'max_new_tokens': 8, 'max_sequence_length': 16}
        rows = build_test_records(self.frame(), {'test-a': 1}, Tokenizer(), config)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['gold'], {'label': 'included', 'sentence_ids': [0]})
        self.assertNotIn('expert_eligibility', json.loads(rows[0]['prompt'][1]['content']))
        for data, allowed, settings in ((self.frame(), {'other': 1}, config),
                (self.frame(), {'test-a': 2}, config),
                (self.frame().assign(criterion_text=None), {'test-a': 1}, config),
                (self.frame(), {'test-a': 1}, {**config, 'max_sequence_length': 4})):
            with self.assertRaises(ValueError):
                build_test_records(data, allowed, Tokenizer(), settings)

    def test_resumed_predictions_must_match_original_input_gold_and_signature(self):
        rows = build_test_records(self.frame(), {'test-a': 1}, Tokenizer(), {'max_new_tokens': 8, 'max_sequence_length': 16})
        saved = [{**rows[0], 'run_signature': 'run'}]
        validate_saved_predictions(saved, rows, 'run')
        for bad in ([saved[0], saved[0]], [{**saved[0], 'run_signature': 'other'}],
                    [{**saved[0], 'gold': {'label': 'not included', 'sentence_ids': []}}]):
            with self.assertRaises(ValueError):
                validate_saved_predictions(bad, rows, 'run')


if __name__ == '__main__':
    unittest.main()
