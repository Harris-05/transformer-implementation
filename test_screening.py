import json
import unittest

from screening import validate_request, explain_prediction, aggregate, export_html


class ScreeningTests(unittest.TestCase):
    def request(self):
        return {'patient_reference': 'Case A', 'trial_name': 'Trial A',
                'note': 'Age 45.\nNo history of diabetes.',
                'criteria': [{'type': 'inclusion', 'text': 'Age at least 18.'}]}

    def test_numbers_original_lines_and_rejects_bad_inputs_without_truncation(self):
        value = validate_request(self.request())
        self.assertEqual(value['note'], '0. Age 45.\n1. No history of diabetes.')
        value['note'] = '0. Age 45.\n1. No history of diabetes.'
        self.assertEqual(validate_request(value)['note'], value['note'])
        for update in ({'note': ''}, {'note': '1. First line.'}, {'criteria': []},
                       {'criteria': [{'type': 'binary', 'text': 'Age?'}]},
                       {'expert_eligibility': 'included'}, {'note': 'a' * 24001}):
            with self.subTest(update=list(update)), self.assertRaises(ValueError):
                validate_request({**self.request(), **update})

    def test_exact_evidence_and_unknown_are_explained_without_inventing_facts(self):
        note = '0. Age 45.\n1. No history of diabetes.'
        result = explain_prediction('{"label":"included","sentence_ids":[0]}', 'inclusion', note)
        self.assertEqual(result['evidence'], [{'sentence_id': 0, 'text': 'Age 45.'}])
        result = explain_prediction('{"label":"not enough information","sentence_ids":[]}', 'exclusion', note)
        self.assertTrue(result['needs_review'])
        self.assertIn('not establish', result['explanation'])
        for raw in ('bad JSON', '{"label":"included","sentence_ids":[99]}',
                    '{"label":"excluded","sentence_ids":[]}'):
            result = explain_prediction(raw, 'inclusion', note)
            self.assertFalse(result['prediction']['output_valid'])
            self.assertTrue(result['needs_review'])

    def test_preliminary_aggregation_preserves_unknown_and_unsupported_decisions(self):
        note = '0. Age 45.'
        make = lambda label, ids: explain_prediction(json.dumps({'label': label, 'sentence_ids': ids}), 'inclusion', note)
        self.assertEqual(aggregate([make('included', [0])])['code'], 'provisionally_satisfied')
        self.assertEqual(aggregate([make('not included', [0])])['code'], 'potential_conflict')
        self.assertEqual(aggregate([make('not enough information', [])])['code'], 'more_information')
        self.assertEqual(aggregate([make('included', [])])['code'], 'review_required')
        self.assertEqual(aggregate([make('not applicable', [])])['code'], 'not_applicable')

    def test_export_escapes_record_and_contains_protocol_and_evidence(self):
        record = validate_request(self.request())
        record['patient_reference'] = '<script>alert(1)</script>'
        result = explain_prediction('{"label":"included","sentence_ids":[0]}', 'inclusion', record['note'])
        report = {**record, 'results': [{**record['criteria'][0], **result}],
                  'outcome': aggregate([result]), 'provenance': {'model_id': 'Qwen', 'precision': 'FP32'},
                  'created_at': '2026-11-10'}
        html = export_html(report)
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertIn('Age 45.', html)
        self.assertIn('FP32', html)
        self.assertIn('not a clinical enrollment decision', html)


if __name__ == '__main__':
    unittest.main()
