"""Input validation, source-grounded explanations and portable screening reports."""

from html import escape
import json
import re

from baseline import parse_prediction
from eda import parse_note


NOTICE = 'Educational screening prototype; not a clinical enrollment decision.'


def validate_request(value):
    if not isinstance(value, dict) or set(value) - {'patient_reference', 'trial_name', 'note', 'criteria'}:
        raise ValueError('Supply only patient_reference, trial_name, note and criteria.')
    note = value.get('note')
    if not isinstance(note, str) or not note.strip() or len(note) > 24000:
        raise ValueError('Enter a patient record of 1-24,000 characters. Long notes are rejected, not truncated.')
    note = note.strip()
    if not re.search(r'(?m)^\s*\d+\.\s+', note):
        lines = [line.strip() for line in note.splitlines() if line.strip()]
        note = '\n'.join(f'{i}. {line}' for i, line in enumerate(lines))
    parse_note(note)
    criteria = value.get('criteria')
    if not isinstance(criteria, list) or not 1 <= len(criteria) <= 12:
        raise ValueError('Enter between 1 and 12 criteria, one requirement per line.')
    cleaned = []
    for row in criteria:
        if (not isinstance(row, dict) or set(row) != {'type', 'text'}
                or row['type'] not in ('inclusion', 'exclusion')
                or not isinstance(row['text'], str) or not 1 <= len(row['text'].strip()) <= 4000):
            raise ValueError('Each criterion needs type inclusion/exclusion and 1-4,000 characters of text.')
        cleaned.append({'type': row['type'], 'text': row['text'].strip()})
    result = {'note': note, 'criteria': cleaned}
    for name, default, limit in (('patient_reference', 'Unidentified case', 80), ('trial_name', 'Entered criteria', 160)):
        text = value.get(name, default)
        if not isinstance(text, str) or len(text) > limit:
            raise ValueError(f'{name} must contain at most {limit} characters.')
        result[name] = text.strip() or default
    return result


def explain_prediction(raw, criterion_type, note):
    prediction = parse_prediction(raw, criterion_type, note)
    sentences = parse_note(note)
    evidence = [{'sentence_id': i, 'text': sentences[i]} for i in prediction['sentence_ids'] if i in sentences]
    label = prediction['label']
    unsupported = label in ('included', 'not included', 'excluded', 'not excluded') and not evidence
    if not prediction['output_valid']:
        explanation = f"The model returned an invalid response: {prediction['error']}. Review the raw response."
    elif label == 'not enough information':
        explanation = 'The record does not establish whether this criterion is met. Additional information is required; absence is not proof.'
    elif label == 'not applicable':
        explanation = 'The model marked this criterion as not applicable. Review that judgment against the criterion and the quoted record.'
    elif unsupported:
        explanation = 'The model assigned a category but cited no supporting sentence. The decision is unsupported and requires review.'
    else:
        ids = ', '.join(str(row['sentence_id']) for row in evidence)
        explanation = f'The model assigned "{label}" and cited sentence IDs {ids}. The exact source statements appear below; citations do not guarantee a correct judgment.'
    return {'prediction': prediction, 'raw_response': raw, 'evidence': evidence,
            'explanation': explanation, 'needs_review': not prediction['output_valid'] or unsupported or label == 'not enough information'}


def aggregate(results):
    labels = [row['prediction']['label'] for row in results]
    if not results or any(not row['prediction']['output_valid'] or (
            row['prediction']['label'] in ('included', 'not included', 'excluded', 'not excluded') and not row['evidence']) for row in results):
        code, text = 'review_required', 'Manual review required'
    elif any(label in ('not included', 'excluded') for label in labels):
        code, text = 'potential_conflict', 'Potential criterion conflict'
    elif 'not enough information' in labels:
        code, text = 'more_information', 'More information needed'
    elif all(label == 'not applicable' for label in labels):
        code, text = 'not_applicable', 'No applicable criteria assessed'
    else:
        code, text = 'provisionally_satisfied', 'Entered criteria provisionally satisfied'
    return {'code': code, 'text': text,
            'rule': 'Invalid or unsupported decisions require review; explicit negative criteria indicate a potential conflict; unresolved criteria require information; otherwise applicable criteria are provisionally satisfied. This is a rule-based aggregation, not a separately validated patient-trial label.'}


def export_html(report):
    esc = lambda value: escape(str(value), quote=True)
    rows = []
    for row in report['results']:
        quotes = ''.join(f'<li>Sentence {esc(e["sentence_id"])}: {esc(e["text"])}</li>' for e in row['evidence'])
        rows.append(f'<section><h2>{esc(row["type"])}: {esc(row["text"])}</h2>'
                    f'<p><strong>{esc(row["prediction"]["label"])}</strong></p><p>{esc(row["explanation"])}</p>'
                    f'<ul>{quotes}</ul><details><summary>Raw model response</summary><pre>{esc(row["raw_response"])}</pre></details></section>')
    provenance = json.dumps(report['provenance'], indent=2, ensure_ascii=False)
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>TrialScreen screening report</title>
<style>body{{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#172827}}section{{border-top:1px solid #ccc;padding:20px 0}}h2{{font-size:18px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}small{{color:#435451}}@media print{{details{{display:block}}}}</style>
<h1>TrialScreen screening report</h1><p>{esc(NOTICE)}</p>
<p>Case: {esc(report['patient_reference'])}<br>Trial: {esc(report['trial_name'])}<br>Created: {esc(report['created_at'])}</p>
<h2>{esc(report['outcome']['text'])}</h2><p>{esc(report['outcome']['rule'])}</p>
<h2>Original numbered record</h2><pre>{esc(report['note'])}</pre>{''.join(rows)}
<h2>Model and inference provenance</h2><pre>{esc(provenance)}</pre></html>'''
