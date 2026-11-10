"""Verify returned held-out predictions against frozen inputs and strict scorers."""
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd

from baseline import parse_prediction
from eda import INPUT_COLUMNS,METADATA_COLUMNS,TARGET_COLUMNS,parse_note
from evaluate_adapter import metric_groups,compare_groups
from evaluate_final import build_test_records,validate_saved_predictions
from prepare_model_inputs import load_tokenizer,sha256

ROOT=Path(__file__).resolve().parent


def verify(archive_path=ROOT/'final_test_results.zip',destination=ROOT/'results/final_test'):
    config=json.loads((ROOT/'configs/model_input_config.json').read_text())
    split=json.loads((ROOT/'data/splits/split_manifest.json').read_text())
    training=json.loads((ROOT/'results/finetuning_qlora_v1/training_summary.json').read_text())
    tokenizer,_=load_tokenizer(ROOT,offline=True)
    allowed={r['patient_id']:r['annotation_count']for r in split['splits']['test']}
    frame=pd.read_parquet(ROOT/split['source_file'],columns=INPUT_COLUMNS+METADATA_COLUMNS+TARGET_COLUMNS)
    records=build_test_records(frame.loc[frame.patient_id.isin(allowed)],allowed,tokenizer,config)
    with zipfile.ZipFile(archive_path) as archive:
        manifest=json.loads(archive.read('run_manifest.json'));identity=manifest['identity']
        assert manifest['run_signature']==hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        assert identity['model']==config and identity['training_signature']==training['signature']
        assert identity['adapter_sha256']==training['adapter_sha256']
        assert identity['dataset_sha256']==sha256(ROOT/split['source_file'])==split['source_sha256']
        assert identity['test_patient_counts']==allowed
        assert identity['records_sha256']==hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
        for name,digest in identity['source_sha256'].items():assert sha256(ROOT/name)==digest,name
        modes={}
        for name in ('before','after'):
            rows=[json.loads(line)for line in archive.read(f'predictions_{name}.jsonl').splitlines()]
            assert len(rows)==len(records)==206
            validate_saved_predictions(rows,records,manifest['run_signature'])
            for row in rows:
                parsed=parse_prediction(row['raw_response'],row['criterion_type'],row['note'])
                assert row['prediction']==parsed
                sentences=parse_note(row['note'])
                assert row['predicted_evidence_text']=={str(i):sentences[i]for i in parsed['sentence_ids']if i in sentences}
                assert row['generation_seconds']>0 and 0<row['output_tokens']<=config['max_new_tokens']
            modes[name]=metric_groups(rows)
        metrics=json.loads(archive.read('metrics.json'))
        assert metrics['split']=='held_out_test' and metrics['complete']
        assert metrics['expected_records_per_model']==206 and metrics['evaluated_records']=={'before':206,'after':206}
        assert metrics['groups']==modes
        assert metrics['comparison']==compare_groups(modes['before'],modes['after'])
        destination.mkdir(parents=True,exist_ok=True)
        for member in archive.infolist():
            path=destination/member.filename
            assert path.resolve().is_relative_to(destination.resolve())
            data=archive.read(member)
            if path.exists():assert path.read_bytes()==data
            else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    print('Verified final test: 8 held-out patients, 206 before + 206 after predictions, exact original inputs/gold, source/adapter identities, strict re-parsing, all metrics and comparison.')
    for name,groups in modes.items():
        overall=groups['overall'];print(name,'accuracy',overall['classification']['accuracy'],'macroF1',overall['classification']['macro_f1'],'evidence',overall['evidence']['micro'],'valid',overall['output_valid_rate'])
        print('per_class',overall['classification']['per_class'])
    return metrics


if __name__=='__main__':verify()
