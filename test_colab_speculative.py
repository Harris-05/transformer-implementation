"""Exercise adapter discovery retries without Colab or model downloads."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from colab_repair_speculative_cell3 import repair_validation_file


ROOT = Path(__file__).resolve().parent


class ColabSpeculativeTests(unittest.TestCase):
    def test_repair_restores_only_bytes_matching_the_training_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'validation.jsonl'
            original = b'{"note":"unchanged"}\r\n'
            expected = hashlib.sha256(original).hexdigest()
            path.write_bytes(original.replace(b'\r\n', b'\n'))
            self.assertTrue(repair_validation_file(path, expected))
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(repair_validation_file(path, expected))
            changed_content = b'{"note":"different"}\n'
            path.write_bytes(changed_content)
            with self.assertRaises(ValueError):
                repair_validation_file(path, expected)
            self.assertEqual(path.read_bytes(), changed_content)

    def test_embedded_restore_preserves_exact_validation_file_bytes(self):
        notebook = json.loads((ROOT / 'colab_speculative.ipynb').read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'SOURCES =' in ''.join(c['source']))
        with tempfile.TemporaryDirectory() as directory:
            source = source.replace('/content/trialgpt_speculative', Path(directory).as_posix())
            exec(source, {})
            restored = Path(directory) / 'data/model_ready/validation.jsonl'
            expected = json.loads((ROOT / 'results/finetuning_qlora_v1/training_manifest.json').read_text())
            self.assertEqual(hashlib.sha256(restored.read_bytes()).hexdigest(),
                             expected['identity']['data_sha256']['validation'])
            self.assertEqual(restored.read_bytes(), (ROOT / 'data/model_ready/validation.jsonl').read_bytes())

    def runtime(self, directory):
        root = Path(directory)
        (root / 'drive/MyDrive').mkdir(parents=True)
        colab = types.ModuleType('google.colab')
        colab.drive = types.SimpleNamespace(mount=lambda _: self.fail('Drive is already mounted.'))
        return root, {'google.colab': colab}

    def test_missing_adapter_then_zip_upload_can_retry_same_cell(self):
        notebook = json.loads((ROOT / 'colab_speculative.ipynb').read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'import tempfile, zipfile' in ''.join(c['source']))
        with tempfile.TemporaryDirectory() as directory:
            root, modules = self.runtime(directory)
            source = source.replace('/content', root.as_posix())
            calls = []
            namespace = {'Path': Path, 'json': json, 'sys': sys, 'run_checked': calls.append}
            with patch.dict(sys.modules, modules):
                with self.assertRaises(FileNotFoundError):
                    exec(source, namespace)
                self.assertIsNone(namespace['TRAINING'])
                with zipfile.ZipFile(root / 'finetuning_results.zip', 'w') as archive:
                    archive.writestr('training_summary.json', json.dumps({
                        'adapter_sha256': {'adapter_model.safetensors': 'test'}}))
                    archive.writestr('training_manifest.json', '{}')
                    archive.writestr('adapter/adapter_model.safetensors', b'fixture')
                exec(source, namespace)
            self.assertTrue((namespace['TRAINING'] / 'adapter/adapter_model.safetensors').is_file())
            self.assertEqual(len(calls), 1)
            self.assertIn('--prepare-only', calls[0])

    def test_drive_backup_recovers_when_training_global_is_none(self):
        source = (ROOT / 'colab_save_adapter_to_drive.py').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory() as directory:
            root, modules = self.runtime(directory)
            weights = b'fixture'
            digest = hashlib.sha256(weights).hexdigest()
            (root / 'training_summary.json').write_text(json.dumps({
                'complete': True, 'signature': 'test',
                'adapter_sha256': {'adapter_model.safetensors': digest}}))
            (root / 'training_manifest.json').write_text(json.dumps({'signature': 'test'}))
            (root / 'adapter_model.safetensors').write_bytes(weights)
            namespace = {'TRAINING': None}
            with patch.dict(sys.modules, modules):
                exec(source.replace('/content', root.as_posix()), namespace)
            self.assertEqual((namespace['DESTINATION'] / 'adapter/adapter_model.safetensors').read_bytes(), weights)


if __name__ == '__main__':
    unittest.main()
