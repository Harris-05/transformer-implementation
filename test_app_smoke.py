import json
from pathlib import Path
import tempfile
import unittest

from app_smoke import save_verification


class AppSmokeVerificationTests(unittest.TestCase):
    def test_failed_check_preserves_evidence_and_fails_the_process(self):
        with tempfile.TemporaryDirectory()as directory:
            output=Path(directory)
            with self.assertRaises(RuntimeError):
                save_verification(output,{'http_complete':True,'tokens_equal':False},{'test_only':True})
            saved=json.loads((output/'verification.json').read_text())
            self.assertFalse(saved['passed'])
            self.assertFalse(saved['checks']['tokens_equal'])

    def test_success_requires_every_check(self):
        with tempfile.TemporaryDirectory()as directory:
            save_verification(Path(directory),{'http_complete':True,'tokens_equal':True},{'test_only':True})
            self.assertTrue(json.loads((Path(directory)/'verification.json').read_text())['passed'])


if __name__=='__main__':unittest.main()
