"""Local API checks; no full model weights or external services."""
import json
from pathlib import Path
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app import create_server


class AppTests(unittest.TestCase):
    def test_explicit_browser_proxy_origin_is_supported_without_opening_other_origins(self):
        server=create_server(port=0,browser_origin='https://demo.example')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}/api/status'
            with urlopen(Request(url,headers={'Origin':'https://demo.example'}))as response:
                self.assertEqual(response.status,200)
            with self.assertRaises(HTTPError)as error:
                urlopen(Request(url,headers={'Origin':'https://other.example'}))
            self.assertEqual(error.exception.code,403);error.exception.close()
        finally:
            server.shutdown();server.server_close();server.jobs.executor.shutdown();thread.join()

    @classmethod
    def setUpClass(cls):
        cls.server = create_server(port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def test_browser_shell_status_and_missing_experiment_state(self):
        with urlopen(self.url + '/') as response:
            self.assertIn(b'TrialScreen', response.read())
        with urlopen(self.url + '/api/status') as response:
            status = json.load(response)
        self.assertFalse(status['loaded'])
        self.assertIn('model_id', status)
        with self.assertRaises(HTTPError) as error:
            urlopen(self.url + '/training_manifest.json')
        self.assertEqual(error.exception.code, 404)
        error.exception.close()

    def test_invalid_requests_return_actionable_errors_without_loading_model(self):
        request = Request(self.url + '/api/screen', data=b'{}', headers={'Content-Type': 'application/json'})
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 400)
        self.assertIn('patient record', json.load(error.exception)['error'])
        error.exception.close()
        request = Request(self.url + '/api/screen', data=b'{}', headers={
            'Content-Type': 'application/json', 'Origin': 'https://other.example'})
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 403)
        error.exception.close()

    def test_completed_screening_job_exports_exact_evidence(self):
        from screening import aggregate, explain_prediction
        class Service:
            def screen(self, value, progress):
                progress('Checking test fixture.')
                result=explain_prediction('{"label":"included","sentence_ids":[0]}','inclusion',value['note'])
                return {**value,'results':[{**value['criteria'][0],**result}],
                        'outcome':aggregate([result]),'provenance':{'test_only':True},'created_at':'test'}
        server=create_server(port=0,service=Service())
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}'
        try:
            data=json.dumps({'note':'Age 45.','criteria':[{'type':'inclusion','text':'Adult.'}]}).encode()
            with urlopen(Request(url+'/api/screen',data=data,headers={'Content-Type':'application/json'})) as response:
                job_id=json.load(response)['job_id']
            for _ in range(100):
                with urlopen(url+'/api/jobs/'+job_id) as response:state=json.load(response)
                if state['state']!='running':break
                time.sleep(.01)
            self.assertEqual(state['state'],'complete')
            with urlopen(url+'/api/jobs/'+job_id+'/export') as response:html=response.read().decode()
            self.assertIn('Sentence 0: Age 45.',html)
            self.assertIn('not a clinical enrollment decision',html)
        finally:
            server.shutdown();server.server_close();server.jobs.executor.shutdown();thread.join()


if __name__ == '__main__':
    unittest.main()
