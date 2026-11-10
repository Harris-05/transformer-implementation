"""Loopback-only standalone TrialScreen application. Start with Start_TrialScreen.cmd."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import urlsplit
import uuid
import webbrowser

from app_model import ModelService, ROOT
from screening import export_html, validate_request


class Jobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.items = {}
        self.active = None

    def submit(self, function):
        with self.lock:
            if self.active is not None:
                raise RuntimeError('An inference job is already running. Wait for it to finish.')
            job_id = uuid.uuid4().hex
            self.active = job_id
            # Keep only the current and four recent reports, in memory only.
            while len(self.items) >= 5:
                self.items.pop(next(iter(self.items)))
            self.items[job_id] = {'state': 'running', 'message': 'Starting.', 'result': None}

        def work():
            def progress(message):
                with self.lock:
                    self.items[job_id]['message'] = message
            try:
                result = function(progress)
                with self.lock:
                    self.items[job_id].update(state='complete', message='Complete.', result=result)
            except Exception as error:
                with self.lock:
                    self.items[job_id].update(state='failed', message=f'{type(error).__name__}: {error}')
            finally:
                with self.lock:
                    self.active = None
        self.executor.submit(work)
        return job_id

    def get(self, job_id):
        with self.lock:
            # Serialize while locked to avoid races with worker mutation.
            return json.loads(json.dumps(self.items[job_id], allow_nan=False))


def create_server(port=8765, training=None, device='auto', service=None):
    model = service or ModelService(training=training, device=device)
    jobs = Jobs()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Never log patient data or request bodies.

        def send(self, value, status=200, mime='application/json; charset=utf-8'):
            body = value if isinstance(value, bytes) else (json.dumps(value, allow_nan=False).encode() if mime.startswith('application/json') else value.encode())
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers(); self.wfile.write(body)

        def allowed(self):
            hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            origin = self.headers.get('Origin')
            return self.headers.get('Host') in hosts and (origin is None or origin in {f'http://{host}' for host in hosts})

        def do_GET(self):
            if not self.allowed():
                return self.send({'error': 'Local origin required.'}, 403)
            path = urlsplit(self.path).path
            if path == '/api/status':
                return self.send({**model.status(), 'busy': jobs.active is not None})
            if path.startswith('/api/jobs/'):
                parts = path.split('/')
                try:
                    job = jobs.get(parts[3])
                except KeyError:
                    return self.send({'error': 'Report expired or unknown job.'}, 404)
                if len(parts) == 5 and parts[4] == 'export':
                    if job['state'] != 'complete' or 'outcome' not in job['result']:
                        return self.send({'error': 'A completed screening report is required.'}, 400)
                    return self.send(export_html(job['result']), mime='text/html; charset=utf-8')
                return self.send(job)
            files = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                     '/style.css': ('style.css', 'text/css'), '/instrument-sans.ttf': ('instrument-sans.ttf', 'font/ttf'),
                     '/instrument-serif.ttf': ('instrument-serif.ttf', 'font/ttf')}
            if path in files:
                name, mime = files[path]
                file = ROOT / 'web' / name
                if file.is_file():
                    return self.send(file.read_bytes(), mime=mime)
            return self.send({'error': 'Not found.'}, 404)

        def do_POST(self):
            if not self.allowed():
                return self.send({'error': 'Local origin required.'}, 403)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 100000 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('Supply JSON smaller than 100 KB.')
                body = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == '/api/load':
                    function = lambda progress: (progress('Loading model weights.'), model.load(), model.status())[-1]
                elif path in ('/api/screen', '/api/benchmark'):
                    if not isinstance(body, dict):
                        raise ValueError('Expected a JSON object.')
                    modes = body.pop('modes', ['cached', 'uncached']) if path == '/api/benchmark' else None
                    value = validate_request(body)
                    if modes is not None:
                        if not isinstance(modes, list) or not modes or len(set(modes)) != len(modes) or 'cached' not in modes or any(m not in ('cached', 'uncached', 'speculative_2', 'speculative_4') for m in modes):
                            raise ValueError('Choose unique benchmark modes including cached.')
                        function = lambda progress: model.benchmark(value, modes, progress)
                    else:
                        function = lambda progress: model.screen(value, progress)
                else:
                    return self.send({'error': 'Not found.'}, 404)
                job_id = jobs.submit(function)
                return self.send({'job_id': job_id}, 202)
            except (ValueError, TypeError) as error:
                return self.send({'error': str(error)}, 400)
            except RuntimeError as error:
                return self.send({'error': str(error)}, 409)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.jobs = jobs
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--training-output', type=Path)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    server = create_server(args.port, args.training_output, args.device)
    print(f'TrialScreen: http://127.0.0.1:{server.server_port} (local only)', flush=True)
    if not args.no_browser:
        webbrowser.open(f'http://127.0.0.1:{server.server_port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close(); server.jobs.executor.shutdown(wait=False)
