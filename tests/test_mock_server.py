"""End-to-end client test against a tiny in-process mock /v1/systemone server.

No GPU, no model weights: the mock answers every noul question with a fixed
probability. Run: ``pytest tests/test_mock_server.py``.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from intel_net.client import SystemOneClient


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/systemone":
            answers = {qid: {"type": "noul", "noul": 0.8}
                       for qid in payload.get("questions", {})}
            self._send({"model": "mock-d1", "answers": answers,
                        "usage": {"input_tokens": 100},
                        "trunk": {"id": payload.get("trunk_id"), "hit": False}})
        elif self.path == "/tokenize":
            self._send({"tokens": list(range(len(payload.get("content", ""))))})
        else:
            self.send_error(404)


def _serve():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_ask_and_tokenize_over_http():
    server = _serve()
    try:
        client = SystemOneClient(f"http://127.0.0.1:{server.server_port}")
        assert client.tokenize("hello") == [0, 1, 2, 3, 4]
        out = client.ask("some state",
                         [{"question_id": "q1", "type": "noul", "instructions": "Is it up?"}],
                         "trunk:test:1")
        assert out["answers"]["q1"]["noul"] == 0.8
        # choice payload normalization: options dict becomes criteria
        out = client.ask("some state",
                         [{"question_id": "c1", "type": "choice", "instructions": "Which?",
                           "options": ["up", "down"]}],
                         "trunk:test:1")
        assert out["answers"]["c1"]["noul"] == 0.8
    finally:
        server.shutdown()
