import json
import urllib.request

import pytest

from intel_net.client import SystemOneClient


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_rejects_non_loopback():
    with pytest.raises(ValueError):
        SystemOneClient("https://example.com")


def test_ask_posts_state_and_trunk(monkeypatch):
    seen = {}

    def fake_open(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _Resp({"answers": {"q": {"type": "noul", "noul": 0.7}}})

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    c = SystemOneClient("http://127.0.0.1:8101/")
    out = c.ask("state", [{"question_id": "q", "type": "noul", "instructions": "i?"}],
                "intel:v1:abc")
    assert out["answers"]["q"]["noul"] == 0.7
    assert seen["url"].endswith("/v1/systemone")
    assert seen["body"]["trunk_id"] == "intel:v1:abc" and seen["body"]["state"] == "state"


def test_tokenize_endpoint(monkeypatch):
    def fake_open(req, timeout=None):
        return _Resp({"tokens": [1, 2, 3]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    assert SystemOneClient("http://127.0.0.1:8101").tokenize("hi") == [1, 2, 3]
