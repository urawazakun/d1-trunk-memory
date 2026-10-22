"""Status escapes hostile text, exposes provenance/lag."""
from intel_net import status

import pathlib as _pl
import uuid as _uuid

import pytest as _pytest


@_pytest.fixture
def workdir():
    d = _pl.Path("reports/intel-net/selftest") / _uuid.uuid4().hex[:8]
    d.mkdir(parents=True, exist_ok=True)
    yield d



def test_escaped_and_provenance(workdir):
    snap = {"generation": "g1", "asof": "2026-10-10T12:00:00+00:00", "coverage": 1.0,
            "status": "ok", "input_watermark": "w",
            "signals": [{"question_id": "q1", "label": "yes", "level": 1.0, "delta": 0.5,
                         "spike": False, "coverage": 1.0,
                         "top_chunks": [{"chunk_id": "c1", "p_yes": 0.9,
                                          "text": '<script>alert("x")</script> http://evil?a=1&b=2'}]}]}
    out = status.render(snap, health={"lag": 5})
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "g1" in out and "lag" in out
    p = status.write_status(snap, workdir / "s.html")
    assert p.exists()
