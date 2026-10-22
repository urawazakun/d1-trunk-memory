from datetime import datetime, timezone
import json
import sqlite3

import pytest

from intel_net.config import Config
from intel_net.contracts import atomic_json, local_url, parse_time
from intel_net.mirror import backup, mirror_status
from intel_net.runner import WriterLease, _ask_task, priority_window, can_start_request


def test_local_models_only():
    for url in ("https://api.example.org", "http://127.0.0.1@evil.org", "http://127.0.0.1/?token=x"):
        with pytest.raises(ValueError):
            local_url(url)
    assert local_url("http://127.0.0.1:8101/") == "http://127.0.0.1:8101"
    with pytest.raises(ValueError):
        parse_time("2026-10-10T00:00:00")


def test_priority_boundary_and_batch_stop():
    assert priority_window(parse_time("2026-10-10T03:12:59+00:00"))
    assert not priority_window(parse_time("2026-10-10T03:13:00+00:00"))
    class Client:
        def __init__(self): self.calls = 0
        def ask(self, *args):
            self.calls += 1
            return {"answers": {}}
    assert not can_start_request(parse_time("2026-10-10T03:58:00+00:00"))
    times = iter([parse_time("2026-10-10T03:57:59+00:00"),
                  parse_time("2026-10-10T03:57:59+00:00"),
                  parse_time("2026-10-10T03:58:00+00:00"),
                  parse_time("2026-10-10T03:58:00+00:00")])
    client = Client()
    result = _ask_task(client, {"state": "state", "trunk_id": "x", "questions": [{}, {}]},
                       Config(batch_size=1), lambda: next(times))
    assert client.calls == 1 and len(result) == 1


def test_receipt_required(tmp_path):
    path = tmp_path / "receipt.json"
    cfg = Config(receipt_path=path, inference_enabled=True)
    assert cfg.receipt() is None
    atomic_json(path, {"status": "accepted", "base_url": cfg.base_url})
    assert cfg.receipt() is None
    receipt = {"status": "accepted", "base_url": cfg.base_url, "decision_prefix_cache": True,
               "model_digest": "model", "template_digest": "template", "binary_digest": "binary",
               "server_generation": "boot", "per_slot_context": 4096}
    atomic_json(path, receipt)
    assert cfg.receipt() == receipt
    assert Config(receipt_path=path).receipt() is None


def test_mirror_includes_committed_wal_and_content(tmp_path):
    primary = tmp_path / "ssd"
    primary.mkdir()
    db = primary / "intel-net.db"
    source = sqlite3.connect(db)
    source.execute("PRAGMA journal_mode=WAL")
    source.execute("CREATE TABLE answers(value TEXT)")
    source.execute("INSERT INTO answers VALUES ('visible')")
    source.commit()
    atomic_json(primary / "signals" / "latest.json", {"generation": "g1"})
    root = tmp_path / "hdd"
    now = parse_time("2026-10-10T09:30:00+00:00")  # 18:30 JST
    receipt = backup(db, root, now=now)
    mirrored = root / receipt["path"]
    with sqlite3.connect(mirrored / db.name) as con:
        assert con.execute("SELECT value FROM answers").fetchone()[0] == "visible"
    manifest = json.loads((mirrored / "manifest.json").read_text())
    paths = [item["path"] for item in manifest["files"]]
    assert "signals/latest.json" in paths
    assert not any(p.endswith(("-wal", "-shm")) for p in paths)
    assert mirror_status(root, now)["daily_ready"]
    assert mirror_status(root, parse_time("2026-10-10T09:58:00+00:00"))["due"]
    assert mirror_status(root, parse_time("2026-10-10T10:00:00+00:00"))["deadline_missed"]
    assert mirror_status(root, parse_time("2026-10-11T10:00:00+00:00"))["deadline_missed"]
    source.close()


def test_failed_backup_does_not_publish_receipt(tmp_path, monkeypatch):
    primary = tmp_path / "ssd"
    primary.mkdir()
    db = primary / "intel-net.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE x(v)")
    root = tmp_path / "mirror"
    from intel_net import mirror
    monkeypatch.setattr(mirror, "file_digest", lambda p: (_ for _ in ()).throw(OSError("disk failure")))
    with pytest.raises(OSError):
        backup(db, root)
    assert not (root / "latest.json").exists()


def test_single_writer_lock(tmp_path):
    path = tmp_path / "writer.lock"
    with WriterLease(path):
        with pytest.raises(RuntimeError):
            with WriterLease(path):
                pass
    with WriterLease(path):
        pass
