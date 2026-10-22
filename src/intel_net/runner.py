"""One writer: ingest, trunk-major inference, signals, status and daily mirror."""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Config
from .contracts import atomic_json, digest, parse_time, utc_iso
from .mirror import backup, mirror_status

LOG = logging.getLogger("intel_net")
JST = timezone(timedelta(hours=9))


def priority_window(now: datetime, quiet_minutes: int = 12) -> bool:
    """Operator quiet window: JST minutes 0..``quiet_minutes`` each hour.

    Reserves time for higher-priority jobs sharing the GPU. ``0`` disables it.
    """
    return quiet_minutes > 0 and parse_time(now).astimezone(JST).minute <= quiet_minutes


def can_start_request(now: datetime, timeout_s=120, quiet_minutes: int = 12) -> bool:
    stamp = parse_time(now).astimezone(JST)
    remaining = 3600 - (stamp.minute * 60 + stamp.second + stamp.microsecond / 1e6)
    return not priority_window(stamp, quiet_minutes) and remaining > timeout_s


class WriterLease:
    """OS-held lock; crashes release it, avoiding stale PID lock recovery."""
    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        self.stream.write(b"0")
        self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise RuntimeError("intel-net writer is already running") from error
        return self

    def __exit__(self, *args):
        if self.stream:
            self.stream.close()


def _baselines(root: Path, asof, revision, bank_hash):
    try:
        values = json.loads((root / "hourly-history.json").read_text(encoding="utf-8"))
        history = [s for s in values if s.get("inference_revision") == revision
                   and s.get("bank_hash") == bank_hash and parse_time(s["asof"]) <= asof - timedelta(hours=1)]
        history.sort(key=lambda s: parse_time(s["asof"]))
        return (history[-1] if history else None), history[-24:], values
    except (OSError, ValueError, TypeError, KeyError):
        return None, [], []


def _save_hourly(root, snapshot, values):
    # One sample per UTC hour, first complete generation after the quiet window.
    # This avoids minute deltas masquerading as hourly spike history.
    stamp = parse_time(snapshot["asof"])
    if stamp.minute <= 12 or snapshot.get("coverage", 0) < .95 or snapshot.get("status") != "ok":
        return
    key = stamp.replace(minute=0, second=0, microsecond=0)
    if any(parse_time(s["asof"]).replace(minute=0, second=0, microsecond=0) == key
           and s.get("bank_hash") == snapshot.get("bank_hash")
           and s.get("inference_revision") == snapshot.get("inference_revision") for s in values):
        return
    cutoff = stamp - timedelta(hours=48)
    kept = [s for s in values if parse_time(s["asof"]) >= cutoff]
    atomic_json(root / "hourly-history.json", (kept + [snapshot])[-48:])


def _ask_task(client, task: dict, config: Config, now_fn, per_slot_context=None, quiet_minutes: int = 12):
    """Trunk-major batches; no DB access from inference worker threads."""
    responses = []
    if not can_start_request(now_fn(), quiet_minutes=quiet_minutes):
        return responses
    if per_slot_context:
        if task.get("token_count_basis") != "tokenizer":
            return [(task, {"answers": {}, "error": "state requires tokenizer validation"}, utc_iso(now_fn()), 0)]
        state_tokens = len(client.tokenize(task["state"]))
        if state_tokens > 1000:
            return [(task, {"answers": {}, "error": "state exceeds validated chunk limit"}, utc_iso(now_fn()), 0)]
    for start in range(0, len(task["questions"]), config.batch_size):
        if not can_start_request(now_fn(), quiet_minutes=quiet_minutes):
            break
        subset = task["questions"][start:start + config.batch_size]
        batch = {**task, "questions": subset}
        started = time.monotonic()
        try:
            if per_slot_context:
                longest = max(len(client.tokenize(json.dumps(q, ensure_ascii=False))) for q in subset)
                if state_tokens + longest + 512 > per_slot_context:
                    raise ValueError("question plus state exceeds slot headroom; revise/split bank question")
                if not can_start_request(now_fn(), quiet_minutes=quiet_minutes):
                    break
            response = client.ask(task["state"], subset, task["trunk_id"])
        except Exception as error:
            response = {"answers": {}, "error": f"{type(error).__name__}: {error}"}
        responses.append((batch, response, utc_iso(now_fn()), (time.monotonic() - started) * 1000))
        if response.get("error"):
            break  # defer retry to a subsequent bounded tick
    return responses


def tick(config: Config, *, now_fn=None) -> dict:
    from . import client, ingest, matrix, questions, registry, signals, status
    now_fn = now_fn or (lambda: datetime.now(timezone.utc))
    now = parse_time(now_fn())
    config.validate()
    receipt = config.receipt()
    qm = config.quiet_minutes
    health = {"asof": utc_iso(now), "inference": "enabled" if receipt else "awaiting-cutover-receipt",
              "priority_paused": priority_window(now, qm), "drain_paused": not can_start_request(now, quiet_minutes=qm),
              "receipt": receipt,
              "mirror": mirror_status(config.mirror_dir, now, config.mirror_hour_jst)}
    bank = questions.load_bank(config.question_bank, assets_path=config.assets_path)
    con = registry.connect(config.db_path)
    try:
        registry.register_questions(con, bank)
        identity = receipt or {"model_digest": "unreceipted-d1", "template_digest": "unreceipted-template"}
        # Offline byte-bounded states cannot be inferred until revalidated under
        # receipted model/template identity (different trunk namespace).
        model = client.SystemOneClient(config.base_url) if receipt else None
        health["ingest"] = ingest.ingest(con, config.vault_dir, config.chunk_dir, asof=now, bank=bank,
            model_digest=identity["model_digest"], template_digest=identity["template_digest"],
            tokenize=model.tokenize if model and not priority_window(now, qm) else None,
            window_h=config.window_h, limit=config.ingest_limit)
        tasks = matrix.pending(con, bank, asof=now, limit=config.trunk_limit,
            model_digest=identity["model_digest"], template_digest=identity["template_digest"], window_h=config.window_h)
        health["pending_trunks_in_page"] = len(tasks)
        health["answered_batches"] = 0
        if model and can_start_request(now, quiet_minutes=qm):
            current_tasks = [t for t in tasks if t["model_digest"] == receipt["model_digest"]
                             and t["template_digest"] == receipt["template_digest"]]
            with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
                futures = [executor.submit(_ask_task, model, task, config, now_fn, receipt["per_slot_context"], qm)
                           for task in current_tasks]
                for future in as_completed(futures):
                    for task, response, completed, wall_ms in future.result():
                        matrix.record(con, task, response, available_at=completed, wall_ms=wall_ms)
                        health["answered_batches"] += 1
        # Publication uses the time after answers finished, never the earlier
        # ingest time; consumers can still query historical point-in-time rows.
        asof = parse_time(now_fn())
        rows = matrix.feature_rows(con, bank, asof=asof, window_h=config.window_h, require_available=True,
            model_digest=identity["model_digest"], template_digest=identity["template_digest"])
        if receipt:
            rows = [r for r in rows if r.get("model_digest") == receipt["model_digest"]
                    and r.get("template_digest") == receipt["template_digest"]]
        revision = digest({"model": identity["model_digest"], "template": identity["template_digest"]})
        previous, history, all_history = _baselines(config.snapshot_dir, asof, revision, bank.get("bank_hash"))
        snapshot = signals.compute(rows, bank, asof=asof, previous=previous, history=history,
            inference_revision=revision, health={"ingest_complete": not health["ingest"].get("incomplete", True)})
        if not receipt:
            snapshot["status"] = "awaiting-inference"
        if health["ingest"].get("incomplete"):
            snapshot["status"] = "ingest-incomplete"
        snapshot["baseline_age_h"] = (asof - parse_time(previous["asof"])).total_seconds() / 3600 if previous else None
        snapshot["health"] = health
        path = signals.publish(snapshot, config.snapshot_dir)
        snapshot["generation"] = path.parent.name
        _save_hourly(config.snapshot_dir, snapshot, all_history)
        health["snapshot"] = str(path)
        status.write_status(snapshot, config.db_path.parent / "status.html", health=health)
        atomic_json(config.db_path.parent / "health.json", health)
        if health["mirror"]["due"]:
            try:
                health["mirror"]["receipt"] = backup(config.db_path, config.mirror_dir, now=asof,
                    retention_days=config.retention_days)
                health["mirror"].update(mirror_status(config.mirror_dir, asof, config.mirror_hour_jst))
            except Exception as error:
                health["mirror"]["error"] = f"{type(error).__name__}: {error}"
                LOG.exception("daily mirror failed; next tick retries")
            atomic_json(config.db_path.parent / "health.json", health)
            status.write_status(snapshot, config.db_path.parent / "status.html", health=health)
        return health
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--inference-enabled", action="store_true", help="also requires accepted operator receipt")
    parser.add_argument("--config", type=Path, help="JSON configuration paths/policies")
    parser.add_argument("--stop-file", type=Path, default=Path("data/STOP"))
    args = parser.parse_args()
    config = Config.from_env()
    if args.config:
        values = json.loads(args.config.read_text(encoding="utf-8-sig"))
        path_keys = {key for key, value in vars(config).items() if isinstance(value, Path)}
        config = replace(config, **{k: Path(v) if k in path_keys else v for k, v in values.items()})
    if args.inference_enabled:
        config = replace(config, inference_enabled=True)
    config.validate()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with WriterLease(config.db_path.parent / "writer.lock"):
        while not args.stop_file.exists():
            try:
                result = tick(config)
                LOG.info("tick %s", json.dumps(result, ensure_ascii=False, default=str))
            except Exception:
                LOG.exception("tick failed")
                if not args.loop:
                    raise
            if not args.loop:
                break
            deadline = time.monotonic() + config.interval_s
            while time.monotonic() < deadline and not args.stop_file.exists():
                time.sleep(min(1, max(0, deadline - time.monotonic())))
