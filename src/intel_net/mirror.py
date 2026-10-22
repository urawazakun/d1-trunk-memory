"""Consistent, verifiable daily SSD mirror without copying a live WAL file."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .contracts import atomic_json, parse_time, utc_iso

JST = timezone(timedelta(hours=9))


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def mirror_status(root: Path, now: datetime, hour: int = 18) -> dict:
    local = parse_time(now).astimezone(JST)
    try:
        receipt = json.loads((root / "latest.json").read_text(encoding="utf-8"))
        completed = parse_time(receipt["completed_at"]).astimezone(JST)
        valid = (root / receipt["path"] / "manifest.json").is_file()
    except (OSError, ValueError, KeyError, TypeError):
        completed, valid = None, False
    today = valid and completed.date() == local.date() and completed.hour >= hour
    # An initial 18:00 backup leaves time for retries. Refresh once during
    # admission drain at 18:58 so later afternoon matrix writes are covered.
    final_ready = bool(today and (completed.hour > 18 or (completed.hour == 18 and completed.minute >= 58)))
    final_due = local.hour > 18 or (local.hour == 18 and local.minute >= 58)
    return {"due": (local.hour >= hour and not today) or (final_due and not final_ready),
            "deadline_missed": local.hour >= 19 and not final_ready, "final_ready": final_ready,
            "last_completed_at": utc_iso(completed) if completed else None,
            "daily_ready": bool(today)}


def backup(db_path: Path, mirror_root: Path, *, now: datetime | None = None, retention_days: int = 90) -> dict:
    """Back up SQLite and all other primary-directory files into one generation.

    Caller is the sole DB/snapshot writer and runs this after publication.
    Old receipts are left intact if copying or checking any file fails.
    """
    now = parse_time(now or datetime.now(timezone.utc))
    primary = db_path.parent.resolve()
    root = mirror_root.resolve()
    if root == primary or root.is_relative_to(primary) or primary.is_relative_to(root):
        raise ValueError("mirror and primary must be separate roots")
    date = now.astimezone(JST).date().isoformat()
    generation = now.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:8]
    final = root / date / generation
    staging = root / date / (generation + ".pending")
    staging.mkdir(parents=True)
    files = []
    try:
        with closing(sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(staging / db_path.name)) as target:
                source.backup(target)
                if target.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise RuntimeError("mirror SQLite integrity check failed")
        for source_file in sorted(primary.rglob("*")):
            if not source_file.is_file():
                continue
            relative = source_file.relative_to(primary)
            if source_file == db_path.resolve() or source_file.name.endswith(("-wal", "-shm", ".lock", ".tmp")):
                continue
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
            if file_digest(source_file) != file_digest(destination):
                raise RuntimeError(f"mirror file changed during backup: {relative}")
        for mirrored in sorted(staging.rglob("*")):
            if mirrored.is_file():
                files.append({"path": mirrored.relative_to(staging).as_posix(), "size": mirrored.stat().st_size,
                              "sha256": file_digest(mirrored)})
        manifest = {"schema_version": 1, "generation": generation, "completed_at": utc_iso(now),
                    "primary": str(primary), "files": files, "retention_days": retention_days}
        atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, final)
        receipt = {"schema_version": 1, "path": final.relative_to(root).as_posix(), "completed_at": utc_iso(now),
                   "manifest_sha256": file_digest(final / "manifest.json")}
        atomic_json(root / "latest.json", receipt)
    except Exception:
        # Keep incomplete generations for diagnosis; never point latest at them.
        raise
    # Delete only checked date directories strictly below our dedicated mirror root.
    cutoff = now.astimezone(JST).date() - timedelta(days=retention_days)
    for day in root.iterdir():
        if not day.is_dir() or len(day.name) != 10:
            continue
        try:
            parsed = datetime.strptime(day.name, "%Y-%m-%d").date()
        except ValueError:
            continue
        resolved = day.resolve()
        if parsed < cutoff and resolved.parent == root and not day.is_symlink():
            shutil.rmtree(resolved)
    return receipt
