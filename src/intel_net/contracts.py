"""Small shared, model-independent persistence and identity helpers."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

SCHEMA_VERSION = 1


def parse_time(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def utc_iso(value: str | datetime | None = None) -> str:
    return parse_time(value or datetime.now(timezone.utc)).isoformat()


def digest(value) -> str:
    data = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def atomic_json(path: str | Path, value) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    fd, temp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return target


def local_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost", "::1"} or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("runtime model endpoint must be an HTTP loopback URL")
    return value.rstrip("/")
