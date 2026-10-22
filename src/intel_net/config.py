"""Explicit paths and operator receipt; unreceipted runs never call inference.

All paths default to repository-relative locations. Override with a JSON
``--config`` file (see ``examples/config.json``) or with environment variables:

=========================== ============================================
field                       environment variable
=========================== ============================================
``db_path``                 ``TRUNK_MEMORY_DB``
``vault_dir``               ``TRUNK_MEMORY_VAULT_DIR``
``chunk_dir``               ``TRUNK_MEMORY_CHUNK_DIR``
``mirror_dir``              ``TRUNK_MEMORY_MIRROR_DIR``
``snapshot_dir``            ``TRUNK_MEMORY_SNAPSHOT_DIR``
``question_bank``           ``TRUNK_MEMORY_QUESTION_BANK``
``assets_path``             ``TRUNK_MEMORY_ASSETS``
``receipt_path``            ``TRUNK_MEMORY_RECEIPT``
``base_url``                ``TRUNK_MEMORY_BASE_URL``
=========================== ============================================
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

from .contracts import local_url

_ENV_VARS = {
    "db_path": "TRUNK_MEMORY_DB",
    "vault_dir": "TRUNK_MEMORY_VAULT_DIR",
    "chunk_dir": "TRUNK_MEMORY_CHUNK_DIR",
    "mirror_dir": "TRUNK_MEMORY_MIRROR_DIR",
    "snapshot_dir": "TRUNK_MEMORY_SNAPSHOT_DIR",
    "question_bank": "TRUNK_MEMORY_QUESTION_BANK",
    "assets_path": "TRUNK_MEMORY_ASSETS",
    "receipt_path": "TRUNK_MEMORY_RECEIPT",
    "base_url": "TRUNK_MEMORY_BASE_URL",
}


@dataclass(frozen=True)
class Config:
    db_path: Path = Path("data/intel-net.db")
    vault_dir: Path = Path("vault-docs")
    chunk_dir: Path = Path("data/chunks")
    mirror_dir: Path = Path("data/mirror")
    snapshot_dir: Path = Path("data/signals")
    question_bank: Path = Path("examples/questions.yaml")
    assets_path: Path | None = Path("examples/assets.json")
    receipt_path: Path = Path("data/cutover-receipt.json")
    base_url: str = "http://127.0.0.1:8101"
    interval_s: float = 60
    ingest_limit: int = 1000
    trunk_limit: int = 20
    batch_size: int = 8
    concurrency: int = 1
    window_h: float = 72
    retention_days: int = 90
    inference_enabled: bool = False
    mirror_hour_jst: int = 18
    quiet_minutes: int = 0

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config, letting ``TRUNK_MEMORY_*`` env vars override defaults."""
        defaults = cls()
        overrides: dict = {}
        path_keys = {f.name for f in fields(cls) if f.type in ("Path", "Path | None")}
        for field_name, env_var in _ENV_VARS.items():
            raw = os.environ.get(env_var)
            if raw is None or not raw.strip():
                continue
            overrides[field_name] = Path(raw) if field_name in path_keys else raw
        return cls(**{**vars(defaults), **overrides})

    def validate(self):
        local_url(self.base_url)
        if not 1 <= self.concurrency <= 4:
            raise ValueError("concurrency must be 1..4")
        if self.interval_s < 1 or min(self.ingest_limit, self.trunk_limit, self.batch_size) < 1:
            raise ValueError("limits and interval must be positive")
        if self.quiet_minutes < 0 or self.quiet_minutes >= 60:
            raise ValueError("quiet_minutes must be 0..59")
        if self.db_path.parent.resolve() != self.snapshot_dir.parent.resolve():
            raise ValueError("snapshot_dir must be inside primary DB directory for complete mirroring")
        return self

    def receipt(self) -> dict | None:
        if not self.inference_enabled:
            return None
        try:
            value = json.loads(self.receipt_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return None
        required = ("model_digest", "template_digest", "binary_digest", "server_generation", "per_slot_context")
        if value.get("status") != "accepted" or value.get("base_url", "").rstrip("/") != self.base_url.rstrip("/"):
            return None
        if any(not value.get(key) for key in required) or value.get("decision_prefix_cache") is not True:
            return None
        if not isinstance(value["per_slot_context"], int) or value["per_slot_context"] < 2048:
            return None
        return value
