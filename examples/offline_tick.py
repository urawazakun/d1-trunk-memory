"""Offline demo: ingest two synthetic docs and build a snapshot.

No GPU, no model, no network. Inference stays in "awaiting-cutover-receipt"
state; the tick still exercises ingest -> trunk registry -> question bank ->
answer matrix -> signals -> status page.

Run from the repo root::

    python examples/offline_tick.py
"""
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from intel_net.config import Config  # noqa: E402
from intel_net.contracts import atomic_json  # noqa: E402
from intel_net.runner import tick  # noqa: E402

DOCS = [
    ("demo-1", "Demo notice: Bitcoin (BTC) deposits resume",
     "The exchange announced today that deposits for Bitcoin (BTC) have resumed after scheduled maintenance."),
    ("demo-2", "Demo notice: rainfall observation",
     "The city weather office recorded 18 millimeters of rainfall today; no transport disruption was reported."),
]


def _stage_docs(vault: Path, now: str) -> None:
    day = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    for doc_id, title, summary in DOCS:
        atomic_json(vault / day / "agency" / f"{doc_id}.json", {
            "id": doc_id, "source": "agency", "title": title, "summary": summary,
            "fetched_at": now, "published_at": now,
            "url": f"https://example.invalid/{doc_id}"})


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    vault = ROOT / "data" / "demo-vault"
    _stage_docs(vault, now)
    cfg = replace(
        Config(),
        db_path=ROOT / "data" / "demo.db",
        snapshot_dir=ROOT / "data" / "demo-signals",
        vault_dir=vault,
        chunk_dir=ROOT / "data" / "demo-chunks",
        mirror_dir=ROOT / "data" / "demo-mirror",
        question_bank=ROOT / "examples" / "questions.yaml",
        assets_path=ROOT / "examples" / "assets.json",
        receipt_path=ROOT / "data" / "demo-receipt.json",
    )
    health = tick(cfg)
    print(json.dumps(health, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
