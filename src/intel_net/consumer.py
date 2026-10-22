"""Bounded read-only snapshot reader (Muse B, REVIEW-B compliant)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .contracts import digest, parse_time

SCHEMA_VERSION = 1


class IntelNetUnavailable(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _resolve_inside(root: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel:
        raise IntelNetUnavailable("malformed manifest path")
    p = Path(rel)
    if p.is_absolute():
        raise IntelNetUnavailable("manifest path escapes root")
    full = (root / p).resolve()
    if full != root.resolve() and root.resolve() not in full.parents:
        raise IntelNetUnavailable("manifest path escapes root")
    return full


def _load(root: Path) -> dict:
    latest = root / "latest.json"
    try:
        man = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise IntelNetUnavailable(f"unreadable manifest: {e}")
    if not isinstance(man, dict):
        raise IntelNetUnavailable("malformed manifest")
    # mirror-root resolver: daily mirror receipt points at a generation dir
    # containing signals/latest.json (documented resolver).
    try:
        cand = _resolve_inside(root, man.get("path", ""))
    except IntelNetUnavailable:
        raise
    if (cand.parent / "manifest.json").is_file() or (cand / "manifest.json").is_file():
        nested = cand / "signals" / "latest.json"
        if nested.is_file():
            return _load(cand / "signals")
        nested2 = root / man.get("path", "") / "signals" / "latest.json"
        if nested2.is_file():
            return _load(nested2.parent)
    snap_path = _resolve_inside(root, man.get("path", ""))
    try:
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise IntelNetUnavailable(f"unreadable snapshot: {e}")
    if not isinstance(snap, dict):
        raise IntelNetUnavailable("malformed snapshot")
    gen = man.get("generation")
    if not isinstance(gen, str) or snap.get("generation") != gen:
        raise IntelNetUnavailable("generation mismatch")
    want = man.get("checksum")
    if not isinstance(want, str):
        raise IntelNetUnavailable("malformed manifest checksum")
    try:
        if digest(snap) != want:
            raise IntelNetUnavailable("checksum mismatch")
    except ValueError as e:
        raise IntelNetUnavailable(f"checksum failure: {e}")
    return snap


def read_snapshot(root, *, asof, required_questions=None, max_age_h=2, min_coverage=.95,
                  mirror_root=None) -> dict:
    asof_dt = parse_time(asof)
    last_err = None
    for cand in [Path(root)] + ([Path(mirror_root)] if mirror_root else []):
        try:
            return _check(_load(cand), asof_dt, required_questions, max_age_h, min_coverage)
        except IntelNetUnavailable as e:
            last_err = e
            continue
    raise last_err


def _check(snap: dict, asof_dt, required_questions, max_age_h, min_coverage) -> dict:
    if snap.get("schema_version") != SCHEMA_VERSION:
        raise IntelNetUnavailable("bad schema/version")
    try:
        sasof = parse_time(snap["asof"])
        scomp = parse_time(snap["computed_at"])
    except (KeyError, ValueError, TypeError):
        raise IntelNetUnavailable("bad timestamps")
    if sasof > asof_dt or scomp > asof_dt:
        raise IntelNetUnavailable("future snapshot (no-lookahead)")
    try:
        age_h = (asof_dt - sasof).total_seconds() / 3600.0
    except TypeError:
        raise IntelNetUnavailable("bad timestamps")
    if not math.isfinite(age_h) or age_h > max_age_h:
        raise IntelNetUnavailable(f"stale snapshot age_h={age_h:.2f}")
    if snap.get("status") not in ("ok", "empty"):
        raise IntelNetUnavailable("snapshot status not servable")
    sigs = snap.get("signals")
    if not isinstance(sigs, list):
        raise IntelNetUnavailable("incomplete signals")
    cov = snap.get("coverage", 0)
    if (not isinstance(cov, (int, float)) or isinstance(cov, bool)
            or not math.isfinite(cov) or not 0.0 <= cov <= 1.0):
        raise IntelNetUnavailable("malformed coverage")
    if cov < min_coverage:
        raise IntelNetUnavailable(f"insufficient coverage {cov}")
    # healthy-complete-empty: servable HOLD; incomplete-empty rejected above
    if not sigs and snap.get("status") == "empty" and cov == 1.0:
        return snap
    if required_questions:
        for qid, qhash in required_questions:
            match = [s for s in sigs if isinstance(s, dict)
                     and s.get("question_id") == qid and s.get("question_hash") == qhash]
            if not match:
                raise IntelNetUnavailable(f"missing required question {qid}")
            for s in match:
                sc = s.get("coverage")
                if (not isinstance(sc, (int, float)) or isinstance(sc, bool)
                        or not math.isfinite(sc) or sc != 1.0):
                    raise IntelNetUnavailable(f"required question {qid} coverage != 1")
                if s.get("status") not in ("ok", "empty"):
                    raise IntelNetUnavailable(f"required question {qid} status invalid")
    return snap
