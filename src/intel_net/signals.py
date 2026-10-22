"""Deterministic signals aggregation (Muse B, REVIEW-B compliant)."""
from __future__ import annotations

import math
from datetime import timezone
from pathlib import Path

from .contracts import atomic_json, digest, parse_time, utc_iso

SCHEMA_VERSION = 1
SPIKE_POLICY_VERSION = "spike-v1-provisional"
COVERAGE_SPIKE_MIN = 0.95
HISTORY_MIN = 6
DOCS_MIN_FOR_SPIKE = 3
LN2 = math.log(2.0)


def _weight(age_h: float, half_life_h: float) -> float:
    return math.exp(-LN2 * max(0.0, age_h) / half_life_h) if half_life_h > 0 else 0.0


def _labels_for(q: dict) -> list[str]:
    if (q.get("type") or "yesno") == "choice":
        opts = q.get("options") or {}
        if isinstance(opts, dict):
            return [str(k) for k in opts.keys()]
        return [str(o) for o in opts]
    return ["yes"]


def _prob_of(row: dict, label: str):
    if label == "yes":
        return row.get("p_yes")
    probs = row.get("probabilities")
    if isinstance(probs, dict):
        return probs.get(label)
    return None


def _eligible(row: dict, q: dict, asof, max_age_h: float, half_life_h: float):
    """Valid answered cell, or None. Choice rows: p_yes null is NOT missing."""
    if row.get("status") != "ok":
        return None
    if row.get("question_hash") != q.get("question_hash"):
        return None
    try:
        pub = parse_time(row["published_at"]) if row.get("published_at") else None
        fet = parse_time(row["fetched_at"]) if row.get("fetched_at") else None
        avail = parse_time(row["available_at"]) if row.get("available_at") else None
    except (ValueError, KeyError, TypeError):
        return None
    if avail is None or avail > asof:
        return None
    if fet is None or fet > asof:
        return None
    if pub is not None and pub > asof:
        return None
    t = pub or fet
    age_h = (asof - t).total_seconds() / 3600.0
    if age_h > max_age_h:
        return None
    # baseline identity: row carrying a divergent bank/model revision is stale
    for key, want in (("bank_version", q.get("_bank_version")),
                      ("bank_hash", q.get("_bank_hash"))):
        if want is not None and row.get(key) is not None and row.get(key) != want:
            return None
    return {"age_h": age_h, "weight": _weight(age_h, half_life_h),
            "available_at": avail}


def _valid_prob(p) -> bool:
    return isinstance(p, (int, float)) and math.isfinite(p) and 0.0 <= p <= 1.0


def compute(rows, bank, *, asof, previous=None, history=None,
            inference_revision="", health=None) -> dict:
    asof_dt = parse_time(asof)
    bank_questions = bank.get("questions") or []
    by_q: dict[str, list] = {}
    for r in rows or []:
        by_q.setdefault(r.get("question_id"), []).append(r)
    # baseline gate for previous/history: version+hash+model must match
    def _baseline_ok(snap: dict) -> bool:
        if not isinstance(snap, dict):
            return False
        if snap.get("bank_version") != bank.get("bank_version"):
            return False
        if snap.get("bank_hash") != bank.get("bank_hash", ""):
            return False
        if inference_revision and snap.get("inference_revision") != inference_revision:
            return False
        try:
            if parse_time(snap["asof"]) > asof_dt:
                return False
        except (KeyError, ValueError, TypeError):
            return False
        return True
    prev_map: dict[tuple, dict] = {}
    prev_asofs: set[str] = set()
    if _baseline_ok(previous):
        try:
            prev_asofs.add(parse_time(previous["asof"]).isoformat())
        except (KeyError, ValueError, TypeError):
            pass
        for s in previous.get("signals") or []:
            prev_map[(s.get("question_id"), s.get("label"))] = s
    hist_map: dict[tuple, list] = {}
    if history:
        for snap in history:
            if not _baseline_ok(snap):
                continue  # minute-to-minute / mismatched revisions ignored
            try:
                hiso = parse_time(snap["asof"]).isoformat()
            except (KeyError, ValueError, TypeError):
                continue
            for s in (snap.get("signals") or []):
                if s.get("delta") is None:
                    continue
                hist_map.setdefault((s.get("question_id"), s.get("label"), hiso), []).append(s["delta"])
    signals = []
    total_obs = total_exp = 0
    watermark = None
    ingest_complete = bool((health or {}).get("ingest_complete"))
    for q in bank_questions:
        if not q.get("active", True):
            continue
        q = dict(q)
        q["_bank_version"] = bank.get("bank_version")
        q["_bank_hash"] = bank.get("bank_hash", "")
        qid = q["question_id"]
        qhash = q.get("question_hash", "")
        half = float(q.get("half_life_h", 6.0))
        max_age = float(q.get("max_age_h", 72.0))
        qrows = by_q.get(qid, [])
        expected = len(qrows)
        # corrections: newest visible group revision wins, no future docs
        groups: dict[str, list] = {}
        for r in qrows:
            try:
                fet = parse_time(r["fetched_at"]) if r.get("fetched_at") else None
            except (ValueError, TypeError):
                fet = None
            if fet is not None and fet > asof_dt:
                continue  # future docs excluded before grouping
            groups.setdefault(str(r.get("correction_group") or r.get("chunk_id") or r.get("content_sha256") or "?"), []).append(r)
        newest: list[dict] = []
        for g in groups.values():
            def _fk(r):
                try:
                    return parse_time(r["fetched_at"]) if r.get("fetched_at") else asof_dt
                except (ValueError, TypeError):
                    return asof_dt
            g.sort(key=_fk)
            latest_fetch = _fk(g[-1])
            newest.extend(r for r in g if _fk(r) == latest_fetch)
        # exact-chunk-state content dedup: keep newest fetched per content hash
        dedup: dict[str, dict] = {}
        for r in newest:
            key = r.get("state_sha256") or r.get("state_path") or r.get("chunk_id")
            prev = dedup.get(key)
            if prev is None:
                dedup[key] = r
            else:
                try:
                    if parse_time(r.get("fetched_at")) > parse_time(prev.get("fetched_at")):
                        dedup[key] = r
                except (ValueError, TypeError):
                    pass
        for label in _labels_for(q):
            valid: list[tuple[dict, dict, float]] = []
            for r in dedup.values():
                el = _eligible(r, q, asof_dt, max_age, half)
                if el is None:
                    continue
                p = _prob_of(r, label)
                if not _valid_prob(p):
                    continue
                valid.append((r, el, float(p)))
            # coverage independent of max-per-document aggregation
            observed = len(valid)
            cov = (observed / expected) if expected else (1.0 if ingest_complete else 0.0)
            # max per document AFTER scoring
            doc_best: dict[str, dict] = {}
            for (r, el, p) in valid:
                score = p * el["weight"]
                # Provenance may list several syndicated source IDs. They are
                # one document contribution, not one vote per source copy.
                d = r.get("content_sha256") or (r.get("doc_ids") or [r.get("chunk_id")])[0]
                cur = doc_best.get(d)
                if cur is None or score > cur["score"]:
                    doc_best[d] = {"row": r, "el": el, "score": score, "p": p}
            contribs = sorted(doc_best.values(), key=lambda c: c["score"], reverse=True)
            level = sum(c["score"] for c in contribs)
            for c in contribs:
                try:
                    ft = parse_time(c["row"].get("fetched_at"))
                    if watermark is None or ft > watermark:
                        watermark = ft
                except (ValueError, TypeError):
                    pass
            total_obs += observed
            total_exp += expected
            prev = prev_map.get((qid, label))
            plevel = prev.get("level") if prev else None
            if not isinstance(plevel, (int, float)) or not math.isfinite(plevel):
                plevel = None
            complete = observed == expected and (expected > 0 or ingest_complete)
            delta = (level - plevel) if plevel is not None and complete else None
            # spike: unknown when coverage low / history thin; never duplicate prev delta in history
            spike = None
            if delta is not None and delta > 0 and cov >= COVERAGE_SPIKE_MIN:
                prior: list[float] = []
                if prev is not None and isinstance(prev.get("delta"), (int, float)):
                    try:
                        p_asof = parse_time(previous["asof"]).isoformat()
                    except (KeyError, ValueError, TypeError):
                        p_asof = ""
                    if p_asof not in {k[2] for k in hist_map if k[:2] == (qid, label)}:
                        prior.append(float(prev["delta"]))
                for (hq, hl, _hiso), ds in hist_map.items():
                    if (hq, hl) == (qid, label):
                        prior.extend(float(d) for d in ds if isinstance(d, (int, float)))
                prior = [d for d in prior if math.isfinite(d)][-24:]
                docs = len(contribs)
                if len(prior) >= HISTORY_MIN and docs >= DOCS_MIN_FOR_SPIKE:
                    mean = sum(prior) / len(prior)
                    var = sum((d - mean) ** 2 for d in prior) / len(prior)
                    sd = math.sqrt(var)
                    spike = bool(delta >= max(1.0, 3 * sd))
                else:
                    spike = None
            elif delta is not None and delta <= 0:
                spike = False
            top = [{"chunk_id": c["row"].get("chunk_id"), "doc_ids": c["row"].get("doc_ids"),
                    "p_yes": c["row"].get("p_yes"), "probabilities": c["row"].get("probabilities"),
                    "weight": c["el"]["weight"], "score": c["score"],
                    "age_h": c["el"]["age_h"],
                    "published_at": c["row"].get("published_at"),
                    "fetched_at": c["row"].get("fetched_at"),
                    "available_at": c["row"].get("available_at"),
                    "state_path": c["row"].get("state_path"),
                    "source_urls": c["row"].get("source_urls", []),
                    "sources": c["row"].get("sources", []),
                    "text": c["row"].get("text", "")} for c in contribs[:8]]
            status = "ok" if (expected and observed == expected) or (not expected and ingest_complete) else "incomplete"
            signals.append({"question_id": qid, "question_hash": qhash,
                            "group": q.get("group"), "subject": q.get("subject"),
                            "label": label, "level": level, "previous_level": plevel,
                            "delta": delta, "spike": spike,
                            "spike_policy_version": SPIKE_POLICY_VERSION,
                            "observed_count": observed, "expected_count": expected,
                            "coverage": cov, "freshness_h": None, "top_chunks": top,
                            "status": status, "error": None})
    if total_exp:
        coverage = total_obs / total_exp
        status_all = "ok" if total_obs == total_exp else "incomplete"
    else:
        coverage = 1.0 if ingest_complete else 0.0
        status_all = "empty" if ingest_complete else "incomplete"
    return {"schema_version": SCHEMA_VERSION, "asof": utc_iso(asof_dt),
            "computed_at": utc_iso(), "bank_version": bank.get("bank_version"),
            "bank_hash": bank.get("bank_hash", ""), "inference_revision": inference_revision,
            "coverage": coverage, "status": status_all,
            "input_watermark": utc_iso(watermark) if watermark else None,
            "signals": signals}


def publish(snapshot: dict, root) -> Path:
    from datetime import datetime
    root = Path(root)
    gen = "g" + parse_time(snapshot["asof"]).strftime("%Y%m%dT%H%M%S") + "-" + digest(snapshot)[:8]
    snap = dict(snapshot)
    snap["generation"] = gen
    gdir = root / gen
    gdir.mkdir(parents=True, exist_ok=True)
    atomic_json(gdir / "snapshot.json", snap)
    atomic_json(root / "latest.json", {"generation": gen, "path": f"{gen}/snapshot.json",
                                       "asof": snap["asof"], "checksum": digest(snap)})
    return gdir / "snapshot.json"
