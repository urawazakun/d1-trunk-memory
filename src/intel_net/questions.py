"""Standing question bank: static groups plus generated per-asset questions.

Per-asset questions come from a small JSON file (see ``examples/assets.json``)::

    {"assets": [{"ticker": "btc", "name": "Bitcoin"},
                {"ticker": "eth", "name": "Ethereum"}]}

Only tickers with a known display name get questions — unknown names are
never guessed. A legacy ``{"pairs": ["btc_jpy", ...]}`` shape is also
accepted (display name falls back to the upper-cased ticker).
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .contracts import digest

SCHEMA_VERSION = 1
BANK_VERSION = "2026-10-10-v1"


def _norm_type(t):
    if isinstance(t, dict):
        # e.g. {name: noul} -> "noul"
        if len(t) == 1:
            return next(iter(t.values()))
        return str(t)
    return t or "noul"


def _qid_hash(qtype, instructions, options) -> str:
    return digest({"type": qtype, "instructions": instructions, "options": options})


def _load_assets(assets_path) -> tuple[list[str], dict[str, str]]:
    """Return (tickers, names). Unknown tickers are dropped when names are explicit."""
    if not assets_path:
        return [], {}
    try:
        data = json.loads(Path(assets_path).read_text(encoding="utf-8-sig"))
    except OSError:
        return [], {}
    if not isinstance(data, dict):
        return [], {}
    names: dict[str, str] = {}
    tickers: list[str] = []
    if isinstance(data.get("assets"), list):
        for item in data["assets"]:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker", "")).lower()
            name = str(item.get("name", "")).strip()
            if ticker and name and ticker not in tickers:
                tickers.append(ticker)
                names[ticker] = name
        return sorted(tickers), names
    # Legacy shape: {"pairs": ["btc_jpy", ...]} — name falls back to TICKER.
    for p in data.get("pairs", []):
        base = str(p).lower().removesuffix("_jpy")
        if base and base not in tickers:
            tickers.append(base)
    return sorted(tickers), {t: t.upper() for t in tickers}


def _asset_questions(tickers: list[str], names: dict[str, str]) -> list[dict]:
    out = []
    for t in tickers:
        name = names.get(t, t.upper())
        out.append({"question_id": f"asset.{t}.specific", "group": "assets",
                    "subject": t, "type": "noul",
                    "instructions": f"does this text specifically report on {name} ({t.upper()})?",
                    "options": None, "half_life_h": 6.0, "max_age_h": 72.0, "active": True})
        out.append({"question_id": f"asset.{t}.good", "group": "assets",
                    "subject": t, "type": "noul",
                    "instructions": f"does this text suggest a positive near-term outlook for {name} ({t.upper()})?",
                    "options": None, "half_life_h": 6.0, "max_age_h": 72.0, "active": True})
    return out


def _static_questions() -> list[dict]:
    rows = [
        ("market.crypto", "crypto", "market", "noul",
         "does the text contain information about the cryptocurrency market or a specific crypto asset?", None),
        ("market.rates", "rates", "rates", "noul",
         "does the text contain information about interest rates or central bank policy?", None),
        ("market.fx", "fx", "fx", "noul",
         "does the text contain information about foreign exchange rates?", None),
        ("market.equities", "equities", "equities", "noul",
         "does the text contain information about equity markets?", None),
        ("market.sentiment", "sentiment", "market", "noul",
         "does the text express positive sentiment about crypto markets?", None),
        ("world.conflict", "world", "conflict", "noul",
         "does the text describe an armed conflict or geopolitical escalation?", None),
        ("world.disaster", "world", "disaster", "noul",
         "does the text describe a natural disaster?", None),
        ("world.policy", "world", "policy", "noul",
         "does the text describe a government policy or regulatory decision?", None),
        ("world.outage", "world", "outage", "noul",
         "does the text describe a major infrastructure or network outage?", None),
        ("freshness.new", "freshness", "event", "noul",
         "does the text announce a new event or development rather than ongoing background information?", None),
        ("freshness.followup", "freshness", "event", "noul",
         "does the text provide a follow-up or correction to an earlier reported event?", None),
    ]
    return [{"question_id": qid, "group": g, "subject": s, "type": t,
             "instructions": instr, "options": o,
             "half_life_h": 6.0, "max_age_h": 72.0, "active": True}
            for qid, g, s, t, instr, o in rows]


def load_bank(path, assets_path=None, pairs_path=None) -> dict:
    seed_qs: list[dict] = []
    seed_version = BANK_VERSION
    if path and Path(path).exists():
        seed = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        seed_version = str(seed.get("bank_version", BANK_VERSION))
        for q in seed.get("questions", []):
            seed_qs.append(dict(q))
    asset_source = assets_path if assets_path is not None else pairs_path
    tickers, names = _load_assets(asset_source)
    defaults = {q["question_id"]: q for q in _static_questions()}
    base = {}
    for q in _asset_questions(tickers, names):
        base[q["question_id"]] = q
    # YAML is the actual bank source: override known ids field-by-field,
    # and support adding arbitrary new questions / editing arbitrary ones.
    for sq in seed_qs:
        qid = sq.get("question_id")
        if not qid:
            continue
        cur = base.get(qid, dict(defaults.get(qid, {"question_id": qid, "group": sq.get("group", "custom"),
                             "subject": sq.get("subject", qid), "active": True,
                             "half_life_h": 6.0, "max_age_h": 72.0})))
        for f in ("group", "subject", "instructions", "options", "half_life_h",
                  "max_age_h", "active", "signal_policy"):
            if f in sq and sq[f] is not None:
                cur[f] = sq[f]
        if "type" in sq and sq["type"] is not None:
            cur["type"] = _norm_type(sq["type"])
        else:
            cur["type"] = _norm_type(cur.get("type", "noul"))
        base[qid] = cur
    questions = sorted(base.values(), key=lambda q: q["question_id"])
    for q in questions:
        q["type"] = _norm_type(q.get("type", "noul"))
        if q["type"] not in {"noul", "choice"} or not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            raise ValueError(f"invalid question type/instructions: {q['question_id']}")
        if q["type"] == "choice" and (not isinstance(q.get("options"), dict) or not q["options"]):
            raise ValueError(f"choice requires nonempty label descriptions: {q['question_id']}")
        if float(q.get("half_life_h", 6)) <= 0 or float(q.get("max_age_h", 72)) <= 0:
            raise ValueError(f"invalid signal policy: {q['question_id']}")
        q["question_hash"] = _qid_hash(q["type"], q.get("instructions", ""), q.get("options"))
    bank_hash = digest(sorted([(q["question_id"], q["question_hash"]) for q in questions]))
    return {"schema_version": SCHEMA_VERSION, "bank_version": seed_version,
            "questions": questions, "bank_hash": bank_hash}
