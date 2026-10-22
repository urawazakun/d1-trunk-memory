"""Durable answer matrix: trunk-major pending, validated record, PIT rows."""
from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path

from .contracts import digest, parse_time, utc_iso

INFERENCE_REVISION = "r1"


def _active_questions(con, bank):
    """Follow the bank's exact active hashes: only bank-active questions whose
    registered active hash equals the bank hash are eligible."""
    reg = {(r["question_id"], r["question_hash"]): r["active"] for r in
           con.execute("SELECT question_id,question_hash,active FROM question_versions")}
    out = []
    for q in bank["questions"]:
        if not q.get("active", True):
            continue
        if reg.get((q["question_id"], q["question_hash"]), 0) != 1:
            continue
        out.append(dict(q))
    return out


def _validate_answer(q: dict, ans) -> tuple[str, float | None, dict | None, str | None]:
    """Return (status, p_yes, probabilities, error)."""
    if not isinstance(ans, dict):
        return "missing", None, None, "absent"
    # type must match expected
    if "type" in ans and ans.get("type") != q.get("type"):
        return "invalid", None, None, "type-mismatch"
    t = q.get("type", "noul")
    try:
        if t == "noul":
            if "noul" not in ans and "p_yes" not in ans:
                return "missing", None, None, "absent"
            value = ans.get("noul", ans.get("p_yes"))
            if isinstance(value, bool):
                return "invalid", None, None, "boolean-not-probability"
            p = float(value)
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                return "invalid", None, None, "nonfinite-or-range"
            return "ok", p, {"yes": p, "no": 1.0 - p}, None
        probs = ans.get("probabilities")
        if probs is None:
            return "missing", None, None, "absent"
        if not isinstance(probs, dict) or not probs:
            return "invalid", None, None, "no-probabilities"
        options = q.get("options")
        if isinstance(options, dict):
            keys = set(options.keys())
        elif isinstance(options, list):
            keys = set(options)
        else:
            return "invalid", None, None, "no-options"
        if set(probs.keys()) != keys:
            return "invalid", None, None, "inexact-choice-keys"
        vals = {k: float(v) for k, v in probs.items()}
        if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in vals.values()):
            return "invalid", None, None, "nonfinite-probability"
        total = sum(vals.values())
        if not math.isfinite(total) or total <= 0 or abs(total - 1.0) > 0.01:
            return "invalid", None, None, "bad-total"
        sel = ans.get("choice", ans.get("selection"))
        if sel is not None and sel not in keys:
            return "invalid", None, None, "unknown-selection"
        return "ok", None, vals, None
    except (TypeError, ValueError):
        return "invalid", None, None, "unparseable"


def pending(con, bank, *, asof, limit=100, model_digest=None,
            template_digest=None, window_h=72) -> list[dict]:
    asof_dt = parse_time(asof)
    questions = _active_questions(con, bank)
    # live doc keys asof (exclude superseded)
    asof_s = asof_dt.isoformat()
    cutoff = (asof_dt - timedelta(hours=window_h)).isoformat()
    visible = con.execute("SELECT * FROM documents WHERE fetched_at<=? AND fetched_at>=? "
                          "AND published_at<=? AND published_at>=? "
                          "AND (superseded_at IS NULL OR superseded_at>?)",
                          (asof_s, cutoff, asof_s, cutoff, asof_s)).fetchall()
    newest = {}
    for doc in visible:
        group = doc["correction_group"]
        if group not in newest or (doc["fetched_at"], doc["doc_key"]) > (newest[group]["fetched_at"], newest[group]["doc_key"]):
            newest[group] = doc
    live_keys = {doc["doc_key"] for doc in newest.values()}
    tasks = []
    # fresh-first ordering
    q = ("SELECT trunks.*, chunks.token_count, chunks.token_count_basis, chunks.state_path AS cpath,"
         " chunks.created_at FROM trunks JOIN chunks ON chunks.chunk_id=trunks.chunk_id"
         " ORDER BY chunks.created_at DESC, trunks.rowid DESC")
    for row in con.execute(q):
        if row["status"] == "retired":
            continue
        if model_digest is not None and row["model_digest"] != model_digest:
            continue
        if template_digest is not None and row["template_digest"] != template_digest:
            continue
        trunk_id = row["trunk_id"]
        docs = [r["doc_key"] for r in con.execute(
            "SELECT doc_key FROM chunk_docs WHERE chunk_id=?", (row["chunk_id"],))]
        vis = [d for d in docs if d in live_keys]
        if not vis:
            continue  # stale/superseded
        # asof-unavailable gate: doc fetched<=asof already ensured; check answers availability below
        missing = []
        for qq in questions:
            hit = con.execute(
                "SELECT status,available_at FROM answers WHERE trunk_id=? AND question_id=? AND question_hash=? AND inference_revision=?",
                (trunk_id, qq["question_id"], qq["question_hash"], INFERENCE_REVISION)).fetchone()
            if hit is None or hit["status"] != "ok":
                missing.append(qq)
            elif hit["available_at"] and parse_time(hit["available_at"]) > asof_dt:
                missing.append(qq)
        if not missing:
            if row["status"] != "ready":
                con.execute("UPDATE trunks SET status='ready' WHERE trunk_id=?", (trunk_id,))
            continue
        # record-level completion: trunk stays pending until ALL bank questions ok
        chunk = con.execute("SELECT * FROM chunks WHERE chunk_id=?", (row["chunk_id"],)).fetchone()
        if chunk is None:
            continue
        try:
            state = json.loads(Path(chunk["state_path"]).read_text(encoding="utf-8"))["text"]
        except OSError:
            continue
        meta = con.execute("SELECT * FROM documents WHERE doc_key=?", (vis[0],)).fetchone() if vis else None
        basis = chunk["token_count_basis"] or "utf8_upper_bound"
        # offline chunks must never claim tokenizer basis
        if not chunk["validated"]:
            basis = "utf8_upper_bound"
        tasks.append({"trunk_id": trunk_id, "chunk_id": row["chunk_id"], "state": state,
                      "state_path": chunk["state_path"], "doc_ids": vis,
                      "published_at": meta["published_at"] if meta else None,
                      "fetched_at": meta["fetched_at"] if meta else None,
                      "model_digest": row["model_digest"], "template_digest": row["template_digest"],
                      "token_count": chunk["token_count"],
                      "token_count_basis": basis if basis == "tokenizer" and chunk["validated"] else "utf8_upper_bound",
                      "questions": missing})
        if len(tasks) >= limit:
            break
    con.commit()
    return tasks


def record(con, task: dict, response: dict, *, available_at, wall_ms=0) -> dict:
    avail_s = utc_iso(parse_time(available_at))
    answers = (response or {}).get("answers", {})
    # 'model' in a model response is the model NAME, not the artifact digest.
    raw_model_name = (response or {}).get("model")
    task_model = task.get("model_digest")
    task_template = task.get("template_digest")
    resp_model_digest = (response or {}).get("model_digest", task_model)
    resp_template_digest = (response or {}).get("template_digest", task_template)
    digest_mismatch = (
        ("model_digest" in (response or {}) and resp_model_digest != task_model)
        or ("template_digest" in (response or {}) and resp_template_digest != task_template))
    ok = invalid = missing = 0
    usage = json.dumps((response or {}).get("usage"), sort_keys=True) if (response or {}).get("usage") is not None else None
    raw_response = json.dumps(response or {}, ensure_ascii=False, sort_keys=True, default=str)
    for q in task.get("questions", []):
        ans = answers.get(q["question_id"])
        if digest_mismatch:
            status, p_yes, probs, error = "invalid", None, None, "digest-mismatch"
        else:
            status, p_yes, probs, error = _validate_answer(q, ans)
        if status == "ok":
            ok += 1
        elif status == "invalid":
            invalid += 1
        else:
            missing += 1
        # never overwrite a successful historical row on retry error
        prior = con.execute(
            "SELECT status FROM answers WHERE trunk_id=? AND question_id=? AND question_hash=? AND inference_revision=?",
            (task["trunk_id"], q["question_id"], q.get("question_hash"), INFERENCE_REVISION)).fetchone()
        if prior is not None and prior["status"] == "ok" and status != "ok":
            if status == "invalid":
                invalid -= 1
            else:
                missing -= 1
            ok += 0  # keep prior; count stays as prior ok below
            continue
        request_id = digest({"trunk": task["trunk_id"], "q": q["question_id"],
                             "h": q.get("question_hash"), "rev": INFERENCE_REVISION})
        raw = answers.get(q["question_id"])
        con.execute(
            "INSERT OR REPLACE INTO answers VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (task["trunk_id"], q["question_id"], q.get("question_hash"), INFERENCE_REVISION,
             status, p_yes, json.dumps(probs, sort_keys=True) if probs else None,
             json.dumps({"answer": raw, "model_name": raw_model_name, "response": response or {}},
                        ensure_ascii=False, sort_keys=True, default=str) if raw is not None else raw_response,
             resp_model_digest, resp_template_digest,
             (response or {}).get("observed_at"), avail_s,
             request_id, wall_ms, usage, error))
    # recompute totals from stored rows so a retry error never flips prior ok
    stored = {r["question_id"]: r["status"] for r in con.execute(
        "SELECT question_id,status FROM answers WHERE trunk_id=? AND inference_revision=?",
        (task["trunk_id"], INFERENCE_REVISION))}
    task_qids = [q["question_id"] for q in task.get("questions", [])]
    all_ok = all(stored.get(qid) == "ok" for qid in task_qids) and len(task_qids) > 0
    # never turn a ready batch into suppression: only mark ready when every
    # bank question for this task is ok
    trunk_status = "ready" if all_ok else "retry"
    ok_n = sum(1 for qid in task_qids if stored.get(qid) == "ok")
    inv_n = sum(1 for qid in task_qids if stored.get(qid) == "invalid")
    miss_n = len(task_qids) - ok_n - inv_n
    con.execute(
        "UPDATE trunks SET status=?, last_attempt_at=?, last_success_at=?, server_generation=?, observed_cache_tokens=?, error=? WHERE trunk_id=?",
        (trunk_status, avail_s, avail_s if ok_n else None,
         (response or {}).get("server_generation"),
         (response or {}).get("observed_cache_tokens"),
         (response or {}).get("error"), task["trunk_id"]))
    con.commit()
    return {"ok": ok_n, "invalid": inv_n, "missing": miss_n, "status": trunk_status}


def feature_rows(con, bank, *, asof, window_h=72, require_available=True,
                 model_digest=None, template_digest=None) -> list[dict]:
    asof_dt = parse_time(asof)
    asof_s = asof_dt.isoformat()
    questions = _active_questions(con, bank)
    cutoff = (asof_dt - timedelta(hours=window_h)).isoformat()
    docs = [dict(r) for r in con.execute(
        "SELECT * FROM documents WHERE fetched_at<=? AND fetched_at>=? "
        "AND published_at<=? AND published_at>=? AND (superseded_at IS NULL OR superseded_at>?)",
        (asof_s, cutoff, asof_s, cutoff, asof_s))]
    newest: dict[str, dict] = {}
    for d in docs:
        g = d["correction_group"]
        if g not in newest or d["fetched_at"] > newest[g]["fetched_at"]:
            newest[g] = d
    live: dict[str, dict] = {d["doc_key"]: d for d in newest.values()}
    # content dedup across multiple doc sources: same content -> one visible
    # revision, but preserve union of doc metadata (doc_ids).
    by_content: dict[str, list[dict]] = {}
    for d in sorted(live.values(), key=lambda d: d["fetched_at"]):
        by_content.setdefault(d["content_sha256"], []).append(d)
    rep_of: dict[str, dict] = {}   # doc_key -> representative doc
    union_ids: dict[str, list[str]] = {}  # content -> all doc_keys
    for content, ds in by_content.items():
        union_ids[content] = [d["doc_key"] for d in ds]
        for d in ds:
            rep_of[d["doc_key"]] = ds[-1]
    rows = []
    seen_state_q: set[tuple] = set()  # per-document content dedup at row level
    for trunk in con.execute("SELECT * FROM trunks ORDER BY rowid"):
        if trunk["status"] == "retired":
            continue
        if model_digest is not None and trunk["model_digest"] != model_digest:
            continue
        if template_digest is not None and trunk["template_digest"] != template_digest:
            continue
        chunk = con.execute("SELECT * FROM chunks WHERE chunk_id=?", (trunk["chunk_id"],)).fetchone()
        if chunk is None:
            continue
        doc_keys = [r["doc_key"] for r in con.execute(
            "SELECT doc_key FROM chunk_docs WHERE chunk_id=? ORDER BY ordinal", (chunk["chunk_id"],))]
        vis = [live[k] for k in doc_keys if k in live]
        if not vis:
            continue
        pub = max(v["published_at"] for v in vis)
        if (asof_dt - parse_time(pub)).total_seconds() > window_h * 3600:
            continue
        try:
            text = json.loads(Path(chunk["state_path"]).read_text(encoding="utf-8"))["text"]
        except OSError:
            text = ""
        # match-revision stable: content_sha256 of the state's docs
        content_sha = vis[0]["content_sha256"]
        # union doc ids across same-content sources
        union = sorted({k for v in vis for k in union_ids.get(v["content_sha256"], [v["doc_key"]])})
        d0 = rep_of.get(vis[0]["doc_key"], vis[0])
        for q in questions:
            key = (chunk["state_sha256"], q["question_id"], q["question_hash"],
                   trunk["model_digest"], trunk["template_digest"])
            if key in seen_state_q:
                continue
            seen_state_q.add(key)
            ans = con.execute(
                "SELECT * FROM answers WHERE trunk_id=? AND question_id=? AND question_hash=? AND inference_revision=?",
                (trunk["trunk_id"], q["question_id"], q["question_hash"], INFERENCE_REVISION)).fetchone()
            if require_available and ans is not None:
                if not ans["available_at"] or parse_time(ans["available_at"]) > asof_dt:
                    ans = None
            rows.append({
                "trunk_id": trunk["trunk_id"], "chunk_id": chunk["chunk_id"],
                "question_id": q["question_id"], "question_hash": q["question_hash"],
                "group": q.get("group"), "subject": q.get("subject"),
                "status": ans["status"] if ans else "missing",
                "p_yes": ans["p_yes"] if ans else None,
                "probabilities": json.loads(ans["probabilities_json"]) if ans and ans["probabilities_json"] else None,
                "doc_ids": union, "correction_group": d0["correction_group"],
                "content_sha256": content_sha, "state_path": chunk["state_path"],
                "state_sha256": chunk["state_sha256"],
                "source_urls": sorted({v["url"] for v in vis if v.get("url")}),
                "sources": sorted({v["source"] for v in vis if v.get("source")}),
                "text": text, "published_at": pub, "fetched_at": d0["fetched_at"],
                "available_at": ans["available_at"] if ans else None,
                "model_digest": trunk["model_digest"],
                "template_digest": trunk["template_digest"]})
    return rows
