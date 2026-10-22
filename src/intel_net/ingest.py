"""Vault ingest: bounded, idempotent, quarantine-aware, stable chunks."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from datetime import timedelta, timezone
from pathlib import Path

from . import chunking
from .contracts import atomic_json, digest, parse_time, utc_iso

INFERENCE_REVISION = "r1"
MAX_PUB_AGE_S = 72 * 3600
_DATED_RE = re.compile(r"(\d{4})[/\\](\d{2})[/\\](\d{2})")


def _read_doc(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            value = json.load(f)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("vault document must be an object")
    return value


def _doc_text(doc: dict) -> str:
    # support dict-typed content containers
    raw = doc.get("raw")
    if isinstance(raw, dict):
        for key in ("text", "body", "content", "article"):
            v = raw.get(key)
            if isinstance(v, str) and v.strip():
                core = v
                break
            if isinstance(v, dict):
                # e.g. {"type": ..., "text": ...}
                t = v.get("text") or v.get("body") or v.get("content")
                if isinstance(t, str) and t.strip():
                    core = t
                    break
        else:
            core = None
        if core is None:
            s = doc.get("summary")
            core = s if isinstance(s, str) else ""
    elif isinstance(raw, str):
        core = raw
    else:
        s = doc.get("summary")
        core = s if isinstance(s, str) else ""
        if not isinstance(core, str):
            raise ValueError("unparseable-content")
    title = doc.get("title")
    header = f"# {title}\n" if isinstance(title, str) and title else ""
    text = header + core
    if not isinstance(text, str):
        raise ValueError("unparseable-content")
    if not core.strip():
        # never silently ingest a header-only/empty document; quarantine instead
        raise ValueError("empty-content")
    return text


def _eligible_paths(vault_dir: Path, asof_dt, window_s: float, limit: int):
    # Visit only recent date subtrees. Do not walk the historic vault before
    # applying the live window. One-day padding covers date-folder timezones.
    roots = []
    day = (asof_dt - timedelta(seconds=window_s, days=1)).date()
    end = (asof_dt + timedelta(days=1)).date()
    while day <= end:
        candidate = vault_dir / day.strftime("%Y/%m/%d")
        if candidate.is_dir():
            roots.append(candidate)
        day += timedelta(days=1)
    # Undated fixture/legacy subtrees; four-digit years are never backfilled.
    if vault_dir.exists():
        roots.extend(p for p in vault_dir.iterdir() if p.is_dir() and not (len(p.name) == 4 and p.name.isdigit()))
    cands = list(vault_dir.glob("*.json")) + list(vault_dir.glob("*.json.gz"))
    for root in roots:
        cands.extend(root.rglob("*.json"))
        cands.extend(root.rglob("*.json.gz"))
    cands = [p for p in cands if not p.name.endswith(".tags.jsonl")]
    # Restrict dated YYYY/MM/DD folders to the window (relative to asof date),
    # undated paths always eligible. Latest-first so recent docs are not starved.
    from datetime import datetime, timezone as _tz
    ok = []
    for p in cands:
        m = _DATED_RE.search(str(p))
        if m:
            try:
                d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=_tz.utc)
            except ValueError:
                continue  # invalid dated folder -> skip (not eligible)
            age = (asof_dt - d).total_seconds()
            if age < -24 * 3600 or age > window_s + 24 * 3600:
                continue  # dated folder outside window (+1 day grace for folder granularity)
        ok.append(p)
    ok.sort(key=lambda p: (p.stat().st_mtime_ns if p.exists() else 0, str(p)), reverse=True)
    return ok


def ingest(con, vault_dir, chunk_dir, *, asof, bank, model_digest,
           template_digest, tokenize=None, window_h=72, limit=1000) -> dict:
    asof_dt = parse_time(asof)
    vault_dir, chunk_dir = Path(vault_dir), Path(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    summary = {"docs_seen": 0, "docs_kept": 0, "chunks": 0, "trunks": 0,
               "quarantined": 0, "reused": 0, "incomplete": False}
    window_s = window_h * 3600
    eligible = _eligible_paths(vault_dir, asof_dt, window_s, limit)
    processed = 0
    for path in eligible:
        try:
            st = path.stat()
        except OSError:
            continue
        try:
            doc = _read_doc(path)
        except Exception as error:
            con.execute("INSERT OR REPLACE INTO quarantine VALUES(?,?,?,?)",
                        (str(path), "unparseable", f"{type(error).__name__}", utc_iso(asof_dt)))
            summary["quarantined"] += 1
            continue
        summary["docs_seen"] += 1
        if doc.get("fetched_at") is None:
            con.execute("INSERT OR REPLACE INTO quarantine VALUES(?,?,?,?)",
                        (str(path), "missing_fetched_at", "", utc_iso(asof_dt)))
            summary["quarantined"] += 1
            continue
        try:
            fetched_at = parse_time(doc.get("fetched_at"))
        except Exception:
            con.execute("INSERT OR REPLACE INTO quarantine VALUES(?,?,?,?)",
                        (str(path), "bad_fetched_at", "", utc_iso(asof_dt)))
            summary["quarantined"] += 1
            continue
        if (fetched_at - asof_dt).total_seconds() > 0:
            continue  # not yet visible; no leakage (does not consume limit)
        if (asof_dt - fetched_at).total_seconds() > window_s:
            continue
        pub_raw = doc.get("published_at")
        basis = "published"
        try:
            published_at = parse_time(pub_raw) if pub_raw else None
        except Exception:
            con.execute("INSERT OR REPLACE INTO quarantine VALUES(?,?,?,?)",
                        (str(path), "bad_published_at", str(pub_raw)[:200], utc_iso(asof_dt)))
            summary["quarantined"] += 1
            continue
        if published_at is None:
            published_at, basis = fetched_at, "fetched_fallback"
        elif (published_at - asof_dt).total_seconds() > 0:
            con.execute("INSERT OR REPLACE INTO quarantine VALUES(?,?,?,?)",
                        (str(path), "future_published_at", str(pub_raw)[:200], utc_iso(asof_dt)))
            summary["quarantined"] += 1
            continue
        # Reject old publication (>72h by max_age) even if recently fetched.
        if (asof_dt - published_at).total_seconds() > window_s:
            con.execute("INSERT OR REPLACE INTO quarantine VALUES(?,?,?,?)",
                        (str(path), "stale_publication", str(pub_raw)[:200], utc_iso(asof_dt)))
            summary["quarantined"] += 1
            continue
        try:
            text = _doc_text(doc)
        except Exception as error:
            con.execute("INSERT OR REPLACE INTO quarantine VALUES(?,?,?,?)",
                        (str(path), "unparseable_content", f"{type(error).__name__}", utc_iso(asof_dt)))
            summary["quarantined"] += 1
            continue
        if not text.strip():
            continue
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            rel = str(path.relative_to(vault_dir))
        except ValueError:
            rel = str(path)
        source = doc.get("source") or path.parent.name
        doc_type = doc.get("type")
        if isinstance(doc_type, dict):
            source = str(doc_type.get("source", source))
        doc_key = digest({"source": source, "path": rel, "hash": content_hash})
        url = doc.get("url")
        correction_group = url or (str(source) + ":" + str(doc.get("id"))) if doc.get("id") else url or doc_key
        cur = con.execute("SELECT doc_key FROM documents WHERE doc_key=?", (doc_key,)).fetchone()
        is_new = cur is None
        con.execute(
            "INSERT OR IGNORE INTO documents(doc_key,vault_path,source,source_doc_id,content_sha256,url,published_at,fetched_at,publication_basis,correction_group,superseded_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (doc_key, rel, source, doc.get("id"), content_hash, url,
             utc_iso(published_at), utc_iso(fetched_at), basis,
             str(correction_group), None))
        # ingest receipt: skip unchanged docs (same path+mtime+size+doc_key) but
        # still ensure trunks exist for this model/template (revalidation skip).
        receipt = con.execute("SELECT doc_key,mtime_ns,size FROM ingest_state WHERE vault_path=?", (rel,)).fetchone()
        unchanged = (receipt is not None and receipt["doc_key"] == doc_key
                     and receipt["mtime_ns"] == st.st_mtime_ns and receipt["size"] == st.st_size)
        # Validation belongs to the loaded model/template, not merely file mtime.
        current = con.execute(
            "SELECT chunks.validated,chunks.token_count_basis FROM trunks JOIN chunks ON chunks.chunk_id=trunks.chunk_id "
            "JOIN chunk_docs ON chunk_docs.chunk_id=chunks.chunk_id WHERE chunk_docs.doc_key=? "
            "AND trunks.model_digest=? AND trunks.template_digest=? AND trunks.status!='retired'", (doc_key, model_digest, template_digest)).fetchall()
        validated_current = bool(current) and all(r["validated"] and r["token_count_basis"] == "tokenizer" for r in current)
        if unchanged and current and (tokenize is None or validated_current):
            summary["reused"] += 1
            continue
        if processed >= limit:
            summary["incomplete"] = True
            continue  # stable docs above do not consume new-work budget
        if tokenize is not None:
            # Offline geometry is a planning artifact. Once the actual model
            # validates this document, only that geometry enters its live matrix.
            con.execute("UPDATE trunks SET status='retired' WHERE model_digest=? AND template_digest=? "
                        "AND chunk_id IN (SELECT chunks.chunk_id FROM chunks JOIN chunk_docs "
                        "ON chunk_docs.chunk_id=chunks.chunk_id WHERE chunk_docs.doc_key=? AND chunks.validated=0)",
                        (model_digest, template_digest, doc_key))
        if is_new:
            con.execute(
                "UPDATE documents SET superseded_at=? WHERE correction_group=? AND fetched_at<? AND doc_key!=? AND superseded_at IS NULL",
                (utc_iso(fetched_at), str(correction_group), utc_iso(fetched_at), doc_key))
        else:
            summary["reused"] += 1
        pieces, chunker_version, basis_tok = chunking.split_text(text, tokenize)
        validated = 1 if basis_tok == "tokenizer" else 0
        for ordinal, piece in enumerate(pieces):
            state_sha = hashlib.sha256(piece.encode("utf-8")).hexdigest()
            state_path = str(chunk_dir / f"{state_sha}.json")
            if tokenize:
                token_count = len(tokenize(piece))
            else:
                token_count = len(piece.encode("utf-8"))
            canonical = {"state": piece, "model": model_digest, "template": template_digest,
                         "chunker": chunker_version, "media": []}
            trunk_id = "intel:v1:" + digest(canonical)
            assert len(trunk_id.encode("utf-8")) < 128
            chunk_id = f"chunk:{chunker_version}:{digest([model_digest, template_digest])[:16]}:{state_sha}:{ordinal}"
            if not Path(state_path).exists():
                atomic_json(state_path, {"text": piece, "state_sha256": state_sha,
                                         "doc_keys": [doc_key]})
            con.execute(
                "INSERT OR IGNORE INTO chunks(chunk_id,state_sha256,state_path,token_count,token_count_basis,chunker_version,validated,media_hashes_json,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (chunk_id, state_sha, state_path, token_count, basis_tok,
                 chunker_version, validated, "[]", utc_iso(asof_dt)))
            con.execute("INSERT OR IGNORE INTO chunk_docs VALUES(?,?,?,?)",
                        (chunk_id, doc_key, ordinal, ""))
            row = con.execute("SELECT trunk_id FROM trunks WHERE trunk_id=?", (trunk_id,)).fetchone()
            if row is None:
                con.execute(
                    "INSERT INTO trunks(trunk_id,chunk_id,model_digest,template_digest,state_version,status,last_attempt_at,last_success_at,server_generation,observed_cache_tokens,error)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (trunk_id, chunk_id, model_digest, template_digest, 1,
                     "pending", None, None, None, None, None))
                summary["trunks"] += 1
            summary["chunks"] += 1
        con.execute("INSERT OR REPLACE INTO ingest_state VALUES(?,?,?,?)",
                    (rel, doc_key, st.st_mtime_ns, st.st_size))
        summary["docs_kept"] += 1
        processed += 1
    con.commit()
    return summary


def _ensure_trunks(con, doc_key, model_digest, template_digest, asof_dt, summary):
    """For unchanged receipted docs: create missing trunks for this model/template
    without re-tokenizing (chunk reuse)."""
    for r in con.execute("SELECT chunk_id FROM chunk_docs WHERE doc_key=?", (doc_key,)):
        chunk = con.execute("SELECT * FROM chunks WHERE chunk_id=?", (r["chunk_id"],)).fetchone()
        if chunk is None:
            continue
        try:
            piece = json.loads(Path(chunk["state_path"]).read_text(encoding="utf-8"))["text"]
        except OSError:
            continue
        canonical = {"state": piece, "model": model_digest, "template": template_digest,
                     "chunker": chunk["chunker_version"], "media": []}
        trunk_id = "intel:v1:" + digest(canonical)
        row = con.execute("SELECT trunk_id FROM trunks WHERE trunk_id=?", (trunk_id,)).fetchone()
        if row is None:
            con.execute(
                "INSERT INTO trunks(trunk_id,chunk_id,model_digest,template_digest,state_version,status,last_attempt_at,last_success_at,server_generation,observed_cache_tokens,error)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (trunk_id, chunk["chunk_id"], model_digest, template_digest, 1,
                 "pending", None, None, None, None, None))
            summary["trunks"] += 1
