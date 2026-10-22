"""SQLite registry: documents/chunks/trunks/question_versions/answers."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  doc_key TEXT PRIMARY KEY, vault_path TEXT, source TEXT, source_doc_id TEXT,
  content_sha256 TEXT, url TEXT, published_at TEXT, fetched_at TEXT,
  publication_basis TEXT, correction_group TEXT, superseded_at TEXT);
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY, state_sha256 TEXT, state_path TEXT,
  token_count INTEGER, token_count_basis TEXT, chunker_version TEXT,
  validated INTEGER DEFAULT 0,
  media_hashes_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS chunk_docs(
  chunk_id TEXT, doc_key TEXT, ordinal INTEGER, offsets TEXT,
  PRIMARY KEY(chunk_id, doc_key));
CREATE TABLE IF NOT EXISTS trunks(
  trunk_id TEXT PRIMARY KEY, chunk_id TEXT, model_digest TEXT,
  template_digest TEXT, state_version INTEGER, status TEXT,
  last_attempt_at TEXT, last_success_at TEXT, server_generation TEXT,
  observed_cache_tokens INTEGER, error TEXT);
CREATE TABLE IF NOT EXISTS question_versions(
  question_id TEXT, question_hash TEXT, bank_version TEXT, grp TEXT,
  subject TEXT, qtype TEXT, instructions TEXT, options_json TEXT,
  signal_policy_json TEXT, active INTEGER,
  PRIMARY KEY(question_id, question_hash));
CREATE TABLE IF NOT EXISTS answers(
  trunk_id TEXT, question_id TEXT, question_hash TEXT, inference_revision TEXT,
  status TEXT, p_yes REAL, probabilities_json TEXT, raw_answer_json TEXT,
  model_digest TEXT, template_digest TEXT, observed_at TEXT, available_at TEXT,
  request_id TEXT, wall_ms REAL, usage_json TEXT, error TEXT,
  PRIMARY KEY(trunk_id, question_id, question_hash, inference_revision));
CREATE TABLE IF NOT EXISTS quarantine(
  vault_path TEXT PRIMARY KEY, reason TEXT, detail TEXT, seen_at TEXT);
CREATE TABLE IF NOT EXISTS ingest_state(
  vault_path TEXT PRIMARY KEY, doc_key TEXT, mtime_ns INTEGER, size INTEGER);
CREATE INDEX IF NOT EXISTS documents_visible ON documents(fetched_at,published_at);
CREATE INDEX IF NOT EXISTS documents_corrections ON documents(correction_group,fetched_at);
CREATE INDEX IF NOT EXISTS chunk_docs_document ON chunk_docs(doc_key);
CREATE INDEX IF NOT EXISTS trunks_model ON trunks(model_digest,template_digest,chunk_id);
CREATE INDEX IF NOT EXISTS answers_available ON answers(available_at);
"""


def connect(path) -> sqlite3.Connection:
    p = Path(str(path))
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    con.executescript(SCHEMA)
    con.commit()
    # migrations for older DBs
    cols = {r["name"] for r in con.execute("PRAGMA table_info(chunks)")}
    if "token_count_basis" not in cols:
        con.execute("ALTER TABLE chunks ADD COLUMN token_count_basis TEXT")
    cols = {r["name"] for r in con.execute("PRAGMA table_info(chunks)")}
    if "validated" not in cols:
        con.execute("ALTER TABLE chunks ADD COLUMN validated INTEGER DEFAULT 0")
        con.execute("UPDATE chunks SET validated=CASE WHEN token_count_basis='tokenizer' THEN 1 ELSE 0 END")
    con.commit()
    return con


def register_questions(con, bank) -> None:
    """Bank is source of truth: deactivate prior versions, upsert current actives."""
    cur_hashes = {(q["question_id"], q["question_hash"]) for q in bank["questions"]}
    con.execute("UPDATE question_versions SET active=0")
    for q in bank["questions"]:
        con.execute(
            "INSERT INTO question_versions(question_id,question_hash,bank_version,grp,subject,qtype,instructions,options_json,signal_policy_json,active)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(question_id,question_hash) DO UPDATE SET bank_version=excluded.bank_version,"
            " grp=excluded.grp,subject=excluded.subject,qtype=excluded.qtype,instructions=excluded.instructions,"
            " options_json=excluded.options_json,signal_policy_json=excluded.signal_policy_json,active=excluded.active",
            (q["question_id"], q["question_hash"], bank["bank_version"], q.get("group", ""),
             q.get("subject", ""), json.dumps(q.get("type")) if isinstance(q.get("type"), dict) else q.get("type", "noul"),
             q.get("instructions", ""),
             json.dumps(q.get("options"), ensure_ascii=False, sort_keys=True),
             json.dumps(q.get("signal_policy", {"half_life_h": q.get("half_life_h", 6.0), "max_age_h": q.get("max_age_h", 72.0)})),
             1 if q.get("active", True) else 0))
    # any registered version not in the bank stays inactive (superseded)
    con.commit()
