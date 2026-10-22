"""Escaped status page (Muse B)."""
from __future__ import annotations

import html
import os
import tempfile
from urllib.parse import urlsplit
from pathlib import Path


def render(snapshot: dict, health: dict | None = None) -> str:
    h = health or {}
    e = html.escape
    rows = []
    for s in snapshot.get("signals") or []:
        tops = []
        for t in (s.get("top_chunks") or [])[:3]:
            links = " ".join(f'<a href="{e(url, quote=True)}">source</a>'
                             for url in t.get("source_urls", []) if isinstance(url, str)
                             and urlsplit(url).scheme in {"http", "https"})
            tops.append(f'<li>{e(str(t.get("chunk_id")))} p={e(str(t.get("p_yes")))} '
                        f'<code>{e(str((t.get("text") or "")[:160]))}</code> {links}</li>')
        rows.append(f"<tr><td>{e(str(s.get('question_id')))}</td><td>{e(str(s.get('label')))}</td>"
                    f"<td>{e(str(s.get('level')))}</td><td>{e(str(s.get('delta')))}</td>"
                    f"<td>{e(str(s.get('spike')))}</td><td>{e(str(s.get('coverage')))}</td>"
                    f"<td><ul>{''.join(tops)}</ul></td></tr>")
    return (f"<!doctype html><html><head><meta charset=utf-8><title>intel-net status</title></head><body>"
            f"<h1>intel-net {e(str(snapshot.get('generation', '')))} asof {e(str(snapshot.get('asof')))}</h1>"
            f"<p>coverage {e(str(snapshot.get('coverage')))} status {e(str(snapshot.get('status')))} "
            f"watermark {e(str(snapshot.get('input_watermark')))}</p>"
            f"<p>health: <code>{e(str(h))}</code></p>"
            f"<table border=1><tr><th>question</th><th>label</th><th>level</th><th>delta</th>"
            f"<th>spike</th><th>coverage</th><th>top</th></tr>{''.join(rows)}</table></body></html>")


def write_status(snapshot: dict, path, health: dict | None = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = render(snapshot, health)
    fd, temp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return target
