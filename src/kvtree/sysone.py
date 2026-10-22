"""SystemOne HTTP client shared by kvtree probes (d1-3B via llama-server /v1/systemone)."""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def systemone(endpoint, state, questions, timeout=300.0):
    """POST one request. questions: {qid: {type, instructions, criteria?}}.
    Returns (answers, usage, wall_ms)."""
    payload = {"state": state, "questions": questions}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = endpoint.rstrip("/") + "/v1/systemone"
    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode("utf-8"))
    ms = (time.perf_counter() - t0) * 1000.0
    return resp.get("answers", {}), resp.get("usage", {}), ms


def noul(qid, instructions, true_desc=None, false_desc=None):
    q = {"type": "noul", "instructions": instructions}
    crit = {}
    if true_desc is not None:
        crit["true"] = true_desc
    if false_desc is not None:
        crit["false"] = false_desc
    if crit:
        q["criteria"] = crit
    return qid, q


def ask_parallel(endpoint, state, questions, timeout=300.0, workers=4):
    """One-question-per-request, run concurrently. Returns list of
    (qid, answers, usage, ms) in input order."""
    items = list(questions.items())

    def one(item):
        qid, q = item
        a, u, ms = systemone(endpoint, state, {qid: q}, timeout=timeout)
        return qid, a, u, ms

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, items))
