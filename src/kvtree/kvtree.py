"""KV-tree prototype for d1-3B (llama.cpp b1 server, /v1/systemone).

Mechanism under test: cross-request prefix KV reuse via the server's
slot KV + prompt cache (cache_prompt). Sequential requests that share a
state prefix should only prefill the differing suffix.

Usage: python -m kvtree EXP [args...]  (see main())
Results print as JSON lines.

Server endpoint: ``D1_BASE_URL`` env var (default http://127.0.0.1:8101),
or pass the URL as the first argument before EXP.
"""
import base64
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("D1_BASE_URL", "http://127.0.0.1:8101")

ROOT = ("You answer short factual questions about the market context below. "
        "Reply with the given options only.")
SHARED = ("Shared market + world state 2026-10-09: BTC 67200 USD (+2.1% day). "
          "ETH 2580 USD (+1.2%). US equities mixed, 10y yield 4.2%. "
          "No major geopolitical escalations reported in the last 24h.")

LEAVES = [
    ("has_btc", "choice", "Is this good news for BTC?", {"yes": None, "no": None}),
    ("direction", "choice", "BTC direction next hour?", {"up": None, "down": None, "flat": None}),
    ("conflict", "choice", "Is armed-conflict risk rising?", {"yes": None, "no": None}),
    ("info_x", "choice", "Does this mention semiconductor exports?", {"yes": None, "no": None}),
    ("urgency", "score", "How urgent is this for a trader?",
     ["ignore", "watch", "act soon", "act now"]),
]


def chunk(i, target_tokens=1000):
    # ~4 chars/token
    base = ("News chunk %d: semiconductor export rules tightened; "
            "chipmaker shares fell 3%% in Tokyo. Analysts see limited "
            "pass-through to crypto miners. " % i)
    rep = max(1, (target_tokens * 4) // len(base))
    return base * rep


def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=15) as r:
        vals = {}
        for line in r.read().decode().splitlines():
            if line.startswith("llamacpp:"):
                name, v = line.split(" ")
                vals[name] = float(v)
    return (int(vals.get("llamacpp:prompt_tokens_total", 0)),
            int(vals.get("llamacpp:prompt_tokens_cached_total", 0)))


def ask(state, questions, timeout=300.0, files=None):
    payload = {"state": state, "questions": questions}
    if files:
        payload["files"] = files
    body = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(BASE + "/v1/systemone", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode())
    return (time.perf_counter() - t0) * 1000.0, resp


def qobj(leaf):
    qid, typ, instr, crit = leaf
    q = {"type": typ, "instructions": instr}
    q["criteria"] = crit
    return {qid: q}


def exp_prefix_reuse():
    """Same branch, leaves asked one per request: is the suffix the only cost?"""
    st = ROOT + "\n" + SHARED + "\n" + chunk(7)
    out = []
    m0 = metrics()
    for leaf in LEAVES:
        m_before = metrics()
        ms, resp = ask(st, qobj(leaf))
        m_after = metrics()
        a = resp["answers"][leaf[0]]
        out.append({"leaf": leaf[0], "ms": round(ms, 1),
                    "processed": m_after[0] - m_before[0],
                    "cached": m_after[1] - m_before[1],
                    "usage": resp.get("usage"),
                    "probs": a.get("probabilities", a.get("noul", a.get("score")))})
    m1 = metrics()
    print(json.dumps({"exp": "prefix_reuse", "branch_tokens_est": len(st) // 4,
                      "tot_processed": m1[0] - m0[0], "tot_cached": m1[1] - m0[1],
                      "leaves": out}, ensure_ascii=False))


def exp_correctness(n_branch=3):
    """Same leaf twice (back-to-back) + interleaved other branch: probs stable?"""
    res = []
    for i in range(n_branch):
        st = ROOT + "\n" + SHARED + "\n" + chunk(i)
        _, r1 = ask(st, qobj(LEAVES[0]))
        _, r2 = ask(ROOT + "\n" + SHARED + "\n" + chunk(99), qobj(LEAVES[1]))  # disturb
        _, r3 = ask(st, qobj(LEAVES[0]))
        p1 = r1["answers"]["has_btc"]["probabilities"]
        p3 = r3["answers"]["has_btc"]["probabilities"]
        res.append({"branch": i, "p_first": p1, "p_repeat": p3,
                    "max_abs_diff": max(abs(p1[k] - p3[k]) for k in p1)})
    print(json.dumps({"exp": "correctness", "rows": res}, ensure_ascii=False))


def exp_tree(n_branches=50, leaves_per_branch=5):
    """Full tree sweep: 50 branches x 5 leaves, sequential; wall time + metrics."""
    t0 = time.perf_counter()
    m0 = metrics()
    n_ask = 0
    per_branch_ms = []
    for i in range(n_branches):
        st = ROOT + "\n" + SHARED + "\n" + chunk(i)
        b0 = time.perf_counter()
        for leaf in LEAVES[:leaves_per_branch]:
            ask(st, qobj(leaf))
            n_ask += 1
        per_branch_ms.append(round((time.perf_counter() - b0) * 1000.0, 1))
    wall = (time.perf_counter() - t0) * 1000.0
    m1 = metrics()
    print(json.dumps({"exp": "tree", "branches": n_branches,
                      "leaves_per_branch": leaves_per_branch, "requests": n_ask,
                      "wall_ms": round(wall, 1),
                      "ms_per_leaf": round(wall / n_ask, 2),
                      "processed": m1[0] - m0[0], "cached": m1[1] - m0[1],
                      "per_branch_ms": per_branch_ms}, ensure_ascii=False))


def exp_leaf_cost():
    """Leaf cost vs question length: fixed overhead vs per-token slope."""
    st = ROOT + "\n" + SHARED + "\n" + chunk(7)
    ask(st, qobj(LEAVES[0]))  # warm the prefix
    rows = []
    for extra in (0, 50, 200, 800):
        instr = LEAVES[0][2] + (" Please consider carefully." * (extra // 5) if extra else "")
        m_b = metrics()
        t0 = time.perf_counter()
        ms_list = []
        for _ in range(3):
            ms, resp = ask(st, {"qq": {"type": "choice", "instructions": instr,
                                       "criteria": {"yes": None, "no": None}}})
            ms_list.append(ms)
        m_a = metrics()
        rows.append({"extra_chars": extra,
                     "usage_input_tokens": resp["usage"]["input_tokens"],
                     "ms_min": round(min(ms_list), 1),
                     "processed_per_req": round((m_a[0] - m_b[0]) / 3, 1),
                     "cached_per_req": round((m_a[1] - m_b[1]) / 3, 1)})
    print(json.dumps({"exp": "leaf_cost", "rows": rows}, ensure_ascii=False))


def exp_batch():
    """5 leaves in ONE request vs 5 sequential requests: wall ms + equality."""
    st = ROOT + "\n" + SHARED + "\n" + chunk(7)
    ask(st, qobj(LEAVES[0]))  # warm
    five = {}
    for leaf in LEAVES:
        five.update(qobj(leaf))
    m_b = metrics()
    t0 = time.perf_counter()
    ms1, r1 = ask(st, five)
    m_a = metrics()
    t0b = time.perf_counter()
    singles = {}
    for leaf in LEAVES:
        _, r = ask(st, qobj(leaf))
        singles[leaf[0]] = r["answers"][leaf[0]]
    ms_seq = (time.perf_counter() - t0b) * 1000.0
    eq = {}
    for leaf in LEAVES:
        qid = leaf[0]
        a, b = r1["answers"][qid], singles[qid]
        pa, pb = a.get("probabilities"), b.get("probabilities")
        eq[qid] = True if pa is None else max(abs(pa[k] - pb[k]) for k in pa) < 1e-6
    print(json.dumps({"exp": "batch", "one_request_ms": round(ms1, 1),
                      "one_processed": (m_a[0] - m_b[0]), "one_cached": (m_a[1] - m_b[1]),
                      "one_usage": r1.get("usage"),
                      "sequential_ms": round(ms_seq, 1),
                      "equal": eq}, ensure_ascii=False))


def exp_qcount(n_list=(8, 16, 32, 64, 128, 256)):
    """Questions-per-request scaling: latency + equality vs one-by-one."""
    st = ROOT + "\n" + SHARED + "\n" + chunk(7)
    ask(st, qobj(LEAVES[0]))
    _, ref = ask(st, qobj(LEAVES[0]))
    pref = ref["answers"]["has_btc"]["probabilities"]
    rows = []
    for n in n_list:
        qs = {}
        for j in range(n):
            qs["q%03d" % j] = {"type": "choice",
                               "instructions": "Is this good news for BTC (variant %d)?" % j,
                               "criteria": {"yes": None, "no": None}}
        try:
            ms, resp = ask(st, qs, timeout=600.0)
            p0 = resp["answers"]["q000"]["probabilities"]
            drift = max(abs(p0[k] - pref[k]) for k in p0)
            rows.append({"n": n, "ms": round(ms, 1),
                         "usage": resp.get("usage"), "drift_q000": round(drift, 5),
                         "ok": True})
        except Exception as e:  # noqa: BLE001
            rows.append({"n": n, "ok": False, "error": str(e)[:160]})
    print(json.dumps({"exp": "qcount", "rows": rows}, ensure_ascii=False))


def exp_long(ntokens=(4096, 8192, 16384, 32768)):
    """Long branches: prefill speed + answer sanity (known-answer probe)."""
    rows = []
    for nt in ntokens:
        st = ROOT + "\n" + SHARED + "\n" + chunk(0, target_tokens=nt)
        st = st[: nt * 4]
        qs = {"q": {"type": "choice", "instructions":
                    "Does the context above mention semiconductor exports?",
                    "criteria": {"yes": None, "no": None}}}
        try:
            m_b = metrics()
            ms, resp = ask(st, qs, timeout=900.0)
            m_a = metrics()
            a = resp["answers"]["q"]
            rows.append({"target_tokens": nt, "ms": round(ms, 1),
                         "usage": resp.get("usage"),
                         "processed": (m_a[0] - m_b[0]),
                         "p_yes": round(a["probabilities"]["yes"], 4), "ok": True})
        except Exception as e:  # noqa: BLE001
            rows.append({"target_tokens": nt, "ok": False, "error": str(e)[:200]})
    print(json.dumps({"exp": "long", "rows": rows}, ensure_ascii=False))


def exp_image(img_path, n_questions=12):
    """Image branch: prefill once, N yes/no leaves; tokens/ms/sanity."""
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    data_url = "data:image/png;base64," + b64
    yn = [("img_dark", "Is the image mostly dark overall?"),
          ("img_white", "Is the image mostly white/bright overall?"),
          ("img_text", "Does the image contain readable text?"),
          ("img_red", "Does the image contain prominent red areas?"),
          ("img_blue", "Does the image contain prominent blue areas?"),
          ("img_face", "Does the image show a human face?"),
          ("img_cat", "Does the image show a cat?"),
          ("img_table", "Does the image contain a table or chart?"),
          ("img_outdoor", "Is this an outdoor scene?"),
          ("img_logo", "Does the image show a company logo?"),
          ("img_error", "Does the image show an error message or dialog?"),
          ("img_animal", "Does the image show any animal?")]
    out = []
    m0 = metrics()
    ms, r0 = ask("What does the image show?",  # image encode happens here
                 {"warm": {"type": "noul", "instructions": "Is there anything visible?"}},
                 timeout=600.0, files=[data_url])
    m1 = metrics()
    first = {"ms": round(ms, 1), "usage": r0.get("usage"),
             "processed": m1[0] - m0[0], "cached": m1[1] - m0[1]}
    leaves = []
    for qid, instr in yn[:n_questions]:
        m_b = metrics()
        ms, resp = ask("What does the image show?",
                       {qid: {"type": "noul", "instructions": instr}},
                       timeout=600.0, files=[data_url])
        m_a = metrics()
        leaves.append({"qid": qid, "ms": round(ms, 1), "usage": resp.get("usage"),
                       "processed": m_a[0] - m_b[0], "cached": m_a[1] - m_b[1],
                       "noul": round(resp["answers"][qid]["noul"], 4)})
    print(json.dumps({"exp": "image", "img": img_path,
                      "first": first, "leaves": leaves}, ensure_ascii=False))


def main(argv=None):
    global BASE
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0].startswith("http"):
        BASE = args.pop(0)
    which = args[0] if args else "prefix_reuse"
    rest = args[1:]
    if which == "prefix_reuse":
        exp_prefix_reuse()
    elif which == "correctness":
        exp_correctness()
    elif which == "tree":
        exp_tree(*(int(a) for a in rest) or (50, 5))
    elif which == "leaf_cost":
        exp_leaf_cost()
    elif which == "batch":
        exp_batch()
    elif which == "qcount":
        exp_qcount(tuple(int(a) for a in rest) or (8, 16, 32, 64, 128, 256))
    elif which == "long":
        exp_long(tuple(int(a) for a in rest) or (4096, 8192, 16384, 32768))
    elif which == "image":
        exp_image(rest[0], int(rest[1]) if len(rest) > 1 else 12)
    else:
        sys.exit("unknown exp: " + which)


if __name__ == "__main__":
    main()
