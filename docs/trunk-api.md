# Trunk API — caller contract (patched server)

Launch the server with `--decision-prefix-cache --decision-trunk-cache 512`.

IDs and snapshots are **process-local**. Full state and identical image bytes
remain required in every request; an ID alone cannot reconstruct a request.
A missing, evicted, incompatible or changed-state entry recomputes safely.
**Never make an application decision conditional on a cache hit.**

## Create and ask later

There is no separate create-only endpoint: the first question creates the
trunk, later questions reuse it.

```python
import hashlib
import json
import urllib.request

base = 'http://127.0.0.1:8101'  # your patched server
state = 'Asset X deposits resumed after maintenance. The market is stable.'
# Namespace and revision describe the producer, state and model/template policy.
trunk_id = 'myapp:v1:' + hashlib.sha256(state.encode('utf-8')).hexdigest()

def ask(questions, bypass=False):
    payload = dict(state=state, questions=questions, trunk_id=trunk_id,
                   cache_bypass=bypass)
    request = urllib.request.Request(base + '/v1/systemone',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)

created = ask({'q1': {'type': 'noul',
                      'instructions': 'Does the state mention Asset X?'}})
later = ask({'q2': {'type': 'noul',
                    'instructions': 'Does the state describe a stable market?'}})
fresh = ask({'q2': {'type': 'noul',
                    'instructions': 'Does the state describe a stable market?'}},
            bypass=True)  # forced recompute reference for acceptance
```

Rules: build state once per evidence revision and reuse it for all
questions; new evidence gets a new ID. Question text is not part of the
caller ID. Do not rewrite/reorder/normalize the state between requests. IDs
must be strings of at most 128 UTF-8 bytes. Pin `id_slot` only for
controlled acceptance measurements; pinning production traffic to one slot
can serialize it. No concurrency/scheduling bit-invariance guarantee is made.

Images use the normal `files` payload. Keep identical original bytes and
include their SHA256 in the caller's canonical state identity. The server
independently compares media chunk hashes; changing an image under the same
ID safely misses. Warm requests still perform HTTP media decode and boundary
probes.

## Eviction and caller release

There is NO per-ID delete, listing or TTL endpoint. Caller release means
remove the state/ID from the caller's registry and stop using it; it does
not immediately free the server snapshot. The server evicts
least-recently-used IDs until a new snapshot fits the RAM budget. Same ID
with changed state can replace its entry after a new successful prefill, but
is not a delete operation. Restart, model unload or a LoRA change clears all
entries. `--decision-trunk-cache 0` disables full ID storage; automatic slot
checkpoints are independent.

`cache_bypass=true` forces cross-request recompute without storing/replacing
the ID entry; it does not delete it. Same-request question branching remains
available. Use matched geometry and every answer field for acceptance, and
record separate flag-off comparisons. A successful response and `/health`
alone do not prove cache equality or retention.
