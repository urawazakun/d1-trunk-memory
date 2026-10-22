# KV tree — one prefilled context, many final questions

Shape: a fixed instruction root → one shared-context level → one branch per
information chunk → short final questions as leaves on any branch. Each node
is prefilled once; a leaf costs only its few question tokens, and new
questions can be added later without re-reading the data.

## Why a patch was needed

d1 is a hybrid model (short-conv + attention): branching needs both the
attention KV and the conv state at the branch point. The stock server only
created checkpoints for token generation, never for decision prompts, so
every `/v1/systemone` request re-prefilled the whole state (0 tokens cached
cross-request, even when the prefix matched). Within-request parent/child
sharing worked; cross-request reuse did not.

`patches/d1-prefix-cache.patch` (default OFF, `--decision-prefix-cache`)
stops the decode at the verified state boundary, saves a checkpoint there,
and restores it on a prefix hit; plus an explicit-ID path (`trunk_id`) that
keeps multiple full prefix snapshots in a bounded RAM LRU
(`--decision-trunk-cache`, default 512 MiB).

## Measured (see `benchmarks/`)

- Pre-patch, within-request batching: 5 leaves in one request = 69.6 ms/leaf
  vs 211.8 ms/leaf sequential (~3x); max drift 0.004, no choice flips.
- Post-patch, cross-request: warm text trunk 77.7 ms vs 968.2 ms cold;
  image 141.0 ms vs 1679.2 ms; within-run repeats bit-identical.
- 8–512 questions per request scale linearly (~40 ms/question), no cap in
  the model (only slot context and batch size limit you).
- Long branches stay correct to 29k tokens with mild confidence decay.
- 512 px images ≈ 290 tokens; 12 yes/no leaves answer sensibly.

## Use

Gates, CAG probes and relation layers all become "add a leaf" on the same
tree. The `src/kvtree` tool (`python -m kvtree EXP`) reproduces the probe
experiments against any patched server via `D1_BASE_URL`.
