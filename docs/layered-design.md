# Layered memory & recall on d1-3B trunks

d1-3B is a decision model: it takes a **state** (text / JSON / images) plus a
set of **typed questions** (yes/no, choice, score) and returns calibrated
answers in a single forward pass, with zero output tokens. Model card:
<https://huggingface.co/LiquidAI/d1-3B>.

Reading a document (prefill) is expensive; answering one more question about
an already-read document is cheap. This repo turns that asymmetry into a
standing memory system with five layers:

1. **Trunks** — each document chunk (~1k tokens) is prefilled once. The
   prefilled state (attention KV + short-conv state) is kept across requests
   by the patched server (`patches/d1-prefix-cache.patch`), so a new
   question on a warm trunk costs ~78 ms instead of ~968 ms (text) or
   ~141 ms instead of ~1679 ms (image).
2. **Leaves** — every question is a cheap leaf on a trunk. Leaves can be
   added later without re-reading the document.
3. **Question tree** — the bank is a tree, not a flat list (an
   Akinator-style bisection: "is this economy-relevant?" → "crypto-relevant?"
   → "which asset?" → "good/bad?"). Children are asked only when the parent
   says yes, so most documents stop after a few root questions. Parent
   thresholds are set low (recall first); precision comes deeper down.
   Pruning loss is measured mechanically: each day a small random sample is
   run through the full flat bank and children that would have fired under a
   pruned parent are counted; thresholds are adjusted from that. The matrix
   keeps "not asked" distinct from "no".
4. **Answer matrix** — (chunk × question) → P(yes) / choice probabilities,
   stored in SQLite. This is the feature store every consumer reads.
   Retroactive questions sweep a new column over all retained trunks in
   minutes, no re-reading.
5. **Signals & escalation** — per question, recency-weighted sums
   (score × exp(-ln2·age/half-life)), level + change; spikes flagged. A
   spike escalates to a bigger local judge for precision; a generative model
   can run in night batches (summaries, forecast questions, source hunting).
   No cloud model in the running loop.

## Tiers

VRAM (hot, last hours) → host RAM via `--decision-trunk-cache` (days) →
disk slot files (weeks) → evict. Measured: 512 MiB RAM budget holds ~31 ×
1k-token or ~10 × 3k-token trunks with 9–25 ms restores; a single 16 GB GPU
holds 64–128 ~2k-token trunks resident at 40–60 questions/s batched.

## Truth

Outcomes only (price settlement, observations, sensor events) plus
mechanical rules (name/ticker match + date). No model-generated labels.
