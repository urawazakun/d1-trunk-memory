# CAG yes/no probe — "does this chunk hold the info?"

A small model holds a collected chunk in its KV cache (prefilled once and
saved) and answers many questions "does this chunk contain information about
X?" with YES/NO (+ probability) in a single forward pass: a cheap first
filter before any expensive model reads anything, and "which of N cached
chunks knows about X".

## Method

- Chunk store: documents grouped into ~1k / 2k / 4k-token chunks, prefilled
  once each.
- Query = cached chunk + short suffix ("does the text above contain
  information about \<X\>? Answer YES or NO"); score = P(YES).
- Benchmark set: positives generated from a chunk (several phrasings incl.
  paraphrase and cross-language), hard negatives (same entity, fact NOT in
  the chunk: different date/number/place, or a neighbouring chunk's fact),
  easy negatives (unrelated topics). Labels certain by construction
  (procedural); no human review.
- Metrics: AUC, accuracy at best threshold, calibration (Brier), ms/query
  cached vs cold, KV size per chunk, chunk-length effect, "pick the right
  chunk out of N" top-1.

## Results (see `benchmarks/`)

d1-3B (Q4, GPU) is the clear pick: 1k-token AUC 0.941, 0.888 accuracy at the
naive 0.5 threshold, ~13 ms cached queries, top-1 0.875. Quality decays with
length (2k: 0.90, 4k: 0.76). d1-omni-600M (CPU) is a strong lightweight
runner-up (AUC 0.90). LFM2.5 1.2B/2.6B and Qwen3-0.6B are yes-biased and need
per-model thresholds (0.77–0.98). LFM2.5-350M fails (AUC 0.46). BM25 scores
higher (0.97–1.0) but is literal word-match; a reranker baseline wins on
quality (AUC 0.995) at ~4 s/query — 25–300x slower than cached CAG queries.

`src/intel_net/probe.py` ships the mechanical-specificity fixture builder
and evaluator used to gate question wording without model labels.
