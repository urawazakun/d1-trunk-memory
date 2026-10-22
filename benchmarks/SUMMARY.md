# Benchmarks — summary (aggregate numbers only, no source text)

Setup reference (all numbers below are from this setup only):
Windows, 2x Tesla V100 16GB, driver 472.12, CUDA 11.8 (sm_70),
d1-3B Q4_K_M GGUF (+ F16 projector for images), patched llama-server
(`patches/d1-prefix-cache.patch`, flags `--decision-prefix-cache
--decision-trunk-cache 512`). Raw JSON in `results.json`.

## 1. Warm trunk vs cold recompute (GPU medians, flag ON vs flag OFF)

| case | flag ON | flag OFF |
|---|---|---|
| text cold ref (2091 tok) | 473.8 ms | 905.7 ms |
| text warm trunk (39 proc + 2052 cached) | **77.7 ms** | 968.2 ms |
| text new question, warm (65 + 3078) | **112.9 ms** | 1510.9 ms |
| image cold ref (2607 tok) | 791.7 ms | 1695.6 ms |
| image warm trunk (39 + 2568) | **141.0 ms** | 1679.2 ms |
| image new question, warm (65 + 3852) | **208.1 ms** | 2445.2 ms |

Within-run repeats are PASS-exact in both runs. Cross-run (separate server
processes) max abs diff 0.00254, zero choice/type flips.

## 2. RAM trunk-cache capacity (-c 4096, 512 MiB budget, text)

| state | retained | MiB/entry | restore median | bypass → restored (HTTP) |
|---|---|---|---|---|
| ~1k tokens | 31 | 16.505 | 10.06 ms | 1163.3 → 242.6 ms |
| ~3k tokens | 10 | 48.411 | 25.14 ms | 998.0 → 60.0 ms |

Budget arithmetic (512/16.505 = 31.0) matches measured retention exactly.
Evicted revisits recompute safely with exact answers.

## 3. VRAM trunks (single V100, ~2k-token trunks, one slot each)

| run | 1Q × N parallel | 10Q batched | tree (576Q) | VRAM |
|---|---|---|---|---|
| 64 trunks | 10.3 q/s | **61.0 q/s** | — | ~4.3 GB |
| 128 trunks | 6.09 q/s | **48.0 q/s** | **40.2 q/s** | ~6.6 GB |

Warm vs bypass: yes/no decisions 160/160 agree; noul floats drift ≤ 0.0154
(decisions safe; not for bitwise-identical float use).

## 4. KV-tree batching (pre-patch server, within-request sharing)

250 leaves (50 branches × 5Q): sequential 211.8 ms/leaf → batched 5Q/request
**69.6 ms/leaf (~3x)**. Batch vs single max drift 0.004, no flips.
Up to 512 questions/request scale linearly (~40 ms/question).
Long branches stay correct to 29k tokens (8064 ms, p_yes 0.884).

## 5. CAG yes/no probe (1k-token chunks, 600 questions)

| model | AUC | acc@0.5 | top1-of-5 |
|---|---|---|---|
| d1-3B Q4 (GPU) | **0.941** | **0.888** | 0.875 |
| d1-omni-600M (CPU) | 0.904 | 0.842 | 0.87 |
| LFM2.5-1.2B Q4 | 0.717 | 0.516 (0.738 @thr 0.98) | 0.655 |
| Qwen3-0.6B Q8 (CPU) | 0.882 | 0.393 (0.828 @thr 0.85) | 0.72 |
| BM25 | 0.966 | — | 0.98 |
| bge-reranker-v2-m3 (CPU) | 0.995 | — | 1.00 |

d1-3B usable at threshold 0.5 as-is; LFM/Qwen rows are yes-biased and need
per-model thresholds. Quality decays with chunk length (d1-3B AUC
0.941 → 0.898 → 0.763 at 1k/2k/4k). Cached queries run at ~13 ms vs 170 ms
cold (14x); KV ≈ 11 MB per 1k chunk. Labels are procedural (certain by
construction); no human review. Reranker wins on quality but costs ~4 s/query.
