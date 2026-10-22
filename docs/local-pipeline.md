# Local-only pipeline — volume filter, precision judge, batch writer

Goal: run collection → judgement with no cloud model in the loop. Three
roles, by strength and cost:

- **d1-3B — volume filter.** Every new document goes through the KV tree:
  prefill once, many leaves (domain tags, named-asset mentions, market-wide
  flags). Cheap, parallel, always resident.
- **Bigger local judge (e.g. 27B decision model) — precision.** Only what d1
  passes: "is this specifically about \<asset\>?", event type, good/bad,
  final direction. Mid volume. Decision model judging, no generative eyes.
- **Generative model (e.g. 3–4B MoE) — writer in batch windows.**
  Entity/relation extraction to structured text, short summaries for the
  decision context, turning events into forecast questions, source hunting.
  Night batch, not in the hot path.

Labels without any generative model: outcomes (price settlement, observed
events, sensor data) and mechanical rules (name/ticker match + date).

## Sharing one or two GPUs (example: 2× 16 GB)

- Resident: d1-3B stack on one GPU; the judge on the other during the day.
- The generative model takes both cards at night for batch writing; the
  serving stack is restored by morning — or parked to a single GPU if the
  harness supports partial offload (verify VRAM fit first: judge quants
  ~13–15 GB, d1-3B Q4 ≈ 1.7 GB + KV pool).
- Time-share alternative: alternate the second GPU between the d1 stack
  (bulk gating, hourly) and the judge (precision pass), with d1-on-CPU as
  fallback.

Fetching (RSS, notices, public slices) needs no model at all and is already
local in any implementation; only the running judgement loop must stay
cloud-free. Building/development may use any tool.
