# d1 trunk-memory — d1-3B を多層配置して記憶・呼び出しするシステム

> **注意（必ずお読みください）:** 以下の数値はすべて、変わった環境
> （Windows、Tesla V100 16GB ×2、driver 472.12、CUDA 11.8、sm_70、
> d1-3B Q4_K_M）で測ったものです。他環境での再現性は未確認であり、
> 参考値としてのみ扱ってください。

## これは何か

d1-3B（Liquid AI の decision model。
モデルカード: <https://huggingface.co/LiquidAI/d1-3B>。
重みは同梱しません）は、文書などの state に対して型付きの質問
（yes/no・選択肢・スコア）に一発で較正付き回答を返すモデルです。
読む（prefill）は高く、読み終わった文書への追加質問は安い——この非対称を
常設の記憶システムにしたものです。

層の構造（`docs/layered-design.md`）：

1. **trunk（幹）** — 文書チャンク（約1kトークン）を一度だけ prefill。
   prefill 済み状態をリクエスト間で保持する llama.cpp パッチで、
   warm trunk への新質問は text **78 ms**（cold 968 ms）、画像 **141 ms**
   （cold 1679 ms）。
2. **leaves（葉）** — 質問は幹への安い葉。後から追加でき、再読込は不要。
3. **質問ツリー** — 質問バンクはフラットな一覧ではなく木
   （Akinator 式の二分探索: 経済関連？→暗号資産関連？→どの資産？→良い話？）。
   親が yes の所だけ子を聞くので、多くの文書は根の数問で終わります。
4. **回答マトリクス** — （チャンク × 質問）→ P(yes) / 選択肢確率を SQLite
   に保存。新質問の追加は全 trunk への sweep で数分、再読込なし。
5. **シグナル** — 質問ごとに recency-weighted 合算
   （score × exp(-ln2·age/half-life)）の水準＋変化量。スパイクは上位層
   （大きなローカル判定モデル）へエスカレーション。夜間バッチで生成モデル
   が要約・予測質問化・ソース探索。実行ループにクラウド LLM は入れません。

## 効率化の手法と測定値（詳細は `benchmarks/`）

- **trunk checkpoint + bounded trunk cache**（`patches/`）:
  cross-request で state 境界に復元点を置き、ID 指定で複数幹を RAM LRU 保持。
  512 MiB で約31本（1kトークン）/ 約10本（3kトークン）、復元 9–25 ms。
- **質問のバッチ化**: 1ブランチの葉を1リクエストに同梱で 212→70 ms/leaf
  （約3倍）。1リクエスト512問まで線形（約40 ms/問、精度劣化なし）。
- **VRAM 常駐**: 1 GPU に 64–128本の約2kトークン trunk を載せ、
  バッチ10問で 48–61問/秒、ツリー質問で 40問/秒。
- **CAG yes/no フィルタ**: d1-3B は 1kチャンクで AUC 0.94、しきい値 0.5 の
  まま正答率 0.89、キャッシュありで1問約13 ms（cold 170 ms の約1/14）。
  チャンク長で劣化（2k: 0.90、4k: 0.76）。
- **長文脈**: 29kトークンまで正答を維持（8064 ms、p_yes 0.884）。
- 正しさ: 同一プロセス・同一形状では全回答フィールド完全一致。
  プロセスをまたぐと noul に最大0.0025程度の差（判定の反転なし）。
  VRAM 常駐の warm 比較では最大0.015（160/160 判定一致）。
  ビット同一性が必要な用途には向きません。

## 動かし方

1. パッチ適用済み llama-server を用意（`patches/README.md`）。
   起動例: `--decision-prefix-cache --decision-trunk-cache 512 --metrics`
2. Python パッケージ（標準ライブラリ + pyyaml のみ）:
   - `python -m pytest` — テスト（GPU 不要、mock サーバ使用）
   - `python examples/offline_tick.py` — 推論なしの offline デモ
     （ingest→trunk登録→質問バンク→マトリクス→シグナル）
   - `python -m intel_net` — 常駐ループ（`--config examples/config.json`、
     または `TRUNK_MEMORY_*` 環境変数。推論には operator receipt が必要）
   - `python -m kvtree batch` — probe 実験（`D1_BASE_URL` で向き先指定）
3. API 契約は `docs/trunk-api.md`。

## 制限

- 重み・モデルファイルは含みません（各自で取得）。
- trunk ID の削除・一覧・TTL API はありません（LRU/再起動で回収）。
- CAG のラベルは機械生成（人手確認なし）。29k超・他言語・他ドメインは未測定。
- キャッシュヒットをアプリケーション判定の条件にしないでください
  （欠落・追い出し・状態変更時は安全に再計算します）。

---

# d1 trunk-memory — layered memory & recall over d1-3B

> **Caveat (please read):** every number below was measured on one unusual
> setup (Windows, 2x Tesla V100 16GB, driver 472.12, CUDA 11.8, sm_70,
> d1-3B Q4_K_M). Nothing is confirmed on other hardware — treat all figures
> as reference only.

## What it is

d1-3B (a decision model by Liquid AI; model card:
<https://huggingface.co/LiquidAI/d1-3B>; weights NOT included) answers
typed questions (yes/no, choice, score) over a state (text/JSON/images) in
one forward pass with calibrated probabilities and zero output tokens.
Reading a document (prefill) is expensive; asking one more question about an
already-read document is cheap. This repo turns that asymmetry into a
standing memory system:

1. **Trunks** — each ~1k-token chunk is prefilled once. A llama.cpp patch
   keeps the prefilled state across requests, so a new question on a warm
   trunk costs **78 ms** instead of 968 ms (text), **141 ms** instead of
   1679 ms (image).
2. **Leaves** — questions are cheap leaves on a trunk, addable later with
   no re-reading.
3. **Question tree** — the bank is a tree, not a flat list (Akinator-style
   bisection: economy-relevant? → crypto-relevant? → which asset? →
   good/bad?). Children run only where the parent says yes; most documents
   stop after a few root questions. Pruning loss is measured mechanically
   against a daily full-bank sample.
4. **Answer matrix** — (chunk × question) → probabilities in SQLite, the
   feature store every consumer reads. New questions sweep all retained
   trunks in minutes.
5. **Signals & escalation** — per-question recency-weighted sums
   (score × exp(-ln2·age/half-life)), level + change; spikes escalate to a
   bigger local judge; a generative model runs night batches. No cloud model
   in the running loop.

## Efficiency techniques & measured numbers (see `benchmarks/`)

- **Trunk checkpoints + bounded trunk cache** (`patches/`): 512 MiB holds
  ~31 × 1k-token or ~10 × 3k-token trunks, 9–25 ms restores.
- **Batching**: one request per branch, all leaves included: 212 → 70
  ms/leaf (~3x); up to 512 questions/request, linear ~40 ms/question.
- **VRAM residency**: 64–128 ~2k-token trunks on one GPU; 48–61 q/s batched,
  40 q/s tree queries.
- **CAG yes/no filter**: d1-3B AUC 0.94 @1k chunks, 0.89 accuracy at the
  naive 0.5 threshold, ~13 ms cached (14x vs cold); decays with length.
- **Long context**: correct to 29k tokens with mild confidence decay.
- Exactness: bit-identical within process/matched geometry; cross-process
  noul drift ≤ ~0.0025 (no flips); warm-VRAM drift ≤ 0.015 with 160/160
  decisions agreeing. Not for uses needing bitwise-identical floats.

## How to run

1. Build the patched llama-server (`patches/README.md`):
   `--decision-prefix-cache --decision-trunk-cache 512 --metrics`.
2. Python package (stdlib + pyyaml only):
   - `python -m pytest` — tests (no GPU; mock-server based)
   - `python examples/offline_tick.py` — offline demo without inference
   - `python -m intel_net` — resident loop (`--config examples/config.json`
     or `TRUNK_MEMORY_*` env vars; inference needs an operator receipt)
   - `python -m kvtree batch` — probe experiments (`D1_BASE_URL`)
3. Caller contract: `docs/trunk-api.md`.

## Limitations

- No weights/model files included. No per-ID delete/list/TTL API (LRU /
  restart reclaims). CAG labels are procedural, no human review. Beyond
  29k tokens, other languages/domains: unmeasured. Never condition an
  application decision on a cache hit (misses recompute safely).
