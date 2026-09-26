# Reproduction Analysis: LB 33.89 Public Notebooks (Sep 2026)

Can we fully reproduce the best public notebooks for `arc-prize-2026-arc-agi-2`,
and improve on them? Short answer: **yes on Kaggle (high confidence), yes locally
(with a >=24GB GPU), and there are 6+ concrete improvements worth 1–3 LB points each.**

## 1. What the LB 33.89 notebook actually does

Sources extracted to `reference/lb3389-source/` (arc_loader.py, arc_decoder.py,
arc_solver.py, starter.py — the 4 `%%writefile` modules + 5 runner cells).

Pipeline (per task, from `arc_solver.py:worker`):

| Stage | Detail |
|---|---|
| Base model | `sorokin/qwen3_4b_grids15_sft139` — Qwen3-4B bf16 SFT'd on ARC grids (public Kaggle Model, transformers/bfloat16/1) |
| Test-time training | 16 augmented copies of the task's train pairs (transpose + rot90 x3 = D8 group, random color permutations `permute0..9`, example shuffling); LoRA r=256, RSLoRA alpha=32, lr 5e-5, 1 epoch, bf16, max_seq 8192 |
| Decoding | `turbo_dfs`: batched DFS/beam search over the 12 ARC tokens (`0-9`, `Ċ` newline, `<|im_end|>`) with shared KV-cache; NLL threshold `max_score = -log(0.2)`; 540s per DFS call, 1200s per puzzle; 16 augmented views of the test input, batched 8 at a time |
| Re-ranking | `calc_scores`: teacher-forced NLL of each candidate grid over 8 fresh augmentations (cached by grid hash) |
| Multi-GPU | 4 workers (1 per L4) via torch.multiprocessing, pulling tasks from a queue; 12h budget minus 600s write buffer |
| Attempts | top-2 candidates by augmented NLL -> `attempt_1` / `attempt_2` |

Key code spots: `turbo_dfs` at `arc_solver.py:112`, TTT at `arc_solver.py:340-395`,
augmentation at `arc_loader.py:248-255` (`augment()`), decoding batches at
`arc_solver.py:407-433`.

The two LB 33.89 notebooks (mikelou1 "minimal perfpatch" and qiuqiuh "replica") are
near-identical (cell-level diff: tiny tweaks in arc_solver.py/starter.py); the mikelou1
version adds a CPU-transfer optimization — only the 12 ARC-token NLLs are moved off
GPU instead of the whole vocab logits — same beam set and ranking, ~2x faster decode.

## 2. Reproduction on Kaggle — HIGH confidence

Fork any of: `mikelou1/arc-agi2-lb33-89-minimal-perfpatch`,
`qiuqiuh/arc-highscore-lb3389-replica`, or the NVARC lineage
(`nihilisticneuralnet/baseline-nvarc-arc-25-winning-solution-for-t4x2`).

Checklist:

1. Attach inputs: competition `arc-prize-2026-arc-agi-2` + model
   `sorokin/qwen3_4b_grids15_sft139` (transformers/bfloat16/1).
   NOTE: the LB33.89 forks reference `/kaggle/input/models/sorokin/...` while
   NVARC notebooks use the standard `/kaggle/input/qwen3_4b_grids15_sft139/...`
   mount — align the path in `arc_solver.py:302` with how you attach it.
2. Settings: L4 x4, 12h runtime, internet ON (pip cell installs unsloth at runtime).
3. Dev loop: without rerun env var, it runs the **evaluation** challenges and can
   self-score via `ArcDataset.validate_submission` — instant local feedback.
4. Expected result: 33.89 on the rerun LB (this IS the current public-best pipeline).

Reproduction risks:
- **unsloth version drift** — the notebook carries `UnslothFixedTrainer` patching
  unsloth issue #2435; pin the same versions the notebook used, or pre-bake a
  wheelhouse dataset (the 2024-1st-place notebook does exactly this).
- Kaggle image / pip availability at rerun time.
- The `sorokin` model being deprecated or unlisted (it is attachable today).
- CUDA nondeterminism: expect +/- ~1 point between runs (multiple teams at 33.47-33.89
  on the LB are running this same notebook).

## 3. Reproduction locally — feasible

- Model weights: **verified downloadable** via
  `kagglehub.model_download('sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1')`
  (~8GB, 2 safetensors shards). Competition data: already in `data/`.
- Requirements: Python 3.11, torch + CUDA, unsloth, transformers, peft, datasets;
  >= 1 GPU with 24GB (L4 / A10 / RTX 3090 / 4090). 4 GPUs to match Kaggle timing;
  with 1 GPU expect ~4x wall clock (12h -> ~2 days for 240 tasks; ~1 day for the
  120-task eval split).
- Code changes: replace `/kaggle/input/...` paths, and downgrade `starter.py`
  multi-worker spawn to single GPU (`CUDA_VISIBLE_DEVICES=0`, one worker).

## 4. Improvement roadmap (ranked by expected LB gain / effort)

1. **P1 — Diverse 2-attempt selection.** Scoring is binary exact-match, so attempt_2
   should maximize *coverage*, not NLL rank. After picking argmax, prefer the
   runner-up with a **different grid shape** (or different cell content at >x% cells).
   Cheap change in the final ranking; expected +0.5-2.
2. **P1 — Wider DFS.** `max_score = -log(0.2)` (~1.61) -> `-log(0.1)` (~2.30) keeps
   ~2x more beams; uncomment the `[:5]` top-candidates truncation at
   `arc_solver.py:142`; raise per-puzzle budget 1200s -> adaptive. Expected +1-3.
3. **P1 — DSL/heuristic ensemble seeding.** For tasks where a deterministic
   train-consistent transform exists (identity, color map, crop, tile — see
   `src/baseline_solver.py`), inject that exact grid as a candidate (or attempt_1)
   before LLM ranking. Free wins on the easy subset. Expected +1-2.
4. **P2 — Self-consistency as a second decoder.** Temperature-sample k=8 completions
   per augmented view, majority-vote, feed winners into the same `calc_scores`
   ranking pool alongside DFS beams. Cheap compute, catches tasks DFS prunes.
5. **P2 — More TTT.** `augment(n=16)` -> 32; 2 epochs for tasks whose post-TTT train
   NLL stays high; more color-permutation draws. Expected +1-3 but compute-hungry
   (watch the 12h wall clock).
6. **P2 — Bigger base model.** Qwen3-14B (4-bit QLoRA TTT) with the same pipeline,
   or re-run sorokin's SFT recipe ("grids15") at larger scale; a 2B variant exists
   (`qwen3_2b_grids15_sft141`) for scaling studies. Highest effort, highest ceiling.
7. **P3 — Task-adaptive time budgeting.** Rank tasks by post-TTT train NLL, solve
   easiest-first (CompressARC strategy), spend leftover time re-decoding borderline
   tasks with wider beams.
8. **P3 — Cross-checkpoint ensemble.** Union candidates from the 4B and 2B SFT
   checkpoints; rank by combined NLL.

Realistic ceiling with 1-6 inside Kaggle's 12h L4x4 box: **~37-40 LB**.
Beyond that is private-team territory (top-3 at 74-83 are likely frontier-model
agents or much larger TTT infra, not this pipeline).

## 5. Verification plan

1. Fork + run on Kaggle in dev mode (eval split) -> confirm ~33% on eval set.
2. Port `src/evaluate.py` scoring to the notebook's outputs for ablations.
3. One change at a time, measure on the 120-task eval split (public solutions),
   keep the 2-attempt format via `src/arc.py::make_submission` guarantees.

## 6. Rules note

Check the competition Rules page for external-data/model policy before
attaching non-Kaggle assets (Kaggle code competitions typically require all
inputs be public Kaggle resources at rerun time — the sorokin model qualifies).
