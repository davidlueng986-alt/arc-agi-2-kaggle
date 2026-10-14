# ARC Prize 2026 — ARC-AGI-2 Track: Grandmaster Workspace

Read **[official/HANDBOOK.md](official/HANDBOOK.md)** first. It is the 2026-10-14 copy of the official competition pages: rules that change a submission, data schema, metric, timeline, prizes, L4x4 quota, the public leaderboard, and the notebooks and discussions refreshed that day. You should not need the competition website for those facts.

Official task JSON is downloaded with the Kaggle CLI and is not committed. The rules forbid publishing Competition Data to anyone who has not accepted them.

```powershell
scripts\download_data.ps1
python src\evaluate.py --split evaluation
python src\make_submission.py
```

## Layout

```
├── official/              handbook, 2026-10-14 leaderboard CSV, file hashes
├── data/                  gitignored; created by scripts/download_data.ps1
│   ├── arc-agi_training_challenges.json      (1000 tasks)
│   ├── arc-agi_training_solutions.json
│   ├── arc-agi_evaluation_challenges.json    (120 tasks)
│   ├── arc-agi_evaluation_solutions.json
│   ├── arc-agi_test_challenges.json          (240-task training placeholder; swapped at rerun)
│   └── sample_submission.json
├── notebooks/             public notebooks pulled with kaggle kernels pull
├── src/                   local starter (load, score, heuristic baseline, submission)
├── submissions/           gitignored generated submission.json files
└── reference/             solution strategy; competition-brief.md points at the handbook
```

## Current state of the art (2026-10-14 public leaderboard)

| | Score | Approach |
|---|---|---|
| **Public LB #1** (Tufa Labs) | 88.06 | not published in the notebooks pulled here |
| Teams at or above 33.89 | 64 | public board; methods mostly unpublished |
| **Best named public notebook** | 33.89 | `koushikrudra/failed-in-aimo` and `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` |
| Qwen3.8-27B public agent | 21.53, rerun 13 | `madarshbb/arc-qwen-3-8-27b-agent` |
| NVARC-2025 lineage | ~19–20 on the old public notebooks | Qwen3-4B + LoRA TTT + DFS |
| Pure heuristic DSL (`src/`) | ~1 | local starting point |

The September note that listed Tufa Labs at 83.06 as a private score is superseded by the public leaderboard CSV in `official/`.

**Strategy — HARNESS-FIRST (current plan)**: `reference/harness-first-qwen38-27b.md`.
Drop SFT and TTT entirely; run Qwen3.8-27B locally via vLLM inside the notebook with a
Poetiq-style code-refinement harness + Land-style holistic judge. Rationale: the
published harness SOTA (54.0% / 72.9% on ARC-AGI-2, **no weight updates**) far exceeds
the TTT lineage (Kaggle 2025 top 24%, 2026 best public notebook 33.89), and the SFT
checkpoint behind the TTT pipeline needs a 32-GPU cluster + 3.2M synthetic puzzles.
The public #1 on 2026-10-14 (Tufa Labs, 88.06) has not published a notebook in this repo.

Superseded but retained: `reference/qwen38-27b-solution-map.md` (SFT/TTT scale-up path),
`reference/reproduction-analysis.md` (4B TTT reproduction).

Vendored open-source SOTA references in `reference/`:
- `poetiq-solver/` — Poetiq ARC solver (54.0% verified semi-private); LiteLLM backend,
  so pointing it at a local vLLM server is a ~10-line change.
- `land-modality-search/` — Land's modality-driven search + holistic trace judging
  (72.9% semi-private / 76.1% public), including `docs/ANALYZE_*.md` negative results.

## What the LB 33.89 notebook does (the one to fork)

Pulled to `notebooks/arc-agi2-lb33-89-minimal-perfpatch/`. Key ideas:

1. **Model:** Qwen (14B-class) loaded with **Unsloth** + 4-bit quantization, LoRA fine-tune.
2. **Test-time training (TTT):** fine-tunes on the *train pairs of each evaluation task*
   (augmented ~28 min), then decodes.
3. **DFS decoding (`turbo_dfs`):** beam/DFS search over the 12 ARC tokens using per-token
   NLL from full-vocab logits; ranks candidate grids instead of greedy sampling.
4. **Two attempts:** attempt_1 = best DFS candidate, attempt_2 = runner-up.
5. **Hardware:** L4x4, 12h budget, ~10 min submission buffer.
6. `arc_loader.py` / `arc_decoder.py` / `arc_solver.py` / `starter.py` modular structure —
   swap in your own model while keeping the decoding + submission scaffolding.

Grandmaster forks to try: larger Qwen checkpoint, ensemble NVARC + CompressARC candidates,
better augmentation, dedup candidates across attempts to maximize the 2-shot hit rate.

## Other pulled notebooks worth studying

- `notebooks/land-solver-v7-kaggle/` — **Johan Land's own Kaggle notebook** for the
  72.9% modality-search solver (API-based; useful as a Kaggle-runtime reference).
- `notebooks/arc-highscore-lb3389-replica/` — clean replica of the LB 33.89 pipeline.
- `notebooks/nvarc-winning-solution-t4x2/` — NVARC ARC'25 winning solution (T4x2).
- `notebooks/reproduce-nvarc-2025-results/` — independent NVARC reproduction.
- `notebooks/arc-agi-enhanced-solution/` (junaid512, 138 votes) — enhanced Qwen solution.
- `notebooks/arc-agi-2-baseline-starter/` — simple starter.
- `notebooks/agi-compressarc-2025/` — boristown's CompressARC (921 votes, 2025).
- `notebooks/lb-5-83-baseline-2024-1st-place/` — 2024 1st-place DSL baseline.

Pulled 2026-10-14, after the first snapshot:

- `notebooks/failed-in-aimo/` — most-voted public notebook; later notes call it the LB 33.89 base.
- `notebooks/learned-from-aimo/` — same family, 212 votes.
- `notebooks/arc-baseline-rebuild/` — 2026-10-13 deterministic rebuild of that 33.89 base.
- `notebooks/arc-agi-31-11/` — 2026-10-13 NVARC-style notebook titled 31.11.
- `notebooks/arc-agi2-nvarc-plus-ckpt/` — NVARC+ v3 with incremental checkpoints.
- `notebooks/arc-qwen-3-8-27b-agent/` — Qwen3.8-27B REPL agent, author-reported public 21.53.
- `notebooks/arc-agi-2-agentic-multi-turn-reasoning/` — one persistent agent per task.

## Submission playbook (code competition)

1. Develop in `notebooks/` from the LB 33.89 fork (Kaggle, L4x4, internet ON).
2. Attach competition dataset + any model as a Kaggle dataset (rerun is offline).
3. Respect the rerun contract: read `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json`,
   write `submission.json` within the time buffer (see pulled notebook cell 8).
4. **All 240 task_ids must be in submission.json with both attempts** — the scaffolding
   in `src/arc.py` (`make_submission`) guarantees this.
5. Submit via "Submit to Competition" on the notebook; daily quota applies.

## Useful commands

```bash
# leaderboard
kaggle competitions leaderboard arc-prize-2026-arc-agi-2 --show

# top notebooks
kaggle kernels list --competition arc-prize-2026-arc-agi-2 --sort-by voteCount

# pull a notebook
kaggle kernels pull <user>/<kernel-slug> -p notebooks/<name>
```
