# ARC Prize 2026 — ARC-AGI-2 Track: Grandmaster Workspace

Everything you need to compete in `arc-prize-2026-arc-agi-2` on Kaggle.
**Downloaded & verified: Sep 26, 2026.**

## Layout

```
arc-agi-2/
├── data/                  competition data (downloaded via kaggle CLI)
│   ├── arc-agi_training_challenges.json    (1000 tasks)
│   ├── arc-agi_training_solutions.json
│   ├── arc-agi_evaluation_challenges.json  (120 tasks)
│   ├── arc-agi_evaluation_solutions.json
│   ├── arc-agi_test_challenges.json        (placeholder → 240 real tasks at rerun)
│   └── sample_submission.json
├── notebooks/             top public notebooks (pulled via `kaggle kernels pull`)
├── src/                   my starter kit (heuristic DSL baseline + eval + submission)
│   ├── arc.py             task loading, scoring, submission writer
│   ├── baseline_solver.py heuristic solver (identity/color-map/crop/tile/symmetry)
│   ├── evaluate.py        local 2-attempt exact-match scorer
│   └── make_submission.py build submission.json from any solver
├── submissions/           generated submission.json files
└── reference/             competition-brief.md (rules/eval/timeline/LB intel)
```

## Quick start (local)

```bash
cd C:\Users\daive\Downloads\arc-agi-2
python src\evaluate.py --split evaluation      # score any solver locally
python src\make_submission.py                  # build submission.json (240 tasks)
```

## Current state of the art (Sep 26, 2026)

| | Score | Approach |
|---|---|---|
| **Private LB #1** (Tufa Labs) | 83.06 | undisclosed (likely frontier-model agent + harness) |
| **Best public notebook** | 33.89 | `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` |
| NVARC-2025 lineage | ~19–20 | Qwen3 + Unsloth LoRA + TTT + DFS decoding |
| 2024 1st place re-baseline | ~2–6 | DSL + search |
| Pure heuristic DSL (src/) | ~1 | this repo's starting point |

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

- `notebooks/arc-highscore-lb3389-replica/` — clean replica of the LB 33.89 pipeline.
- `notebooks/nvarc-winning-solution-t4x2/` — NVARC ARC'25 winning solution (T4x2).
- `notebooks/reproduce-nvarc-2025-results/` — independent NVARC reproduction.
- `notebooks/arc-agi-enhanced-solution/` (junaid512, 138 votes) — enhanced Qwen solution.
- `notebooks/arc-agi-2-baseline-starter/` — simple starter.
- `notebooks/agi-compressarc-2025/` — boristown's CompressARC (921 votes, 2025).
- `notebooks/lb-5-83-baseline-2024-1st-place/` — 2024 1st-place DSL baseline.

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
