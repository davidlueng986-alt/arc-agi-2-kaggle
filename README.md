# ARC Prize 2026 — ARC-AGI-2 Track: Grandmaster Workspace

Everything you need to compete in `arc-prize-2026-arc-agi-2` on Kaggle.
**Downloaded & verified: Sep 26, 2026.**

---

## This branch: `h2/duck-repl` — Duck-REPL Transfer

One hypothesis branch of the entry. **Read `HYPOTHESIS.md` first** — it has the
claim, the Duck→ARC-AGI-2 mechanism mapping, the Duck-vs-Land image
reconciliation, the measured token/wall-clock budget with the arithmetic shown,
the predicted score, and exactly what observation would refute it.

```
HYPOTHESIS.md          the claim, budget arithmetic, refutation criteria
harness/               the agentic REPL harness (16 modules, no torch dependency)
  oracle.py            exact-replay verification + diff diagnostics (spliceable)
  segmentation.py      4-connected components, hashes, adjacency, nesting
  sandbox.py           subprocess REPL: 30s cap, 4096-char output, state reset
  agent.py             the per-task act-observe loop (verify == the Duck's action)
  llm.py               vLLM/OpenAI-compatible client + MockLLM + markup recovery
  prompts.py           system prompt, task block, status/ledger messages
  context.py           token estimation + eviction (protected prefix)
  budget.py            token/wall-clock accounting + watchdog
  submission.py        candidate pool, 2-attempt selection, schema, atomic write
  runner.py            workers, cumulative submission, adaptive scheduling
  render.py            4x-style panel PNG sized to a vision-token budget
tests/                 377 tests: every non-LLM component + a real notebook dry run
scripts/eval_local.py  score a split against ground truth; --mock, --limit, --project
scripts/measure_context.py  measured per-turn context cost over all 120 eval tasks
scripts/build_notebook.py   generates notebook/*.ipynb from harness/ + cells/
scripts/run_notebook_locally.py  executes the real notebook cells on a laptop
scripts/build_wheelhouse.sh offline vLLM wheels for the internet-less rerun
notebook/              arc26_h2_duck_repl.ipynb + kernel-metadata.json (enable_gpu: false)
reference/duck-harness/ The Duck, vendored (MIT); example-run/ dropped, 2.9 GB
```

```bash
python -m pytest tests -q                              # 377 tests, ~45 s, no GPU
python scripts/measure_context.py                      # measured context budget
python scripts/eval_local.py --mock --project          # plumbing + 240-task forecast
python scripts/eval_local.py --base-url http://127.0.0.1:8000/v1 --limit 20
python scripts/build_notebook.py                       # regenerate the notebook
kaggle kernels push -p notebook                        # -> liangwanyiudavid/arc26-h2-duck-repl
```

Everything runs with **no GPU and no torch** (the dev machine is an 8 GB laptop
that cannot host a 27B model), so the LLM client is injectable and `MockLLM`
drives the whole stack in tests.

---

## Shared workspace (all branches)

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
| **Frontier reference** (GPT-6 Astra, ARC-AGI-2) | 95.0% | frontier API model (harness-sensitive) |
| **Private LB #1** (Tufa Labs) | 83.06 | undisclosed (likely frontier-model agent + harness) |
| **Best public notebook** | 33.89 | `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` |
| NVARC-2025 lineage | ~19–20 | Qwen3 + Unsloth LoRA + TTT + DFS decoding |
| 2024 1st place re-baseline | ~2–6 | DSL + search |
| Pure heuristic DSL (src/) | ~1 | this repo's starting point |

**Strategy — HARNESS-FIRST (current plan)**: `reference/harness-first-qwen38-27b.md`.
Drop SFT and TTT entirely; run Qwen3.8-27B locally via vLLM inside the notebook with a
Poetiq-style code-refinement harness + Land-style holistic judge. Rationale: the
published harness SOTA (54.0% / 72.9% on ARC-AGI-2, **no weight updates**) far exceeds
the TTT lineage (Kaggle 2025 top 24%, 2026 best public notebook 33.89), the SFT
checkpoint behind the TTT pipeline needs a 32-GPU cluster + 3.2M synthetic puzzles, and
Kaggle's private #1 (Tufa Labs = MindsAI) is a **program-synthesis** solver.

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
