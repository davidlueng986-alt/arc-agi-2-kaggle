# ARC Prize 2026 — ARC-AGI-2 Track: Competition Brief

- **Slug:** `arc-prize-2026-arc-agi-2`
- **Type:** Featured (Code Competition — notebook rerun)
- **Deadline:** Nov 2, 2026 (Entry deadline Oct 26, 2026; Start Mar 25, 2026)
- **Prizes:** $700,000 total — Progress Prizes $275,000 + Grand Prize
- **Compute:** Kaggle L4x4 machines ("Upgraded Accelerators")

## Evaluation

Score = percentage of correct task test outputs. For each task test output you get
**2 attempts** (`attempt_1`, `attempt_2`). Exact match only (grid dimensions AND every
cell must match). Score per output = 1 if either attempt matches, else 0. Final score =
mean over all task test outputs.

Submission format (`submission.json`):

```json
{"00576224": [{"attempt_1": [[0,0],[0,0]], "attempt_2": [[0,0],[0,0]]}],
 "12997ef3": [{"attempt_1": ..., "attempt_2": ...},
              {"attempt_1": ..., "attempt_2": ...}]}
```

- Every task_id in the test challenges file MUST appear in submission.json.
- Both attempts must be present, even if you don't have 2 distinct predictions.
- Multiple test inputs per task: outputs must be in the SAME ORDER as test inputs.

## Data Files (in ./data)

| File | Purpose |
|---|---|
| `arc-agi_training_challenges.json`   | 1000 public training tasks (train pairs) |
| `arc-agi_training_solutions.json`    | ground-truth outputs for those tasks |
| `arc-agi_evaluation_challenges.json` | 120 public evaluation tasks (validation) |
| `arc-agi_evaluation_solutions.json`  | ground-truth outputs for those tasks |
| `arc-agi_test_challenges.json`       | PLACEHOLDER during development; swapped for the real 240 unseen tasks at rerun |
| `sample_submission.json`             | format reference |

Task structure: `{"train": [{input, output}, ...], "test": [{input}, ...]}`
Grids: rectangular matrices of ints 0–9, sizes 1x1 to 30x30.

## Rerun facts

- Rerun uses **240 unseen tasks** in `arc-agi_test_challenges.json`.
- Most tasks have 1 test input; a few have 2.
- 12h notebook runtime limit; must write `submission.json` before the time buffer.

## Leaderboard snapshot (Sep 26, 2026)

| Rank | Team | Score |
|---|---|---|
| 1 | Tufa Labs | 83.06 |
| 2 | rabbithole | 76.94 |
| 3 | nvbanana | 74.17 |
| 4 | Yi-Chia Chen | 55.14 |
| 5 | Kha Vo | 37.50 |
| 6 | Junhua Yang | 37.22 |
| 7 | Nguyen | 36.94 |
| 8 | AI Winter | 36.25 |
| ... | Team BlackBox / Mithil A Vakde | 33.89 ← **best PUBLIC notebook score** |

## Code Requirements (Kaggle code competition)

- Submission via notebook rerun (Kerneler), no internet in rerun, standard Kaggle GPU quotas.
- All assets (models/datasets) must be attached as Kaggle datasets/models.
