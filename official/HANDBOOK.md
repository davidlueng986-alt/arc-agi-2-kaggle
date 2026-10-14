# ARC Prize 2026 — ARC-AGI-2 handbook

Snapshot date: **2026-10-14**. Source: Kaggle CLI against `arc-prize-2026-arc-agi-2` (pages, file list, `competitions download`, public leaderboard download, kernel list, and discussion threads). This file is the working copy of the official pages. Read it, `official/data-manifest.json`, and `official/public-leaderboard-2026-10-14.csv` before opening the website.

The raw task JSON is **not** in git. The competition rules allow Apache-2.0 use of the data and, in the same section, forbid publishing that data to anyone who has not accepted the rules. A public GitHub repo would be that publication. A participant fetches the files with one command:

```powershell
scripts\download_data.ps1
```

That writes `data/*.json` and leaves them gitignored. Hashes from the 2026-10-14 download are in `official/data-manifest.json`.

## Identity

| | |
|---|---|
| Title | ARC Prize 2026 - ARC-AGI-2 |
| Slug | `arc-prize-2026-arc-agi-2` |
| Website | https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/ |
| Sponsor | ARC Prize Foundation, 548 Market Street #83849, San Francisco, CA 94104 |
| Category | Featured code competition |
| Prize pool | $700,000 |
| Teams | 2,885 (competition list, 2026-10-14) |
| Public leaderboard rows | 2,886 scored rows in the CSV downloaded the same day |
| What it is | A relaunch of ARC Prize 2025. Sibling contests: `arc-prize-2026-arc-agi-3` ($850,000, deadline 2026-11-02, 4,280 teams) and `arc-prize-2026-paper-track` ($450,000, deadline 2026-11-09, 254 teams). |

The task is abstract reasoning on grids the solver has not memorized. A task gives demonstration pairs (`train`) and one or more test inputs. The solver must build the output grid: height, width, then every cell an integer 0–9. Only an exact cell-by-cell match counts. Two attempts are allowed per test output.

## Data files

Filenames below are the ones `kaggle competitions download` actually wrote on 2026-10-14. The website’s data page uses hyphens in a few names (`arc-agi_training-challenges.json`). The zip uses underscores. Trust the zip.

| File | Bytes | Tasks | SHA-256 |
|---|---:|---:|---|
| `arc-agi_training_challenges.json` | 4,010,050 | 1,000 | `779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5` |
| `arc-agi_training_solutions.json` | 658,743 | 1,000 | `9f07a38bd25af5e83aa5bf85c5cb1a1fefdb30f6a755256fa65429e697ca97f9` |
| `arc-agi_evaluation_challenges.json` | 984,679 | 120 | `e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8` |
| `arc-agi_evaluation_solutions.json` | 223,838 | 120 | `84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1` |
| `arc-agi_test_challenges.json` | 1,015,295 | 240 | `232264c58f825ee77327dcfc9f4e5cb2f83b8d997eb69032be1bf2205bbe1a83` |
| `sample_submission.json` | 19,936 | 240 | `6b372dce41ad86a941ccdcbebb2b1926fd4f011d77c824ff421d77730f169a9a` |

No id is shared between training and evaluation. Downloaded test ids do not overlap evaluation either. All 240 downloaded test ids are training ids, and those tasks’ demonstration pairs and test inputs are identical to the training file. The website says the test file shown on the page is a placeholder copied from the evaluation challenges and is swapped, at notebook rerun, for 240 unseen tasks. The CLI zip does not match that sentence: it is a 240-task slice of **training**, still without test outputs. Treat `data/arc-agi_test_challenges.json` from `competitions download` as a placeholder. The scored file is the one Kaggle mounts at rerun.

### Shape

A challenge file is a JSON object keyed by task id.

```json
{
  "task_id": {
    "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
    "test": [{"input": [[2, 0], [0, 2]]}]
  }
}
```

A solutions file maps the same task id to a list of output grids, one per test input, in that order. A test item in the challenge file has `input` only. Grids are rectangular lists of lists of integers 0–9. Official limits are 1×1 through 30×30. The 2026-10-14 files use that full range on both axes.

| Split | Train pairs per task | Tasks with more than one test input |
|---|---|---|
| Training (1,000) | 2–10, mean 3.23 | 69 |
| Evaluation (120) | 2–6, mean 2.98 | 49 |
| Downloaded test placeholder (240) | 2–8, mean 3.20 | 17 |

Most leaderboard tasks have one test input. A minority have two or three, and those predictions must stay in input order. `src/arc.py` loads `data/arc-agi_{split}_challenges.json` and the matching solutions file when it exists.

`official/schema-example.json` is a fake one-cell illustration of the three shapes. It is not a competition task.

## Score and `submission.json`

The score is the fraction of test outputs that are solved. For each test output you submit exactly two grids, `attempt_1` and `attempt_2`. If either grid equals the ground truth cell for cell, that output scores 1, otherwise 0. The competition score is the mean of those 0/1 values. The public leaderboard prints this as a percentage (`88.06` means 88.06% of scored outputs).

The rerun file must be named `submission.json`. Every task id in the challenges file must appear. Both attempts must be present even when you only have one real guess. Multiple test inputs are a list, in input order.

```json
{
  "00576224": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
  "12997ef3": [
    {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]},
    {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}
  ]
}
```

During the contest the public leaderboard is a sample of the test set. Final rank is the private leaderboard. Ties break toward the earlier submission.

## Code-competition contract

Submit a notebook. The Submit button stays disabled unless all of these hold:

- CPU notebook runtime ≤ 12 hours, or GPU notebook runtime ≤ 12 hours
- Internet disabled
- Output file named `submission.json`
- External data and pretrained models are allowed when they are free and public

Displayed runtimes are obfuscated. Resubmitting the same notebook can move the reported time by up to 10 minutes.

### L4x4

This competition can use Kaggle’s L4x4 machines: 96 GB of GPU memory. See Google Cloud’s L4 GPU page for the hardware. Rules on that accelerator:

- L4x4 notebooks consume GPU quota at **twice** the rate of T4x2 or P100. Kaggle may raise that rate to keep scoring capacity.
- L4s are only for notebooks attached to this competition. Circumventing that can mean a team ban or an account ban.
- Every L4 session has internet off.

Written submission cap: **1 submission per day**, and **2 final submissions** selected for judging. Team size max is **5**. Merges are allowed before the merger deadline if the combined submission count does not exceed days-run times the daily cap.

Capacity is tighter than the written daily cap. Kaggle staff (Dustin, topic 747515) said on 2026-10-12 that an L4x4 quota-accounting bug had been fixed and that capacity was about one submission every other day per team. On 2026-10-10 the queue delay was about 33 hours. On 2026-10-13 it had fallen to about 14 hours, with most runs finishing within about 24 hours of submit. Those are staff status reports, not a rewrite of the 1-per-day rule.

## Timeline

All deadlines are 23:59 UTC.

| Date | Event |
|---|---|
| 2026-03-25 | Start |
| 2026-10-26 | Entry deadline. Rules must be accepted by this date. |
| 2026-10-26 | Team merger deadline |
| 2026-11-02 | Final submission deadline |
| 2026-12-04 | Winners announcement |

The organizer can move dates. Twelve days remained after this snapshot before entry and mergers closed, and nineteen days before the final deadline.

## Prizes

Total **$700,000**. A prize-eligible team that does not open-source the solution is removed.

**Progress prizes, $275,000**, by private-leaderboard rank: 75k, 50k, 40k, 35k, 25k, 20k, 15k, 15k for 1st through 8th.

**Innovation prize, $275,000.** The prizes page calls the same award the Grand Prize. It goes to the highest-scoring solution writeup, not automatically to leaderboard rank. Artifacts must be open-sourced and attached to an official solution writeup within seven days of the submission deadline. Six criteria, each 0–5, averaged:

| Criterion | Question |
|---|---|
| Accuracy | Leaderboard performance |
| Universality | Does the method travel to similar problems? |
| Progress | Does it raise the chance anyone reaches 85% on ARC-AGI-2? |
| Theory | Does the writeup explain why it works, not only how? |
| Completeness | Does the writeup cover the leaderboard submission? |
| Novelty | How new is it relative to public research? |

**Bonus, $150,000**, unlocks if any team reaches **85%** on the competition leaderboard. It is then split across the top 5 teams that reached 85%: 75k, 25k, 20k, 20k, 10k. Fewer than five qualifiers share the pool proportionally. On the 2026-10-14 public leaderboard, one team is at or above 85% (Tufa Labs, 88.06).

Winner license: CC BY 4.0, and the open-source definition used by the rules includes system, model, and weights. Pretrained models or commercial software the sponsor can buy without undue cost do not themselves have to be relicensed. Winners must deliver training code, inference code, the compute environment, do a sponsor interview, and sign prize paperwork including tax forms. Prize money inside a team splits evenly unless the team agrees otherwise before payout. Taxes are the winner’s. Governing law is California, Santa Clara County courts.

Eligibility that changes who can be paid: a Kaggle account, at least 18 or the local age of majority, and not a resident of Crimea, the so-called Donetsk or Luhansk People’s Republics, Cuba, Iran, or North Korea, and not a person under U.S. sanctions. Sponsor, Kaggle, and affiliate employees may compete and may not win prizes. One Kaggle account only. No private sharing of code or data outside a team. Public sharing must be on this competition’s forums or notebooks and is treated as OSI-approved and commercial-use-allowed. External data and models are allowed when every participant can get them at no cost, or when the host’s reasonableness test says the cost is not exclusionary. Hand labels of validation or test records are explicitly permitted by the general rules.

Data terms, both of them: access and use are Apache 2.0, including commercial and academic use. Separately, you must keep the files away from anyone who has not accepted these rules, and you must not publish or redistribute the Competition Data to a non-participant. That second sentence is why `data/` is gitignored on this public repo.

## Public leaderboard, 2026-10-14

Downloaded file: `official/public-leaderboard-2026-10-14.csv` (rank, team id, team name, score, submission count, last submission time). This is the public board, not the private board. The September note in this repo that called 83.06 a private score is out of date.

| Rank | Team | Score | Last submission (UTC) |
|---:|---|---:|---|
| 1 | Tufa Labs | 88.06 | 2026-10-12 22:54 |
| 2 | Yi-Chia Chen | 82.64 | 2026-10-12 13:48 |
| 3 | rabbithole | 80.56 | 2026-10-13 15:22 |
| 4 | nvbanana | 77.08 | 2026-10-12 20:49 |
| 5 | _hans | 69.31 | 2026-10-12 21:10 |
| 6 | ALbert RUehlman | 67.36 | 2026-10-13 03:01 |
| 7 | Dawid Kopiczko | 66.25 | 2026-10-12 00:05 |
| 8 | Domdolus Tolus | 65.14 | 2026-10-13 11:51 |
| 9 | ARC Raiders | 64.58 | 2026-10-12 16:48 |
| 10 | Majkel1337 | 62.08 | 2026-10-13 22:48 |
| 11 | Richard Csaky | 60.00 | 2026-10-12 20:14 |
| 12 | matu997 | 54.31 | 2026-10-11 14:46 |

Distribution of the 2,886 public scores: **1** team ≥ 85, **14** teams ≥ 50, **64** teams ≥ 33.89, **1,925** teams ≥ 10. Median 28.89. 90th percentile 31.81.

So 33.89, the best score written on a widely forked public notebook, is no longer the public ceiling. Sixty-four teams are at or above it. The teams above that line have not published a notebook whose title or kernel metadata carries a higher score.

## Public notebooks

`kaggle kernels list --competition arc-prize-2026-arc-agi-2 --sort-by scoreDescending` on 2026-10-14 did **not** return a score field, and the order matched vote count. Do not read that order as a leaderboard. Scores below are the ones printed by the authors on the notebook or in a discussion that day.

Already in `notebooks/` before this refresh: the LB 33.89 notebook and its replica, Land’s public solver, NVARC 2025 and a reproduction, CompressARC, the 2024 DSL baseline, an enhanced Qwen notebook, and a starter.

Pulled on 2026-10-14:

| Folder | Source | What it is |
|---|---|---|
| `notebooks/failed-in-aimo` | `koushikrudra/failed-in-aimo` (873 votes) | The community base that later notes call LB 33.89. GPU notebook, internet off, model `sorokin/qwen3_4b_grids15_sft139`. |
| `notebooks/learned-from-aimo` | `chiakazirim/learned-from-aimo` (212 votes) | Same loader/decoder family as the notebook above. |
| `notebooks/arc-baseline-rebuild` | `yusuketogashi/arc-baseline-rebuild` (95 votes, run 2026-10-13) | Author text: base is failed-in-aimo at LB 33.89, plus deterministic scoring (`PYTHONHASHSEED=0`, `zlib.crc32` instead of salted `hash`) and a GPU NLL path. |
| `notebooks/arc-agi-31-11` | `junaid512/arc-agi-31-11` (147 votes, run 2026-10-13) | Same NVARC-style loader/decoder. Title states 31.11, under 33.89. |
| `notebooks/arc-agi2-nvarc-plus-ckpt` | `dalezhong/arc-agi2-nvarc-plus-ckpt` (run 2026-10-12) | NVARC+ v3: Qwen3-4B, 16-token grid vocabulary, per-task LoRA test-time training, DFS, three passes. |
| `notebooks/arc-qwen-3-8-27b-agent` | `madarshbb/arc-qwen-3-8-27b-agent` | REPL program-synthesis agent on Qwen3.8-27B FP8. Author’s discussion posts: about 15, then **21.53** public, with a rerun dropping to 13. |
| `notebooks/arc-agi-2-agentic-multi-turn-reasoning` | `khoinguyennguyen/arc-agi-2-agentic-multi-turn-reasoning` | One persistent Python agent per task. Shared from the Qwen3.8 thread as a starting point; no score was attached. |

The highest public score anyone documented in that day’s Qwen3.8 thread (`madarshbb`, topic 748082) is 21.53, and the author said a small change dropped a rerun to 13. Nobody in the fetched threads posted a public notebook above the 33.89 line.

## Discussions worth knowing

Fetched 2026-10-14. Participant posts are claims. Dustin’s posts are Kaggle staff.

| Id | Date | What is settled |
|---|---|---|
| 747515 | 2026-10-08, updates through 10-13 | “Kaggle Error” and queue delays. Staff: L4x4 quota bug fixed; about one submission every two days of capacity; queue ~33h on Oct 10, ~14h on Oct 13. |
| 689054 | 2026-04-07, still active | The L4 upgrade thread. Staff added capacity on 2026-09-29 after the queue jammed. Later posts are the same outage. |
| 748082 | 2026-10-10 | Qwen3.8-27B inside Kaggle limits. A public agent was reported at ~15, then 21.53, high variance. Link shared: `madarshbb/arc-agent-sub-q38-direct-b98k-c16-792ef9` (version 358248812). |
| 749035 | 2026-10-13 | “Pareto front says nvbanana is likely to win.” nvbanana is public rank 4 at 77.08. A teammate reply only said their submissions run fast. Not a method disclosure. |
| 749022 | 2026-10-13 | “How much can you trust evaluation score.” Title only; the thread body did not come back in the first comment page. |
| 748895 | 2026-10-13 | “Tried fine-tuning NVARC on new synthetic ARC tasks.” Title only; no comments on the first page. |
| 748741 | 2026-10-12 | Whether Codex or Claude synthetic puzzles are allowed. The only reply refused to give legal advice. The written rule still governs: external data must be public to every participant or pass the host’s cost test. |
| 748766 | 2026-10-13 | Argument about cutting the submission cap because Gemma 4 Developer Agent also uses L4s. No rule change. The written cap remains 1/day, and staff capacity comments are in 747515. |

## Local starter

`src/arc.py` loads tasks, checks grids, writes a two-attempt submission, and scores it when solutions exist. `src/baseline_solver.py` is a small heuristic (identity, color map, crop, tile, symmetry), on the order of 1% locally. `src/evaluate.py` and `src/make_submission.py` are the commands. Generated submissions stay in `submissions/`, which is gitignored.

```powershell
python src\evaluate.py --split evaluation
python src\make_submission.py
```

Strategy notes that are not official pages live under `reference/`: `harness-first-qwen38-27b.md` is the current plan, with the older SFT/TTT notes and the vendored Poetiq and Land solvers beside it. Those scores (Poetiq 54.0, Land 72.9 / 76.1) are from the solver writeups, not from this leaderboard snapshot.

## Refresh

```powershell
kaggle competitions pages arc-prize-2026-arc-agi-2 list --content --format json
kaggle competitions files arc-prize-2026-arc-agi-2 --format json
kaggle competitions leaderboard arc-prize-2026-arc-agi-2 -d -p .\official -q
kaggle kernels list --competition arc-prize-2026-arc-agi-2 --sort-by voteCount --page-size 20
kaggle competitions topics list arc-prize-2026-arc-agi-2 --sort-by recent
```

Re-check topic 747515 before trusting the queue numbers. They moved twice in five days.
