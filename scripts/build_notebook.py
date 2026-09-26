"""Assemble notebook/arc26_h4_poetiq_judge.ipynb from the harness sources.

The Kaggle notebook must be self-contained (offline rerun), so every harness
module is emitted as a %%writefile cell, followed by a vLLM launcher + runner
cell built on harness.kaggle_main, and a final validation cell.

Usage:  python scripts/build_notebook.py
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS_DIR = os.path.join(ROOT, "harness")
OUT_DIR = os.path.join(ROOT, "notebook")
OUT_PATH = os.path.join(OUT_DIR, "arc26_h4_poetiq_judge.ipynb")

MODULE_ORDER = [
    "__init__.py",
    "types.py",
    "grids.py",
    "parsing.py",
    "budget.py",
    "sandbox.py",
    "oracle.py",
    "pool.py",
    "prompts.py",
    "llm.py",
    "experts.py",
    "judge.py",
    "selection.py",
    "submission.py",
    "config.py",
    "runner.py",
    "kaggle_main.py",
]

MD_INTRO = """# ARC Prize 2026 (ARC-AGI-2) — H4: Verified Ensemble + Holistic Judge, pass@2-optimised

Ports the two ARC-Prize-**verified** harness mechanisms to a **locally-served Qwen3.8-27B** (no external APIs ⇒ Kaggle-eligible):

1. **Poetiq code-refinement engine** (54.0% semi-private, verified): 2 experts (8+4 iterations), pool-5 scored feedback, shuffled train order per iteration, sandbox verification, early-exit on full train match (precision ~97%).
2. **Land holistic trace judge** (72.9% semi-private): ONE long-context prompt with ALL candidate traces per test input; council of 3 samples at `reasoning_effort=medium`; 2 pts per first choice / 1 pt per second choice; may **synthesize** a new grid.
3. **pass@2-aware attempt selection**: shape-prior repair hedge + source decorrelation (attempt_2 maximises conditional coverage, not rank).

Budget: ~99K GPU-effective tokens/task worst case, ~60–80K expected; degradation ladder at 60/75/88/93% of wall clock; cumulative `submission.json` write after every task; watchdog finalises 60 s before the deadline (12 h − 600 s buffer pattern).

**Required attachments** (offline rerun — internet is disabled):
- Competition data: `arc-prize-2026-arc-agi-2`
- Model dataset: `saltb0x/qwen3-8-27b-fp8` (25.3 GB FP8; Ada sm_89-safe — do NOT use NVFP4 mirrors)
- vLLM: preinstalled in the current Kaggle image; if missing, attach a vLLM wheelhouse dataset and the notebook installs it `--no-index`.

If GPUs / model / vLLM are unavailable, the notebook still writes a schema-valid fallback submission (identity + modal-colour grids). **Never lose the submission.**

See `HYPOTHESIS.md` in the repo for the full claim, budget arithmetic and ablation design."""

CELL_SETUP = """# --- H4 setup: deadline (12h - 600s buffer, lb3389 pattern), paths, package dir ---
import os, sys, time, json

NOTEBOOK_START = time.time()
GLOBAL_END_TIME = NOTEBOOK_START + 12 * 3600 - 600  # 600s submission/write buffer
os.makedirs("/kaggle/working/harness", exist_ok=True)
if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

rerun_mode = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
comp_dir = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
challenges_path = os.path.join(
    comp_dir,
    "arc-agi_test_challenges.json" if rerun_mode else "arc-agi_evaluation_challenges.json",
)
print("rerun_mode:", rerun_mode)
print("challenges:", challenges_path, os.path.exists(challenges_path))
print("global end time in", round(GLOBAL_END_TIME - time.time()), "s")"""

CELL_IMPORT_CHECK = """# --- sanity: import the harness written above ---
import harness
from harness import kaggle_main
from harness.config import PipelineConfig
from harness.submission import validate_submission
print("harness import OK:", harness.__file__)
model_dir = kaggle_main.find_model_dir()
print("model dir:", model_dir)
print("vllm importable:", kaggle_main.have_vllm())"""

CELL_RUN = """# --- main run: vLLM launch (TP=4, offline) + experts + judge + watchdog ---
# Blocks until the run finishes or the deadline ladder finalizes the submission.
# Falls back to a schema-valid placeholder submission when GPUs/model/vLLM are
# unavailable, so the kernel always produces /kaggle/working/submission.json.
from harness import kaggle_main

total_time_s = max(600.0, GLOBAL_END_TIME - time.time())
summary = kaggle_main.main(total_time_s=total_time_s)
print("mode:", summary.get("mode"))"""

CELL_VALIDATE = """# --- final validation (always runs; guarantees the contract even if the cell
# above crashed): every task_id present, one {attempt_1, attempt_2} per test
# input in order, rectangular int grids, both attempts non-empty ---
import json, os
from harness.submission import SubmissionWriter, validate_submission

comp_dir = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
rerun_mode = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
challenges_path = os.path.join(
    comp_dir,
    "arc-agi_test_challenges.json" if rerun_mode else "arc-agi_evaluation_challenges.json",
)
with open(challenges_path) as f:
    challenges = json.load(f)

sub_path = "/kaggle/working/submission.json"
if os.path.exists(sub_path):
    with open(sub_path) as f:
        sub = json.load(f)
else:
    sub = {}

writer = SubmissionWriter(sub_path, challenges)
writer.submission = {k: v for k, v in sub.items() if k in challenges and v}
res = writer.finalize()
print("validation errors:", res["errors"])
assert not res["errors"], res["errors"][:5]

# score against ground truth when available (interactive / eval-split runs)
sol_path = os.path.join(comp_dir, "arc-agi_evaluation_solutions.json")
if not rerun_mode and os.path.exists(sol_path):
    from harness.runner import score_attempts
    with open(sol_path) as f:
        solutions = json.load(f)
    hits = n = 0
    for tid, outs in solutions.items():
        if tid in challenges and tid in sub:
            h2, _h1, k = score_attempts(sub[tid], outs)
            hits += h2
            n += k
    print(f"local eval-split pass@2: {hits}/{n} = {hits / max(n, 1) * 100:.2f}%")
print("submission written:", sub_path, os.path.getsize(sub_path), "bytes")"""

MD_FOOTER = """## Local development & ablations (outside Kaggle)

The identical harness runs against any OpenAI-compatible endpoint or the deterministic MockLLM:

```bash
python -m pytest tests -q                          # 157 tests, no GPU needed
python scripts/eval_local.py --mock --limit 10     # pipeline dry run
python scripts/eval_local.py --base-url http://127.0.0.1:8000/v1   # real model

# ablations (each component's contribution is measurable):
python scripts/eval_local.py --no-judge            # judge on/off
python scripts/eval_local.py --no-diversity        # pass@2 selection on/off
python scripts/eval_local.py --experts 1 --pool-size 1   # chain-refinement (Land's failure mode)
python scripts/eval_local.py --experts 2 --pool-size 5   # population refinement (Poetiq)
```"""


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def writefile_cell(module: str) -> dict:
    path = os.path.join(HARNESS_DIR, module)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    src = f"%%writefile /kaggle/working/harness/{module}\n" + content
    return code_cell(src)


def build() -> dict:
    cells = [md_cell(MD_INTRO), code_cell(CELL_SETUP)]
    for module in MODULE_ORDER:
        cells.append(writefile_cell(module))
    cells.append(code_cell(CELL_IMPORT_CHECK))
    cells.append(code_cell(CELL_RUN))
    cells.append(code_cell(CELL_VALIDATE))
    cells.append(md_cell(MD_FOOTER))
    return {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells,
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    nb = build()
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    # guard: no source line may start with a foreign magic that Jupyter would execute
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = cell["source"]
        first = lines[0].strip() if lines else ""
        assert first.startswith("%%writefile") or not first.startswith(("%%", "%", "!")), first
        for line in lines[1:]:
            assert not line.startswith("%%"), line[:60]
    print(f"wrote {OUT_PATH}: {len(nb['cells'])} cells, "
          f"{os.path.getsize(OUT_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
