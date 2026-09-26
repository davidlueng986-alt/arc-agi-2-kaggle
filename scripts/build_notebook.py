#!/usr/bin/env python
"""Build notebook/arc26_h3_code_verify.ipynb from the harness sources.

The Kaggle notebook must be self-contained: every harness module is embedded
as a %%writefile cell, followed by the vLLM launcher and the solver runner.
Re-run this script after any harness change and re-push the kernel:
    python scripts/build_notebook.py
    kaggle kernels push -p notebook
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "harness"
OUT = ROOT / "notebook" / "arc26_h3_code_verify.ipynb"

MODULES = [
    "types.py",
    "grid.py",
    "extraction.py",
    "_sandbox_worker.py",
    "sandbox.py",
    "oracle.py",
    "pool.py",
    "disambiguate.py",
    "prompts.py",
    "llm.py",
    "budget.py",
    "submission.py",
    "data.py",
    "solver.py",
    "__init__.py",
]

MD_INTRO = """# ARC Prize 2026 — H3: Program Synthesis + Verified-Set Disambiguation (Qwen3.8-27B)

**Hypothesis (see HYPOTHESIS.md in the repo):** Qwen3.8-27B's coding is frontier-class
(SWE-bench Pro 61.7 > Opus4.6 Max 53.4) while its abstract reasoning is not (HLE 30.8 vs 40.0).
So we never ask it for answer grids — we ask it for `def transform(grid)` programs, verify them
exactly against the train pairs (a free deterministic oracle), and resolve the **disagreement set**
(verified programs that disagree on the test input) with a principled, zero-token layer:

- **S1 symmetry equivariance** — D4 + data-plausible color-permutation pseudo-labels, two-stage gated
  (data consistency, then pool-majority resolvability), train-side falsification + test-side self-consistency
- **S1c unanimous shape-law check** — programs contradicting every train-consistent shape law are penalized
- **S2 LOO corroboration** (gated) — held-out-fold resynthesis; passing a withheld pair is direct generalization evidence
- **S3 MDL prior + S3b literal-memorization screen** — AST size; embedded train grids are disqualifying
- **S4 behavioral clustering** on perturbed test inputs — independent-agreement support
- **S5 soft fallback** — verification failure never discards (Land: precision ~97%, recall ~43%)
- **S6 determinism check**, **S7 adjudication call** (gated), **pass@2 shape-diverse selection**

**Attachments required:** competition data (`arc-prize-2026-arc-agi-2`) and a Qwen3.8-27B FP8 model.
Attach any FP8 mirror in the Kaggle UI (Add Input → Models); the runner auto-detects the mount path
from candidates including `qwen3-8-27b-fp8` (michaelpoluektov / hark99), `qwen3-8-27b-fp8-official`
(mikedan7), `qwen-qwen3-8-27b-fp8` (dangkhoa2016). Do NOT use NVFP4 mirrors (Blackwell-only; L4 is Ada sm_89).
The `model_sources` field is intentionally left empty in `kernel-metadata.json` because the CLI requires a
full instance-version path that resolves only from the model owner's account — attach the model in the UI once.

**GPU note:** metadata keeps `enable_gpu: false` per branch policy. The runner auto-detects CUDA:
with GPUs it launches vLLM (TP=4) and runs the full pipeline; without GPUs it falls back to a
built-in template MockLLM so the notebook still produces a schema-valid submission and exercises
the whole harness (useful smoke mode). Rerun wall clock: 12h with cumulative atomic
`/kaggle/working/submission.json` writes after every task + a watchdog."""

CODE_SETUP = '''import os, sys, json, time, shutil, subprocess

START_TS = time.time()
WORKING = "/kaggle/working"
H3 = os.path.join(WORKING, "h3")
os.makedirs(os.path.join(H3, "harness"), exist_ok=True)

# 12h rerun limit minus a 10-minute safety buffer
END_TS = START_TS + 12 * 3600 - 600

RERUN = os.getenv("KAGGLE_IS_COMPETITION_RERUN", "") != ""

DATA_DIR = None
for p in (
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
    "/kaggle/input/arc-prize-2026-arc-agi-2",
):
    if os.path.isdir(p):
        DATA_DIR = p
        break

MODEL_DIR = None
for p in (
    "/kaggle/input/qwen3-8-27b-fp8",
    "/kaggle/input/qwen3-8-27b-fp8-official",
    "/kaggle/input/qwen-qwen3-8-27b-fp8",
    "/kaggle/input/qwen3-8-27b-fp8-hf-snapshot",
    "/kaggle/input/qwen3-8-27b-bf16",
    "/kaggle/input/qwen3-8-27b",
):
    if os.path.isdir(p):
        MODEL_DIR = p
        break

def _has_gpu():
    try:
        if shutil.which("nvidia-smi") is None:
            return False
        return subprocess.call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    except Exception:
        return False

HAS_GPU = _has_gpu()
print("rerun mode:", RERUN)
print("data dir:", DATA_DIR)
print("model dir:", MODEL_DIR)
print("gpu available:", HAS_GPU)
print("wall budget (s):", END_TS - START_TS)
'''

CODE_IMPORT = '''sys.path.insert(0, H3)
import harness
from harness.sandbox import Sandbox
from harness.llm import VLLMClient, MockLLM, library_responder

# sandbox smoke test (subprocess isolation + numpy worker)
_sb = Sandbox(per_job_timeout=2.0)
_r = _sb.run("import numpy as np\\n\\ndef transform(grid):\\n    return grid + 1\\n", [[0, 1], [2, 3]])
assert _r.ok, _r.error
_sb.close()
print("harness import + sandbox OK:", harness.__name__)
'''

MD_VLLM = """## Launch vLLM (offline, TP=4) or fall back to the CPU template MockLLM

Verified launch line for Qwen3.8 (traps handled): `--reasoning-parser qwen3`,
`--tool-call-parser qwen3_coder`, `--enable-prefix-caching` with `--mamba-cache-mode align`
(hybrid Gated DeltaNet), NO speculative decoding (breaks constrained decoding), and the client
pins `reasoning_effort` to low/medium — `high` raises HTTP 500 (QwenLM/Qwen3.8#217) and the
default `xhigh` would emit 8-16K thinking tokens per call (~4x over budget)."""

CODE_VLLM = '''import json as _json, time as _time, urllib.request, urllib.error

VLLM_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "qwen3.8-27b"
LLM = None
_vllm_proc = None

def _server_ready(timeout_s=60):
    t0 = _time.time()
    while _time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(VLLM_URL + "/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            _time.sleep(5)
    return False

def _launch_vllm(max_model_len):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_DIR,
        "--served-model-name", MODEL_NAME,
        "--tensor-parallel-size", "4",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", "0.90",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--enable-prefix-caching", "--mamba-cache-mode", "align",
        "--port", "8000",
    ]
    logf = open(os.path.join(WORKING, "vllm.log"), "wb")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)

if HAS_GPU and MODEL_DIR is not None:
    _vllm_proc = _launch_vllm(262144)
    if not _server_ready(timeout_s=1800):
        # KV/activation headroom issue at 256K context -> retry smaller
        try:
            _vllm_proc.kill()
        except Exception:
            pass
        print("first launch failed; retrying with --max-model-len 32768 (see vllm.log)")
        _vllm_proc = _launch_vllm(32768)
        if not _server_ready(timeout_s=1800):
            raise RuntimeError("vLLM failed to become ready; inspect /kaggle/working/vllm.log")
    LLM = VLLMClient(base_url=VLLM_URL, model=MODEL_NAME, request_timeout=1200.0, max_retries=3)
    _probe = LLM.chat(
        [{"role": "user", "content": "Reply with exactly: OK"}],
        temperature=0.0, max_tokens=512, reasoning_effort="low",
    )
    print("vLLM smoke reply:", _probe.text[:120].replace("\\n", " "))
else:
    print("No GPU/model available -> CPU fallback mode (template MockLLM).")
    print("The notebook still produces a schema-valid submission via the oracle fallback path.")
    LLM = MockLLM(library_responder())
'''

CODE_RUN = '''from harness.budget import Budget, Watchdog
from harness.data import load_challenges, tasks_from_challenges
from harness.disambiguate import DisambiguationConfig
from harness.solver import SolverConfig, solve_all
from harness.submission import SubmissionWriter, validate_submission

assert DATA_DIR is not None, "competition data not attached"
test_file = "arc-agi_test_challenges.json" if RERUN else "arc-agi_evaluation_challenges.json"
challenges = load_challenges(os.path.join(DATA_DIR, test_file))
tasks = tasks_from_challenges(challenges)
print(f"loaded {len(tasks)} tasks from {test_file}")

SUB_PATH = os.path.join(WORKING, "submission.json")
writer = SubmissionWriter(SUB_PATH, tasks)   # writes all-task fallback immediately

budget = Budget(
    deadline_ts=END_TS - 300,          # stop starting tasks 5 min before the buffer edge
    per_task_tokens=75_000,            # HYPOTHESIS.md budget: ~57K raw prompt + ~13.5K completion
    total_tokens=24_000_000,
    reserve_seconds=0.0,
)

def _finalize():
    try:
        writer.flush()
        with open(SUB_PATH) as f:
            data = json.load(f)
        problems = validate_submission(data, tasks)
        print("[watchdog] final submission valid:", not problems)
    except Exception as e:
        print("[watchdog] finalize error:", e)

watchdog = Watchdog(END_TS - 120, _finalize, hard_kill_grace_s=120)
watchdog.start()

cfg = SolverConfig(
    num_experts=2,
    max_iterations=5,
    stop_distinct_behaviors=4,
    patience_no_new=3,
    sandbox_timeout=3.0,
    use_adjudication=True,
    use_loo=True,
    loo_max_folds=3,
    seed=20260326,
    disambiguation=DisambiguationConfig(seed=20260326),
)

_t0 = time.time()
summary = solve_all(
    tasks, LLM, budget, cfg,
    submission_writer=writer,
    workers=4,
    executor_factory=lambda: Sandbox(per_job_timeout=3.0, startup_timeout=60.0),
    on_task=lambda r: print(f"  [{r.task_id}] cands={r.n_candidates} verified={r.n_verified} "
                            f"behaviors={r.n_distinct_verified_behaviors} calls={r.llm_calls} "
                            f"tok={r.prompt_tokens + r.completion_tokens} wall={r.wall_s:.0f}s", flush=True),
)
watchdog.cancel()
_finalize()
print("solve_all done in", round(time.time() - _t0, 1), "s")
print({k: v for k, v in summary.items() if k != "results"})
print("budget:", budget.snapshot())
'''

CODE_VALIDATE = '''import json as _json

with open(SUB_PATH) as f:
    _data = _json.load(f)
_problems = validate_submission(_data, tasks)
print("submission path:", SUB_PATH)
print("task_ids present:", len(_data), "of", len(tasks))
print("schema problems:", _problems[:5] if _problems else "NONE - valid")
_n_att = sum(len(v) for v in _data.values())
print("attempt packs:", _n_att)
assert not _problems, "submission schema invalid"
'''


def md_cell(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def writefile_cell(rel_path: str, content: str) -> dict:
    header = f"%%writefile /kaggle/working/h3/{rel_path}\n"
    return code_cell(header + content.rstrip("\n") + "\n")


def main() -> None:
    cells = [md_cell(MD_INTRO), code_cell(CODE_SETUP)]
    for mod in MODULES:
        src = (HARNESS / mod).read_text(encoding="utf-8")
        cells.append(writefile_cell(f"harness/{mod}", src))
    cells.append(code_cell(CODE_IMPORT))
    cells.append(md_cell(MD_VLLM))
    cells.append(code_cell(CODE_VLLM))
    cells.append(code_cell(CODE_RUN))
    cells.append(code_cell(CODE_VALIDATE))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT} ({len(cells)} cells, {OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
