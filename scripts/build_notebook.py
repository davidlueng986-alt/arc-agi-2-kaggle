#!/usr/bin/env python3
"""Build notebook/arc26_h2_duck_repl.ipynb from the harness source + cell files.

The notebook is generated, never hand-edited, so the code that runs on Kaggle is
byte-identical to the code the test suite covers. The harness package is zipped
and base64-embedded into cell 1, which makes the notebook self-contained: the
rerun has no internet, and depending on a separately-attached source dataset is
one more thing that can silently go wrong.

    python scripts/build_notebook.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HARNESS_DIR = os.path.join(ROOT, "harness")
CELLS_DIR = os.path.join(ROOT, "notebook", "cells")
OUT_DIR = os.path.join(ROOT, "notebook")
NOTEBOOK_PATH = os.path.join(OUT_DIR, "arc26_h2_duck_repl.ipynb")
METADATA_PATH = os.path.join(OUT_DIR, "kernel-metadata.json")

OWNER_SLUG = "liangwanyiudavid"
KERNEL_SLUG = "arc26-h2-duck-repl"
# Kaggle derives the kernel slug from the title, and rejects a push whose title
# does not resolve to the declared id (HTTP 400). "Arc26 H2 Duck Repl" ->
# "arc26-h2-duck-repl". The descriptive title lives in the notebook's first
# markdown cell instead.
TITLE = "Arc26 H2 Duck Repl"
DOC_TITLE = (
    "ARC Prize 2026 (H2) ??Duck-REPL Transfer: an agentic exact-oracle REPL loop "
    "for ARC-AGI-2 on Qwen3.8-27B"
)
COMPETITION_SLUG = "arc-prize-2026-arc-agi-2"
# Kaggle's notebook metadata wants NUMERIC ids for data sources, not slugs.
# Passing the slug is silently ignored and the kernel boots with an empty
# /kaggle/input -- which is exactly what the first two pushes did. These two
# numbers are copied from the working arc-prize-2026-arc-agi-2 notebooks vendored
# under notebooks/ (arc-agi-2-baseline-starter, arc-highscore-lb3389-replica,
# reproduce-nvarc-2025-results all agree on them).
COMPETITION_SOURCE_ID = 133469
COMPETITION_DATABUNDLE_VERSION_ID = 16010536
MODEL_DATASETS = ["saltb0x/qwen3-8-27b-fp8"]
# Set by --gpu. The accelerator string for 4x L4 should be confirmed in the
# Kaggle UI before the real rerun; `enable_gpu` defaults to False on purpose.
ACCELERATOR_GPU = "nvidiaL4x4"

MODULES = [
    "__init__.py",
    "oracle.py",
    "segmentation.py",
    "grids.py",
    "render.py",
    "sandbox.py",
    "llm.py",
    "prompts.py",
    "context.py",
    "budget.py",
    "config.py",
    "submission.py",
    "dataio.py",
    "agent.py",
    "runner.py",
    "mock_agent.py",
]


def build_payload() -> tuple[str, int, int]:
    buffer = io.BytesIO()
    raw = 0
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in MODULES:
            path = os.path.join(HARNESS_DIR, name)
            if not os.path.isfile(path):
                raise SystemExit(f"missing harness module: {path}")
            with open(path, "rb") as handle:
                data = handle.read()
            raw += len(data)
            archive.writestr(f"harness/{name}", data)
        for extra in ("docs/measured_context_budget.json",):
            path = os.path.join(HARNESS_DIR, extra.replace("/", os.sep))
            if os.path.isfile(path):
                archive.writestr(f"harness/{extra}", open(path, "rb").read())
    blob = buffer.getvalue()
    return base64.b64encode(blob).decode("ascii"), len(blob), raw


def read_cell(name: str) -> str:
    path = os.path.join(CELLS_DIR, name)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def code_cell(source: str) -> dict:
    lines = source.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"_kg_hide-input": False},
        "outputs": [],
        "source": lines,
    }


def markdown_cell(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


INTRO = f"""# {DOC_TITLE}

**Hypothesis branch H2 ??"Duck-REPL Transfer"** (`h2/duck-repl`).

The claim: the mechanism that won ARC-AGI-3 Milestone 1 ??The Duck harness (Tufa Labs),
a tool-using REPL loop where *"the model never presses a key. It writes Python, and the
Python presses keys"* ??transfers to ARC-AGI-2, because ARC-AGI-2 ships **ground-truth
train outputs**, which means an exact, free, deterministic oracle is available at
inference time. The Duck's `action(actions)` host round-trip becomes `verify(fn)`: the
agent writes a candidate transform, executes it against every train pair, gets exact
pass/fail plus diff diagnostics, and iterates.

Supporting evidence: arXiv 2607.15439 ablated exactly this on ARC-AGI-3 and found the
**verification variant ranked first in every setting**. Its cost was 2.8x the baseline
(90.75M vs 32.32M cost-tokens at xhigh); this design lands at ~35.5M for 240 tasks
because the oracle is exact (so it *terminates* search rather than extending it), the
context is ~9.6K tokens/turn rather than ~25-30K, and `reasoning_effort` is pinned off
the `xhigh` default.

Full reasoning, measured context costs, the budget arithmetic and the refutation
criteria are in `HYPOTHESIS.md` in the source repository.

### What this notebook does

1. writes the embedded harness package to `/kaggle/working/h2`
2. installs vLLM from an attached **offline wheelhouse** (no internet at rerun)
3. locates the Qwen3.8-27B weights among the attached inputs, rejecting NVFP4 mirrors
   (NVFP4 is Blackwell-native; L4 is Ada sm_89)
4. launches `vllm serve` with `TP=4`, `--reasoning-parser qwen3`,
   `--enable-auto-tool-choice --tool-call-parser qwen3_coder`,
   `--enable-prefix-caching --mamba-cache-mode align`
5. runs the agentic REPL loop over every test challenge with 16 concurrent workers
6. writes `/kaggle/working/submission.json` **cumulatively and atomically**, pre-filled
   with valid fallbacks for all task ids before the first task starts, and finalised by
   a watchdog before the 600s buffer

### Important

`kernel-metadata.json` ships with **`"enable_gpu": false`** so this notebook can be
pushed and smoke-tested without consuming GPU quota. In that state there is no vLLM
server, so the run falls back to a deterministic mock agent: it still produces a
**schema-valid submission** and exercises every non-LLM code path, but its score is
meaningless. **Flip `enable_gpu` to `true` and attach a vLLM wheelhouse dataset for the
real rerun.**
"""

CELL_NOTES = {
    "c01_env.py": "## 1. Environment, deadline and embedded harness source\n\n"
    "Detects the competition rerun, counts GPUs without importing torch, computes the "
    "hard deadline (12h minus the 600s buffer), and unzips the harness package that is "
    "base64-embedded at the bottom of this cell.",
    "c02_install.py": "## 2. Offline runtime install\n\n"
    "No internet at rerun, so vLLM has to come from an attached wheelhouse dataset "
    "(`pip install --no-index --find-links ...`, the same pattern the Duck's Kaggle "
    "notebook uses for the ARC runtime). If neither vLLM nor a GPU is present the "
    "notebook degrades to the mock agent instead of failing.",
    "c03_model.py": "## 3. Locate the model weights\n\n"
    "Scans every mounted input for a HF-style Qwen3 snapshot rather than hardcoding a "
    "path, so the weights dataset can be swapped without editing the notebook. Prefers "
    "FP8 (~25.3 GB) and **rejects NVFP4 outright**: NVFP4 is Blackwell-native and L4 is "
    "Ada (sm_89).",
    "c04_serve.py": "## 4. Launch vLLM\n\n"
    "The official Qwen3.8 serving pattern. Flags are probed against `--help` first "
    "because vLLM renames them between releases and the rerun cannot check online. "
    "`--mamba-cache-mode align` is required for prefix caching on this GatedDeltaNet "
    "hybrid. Speculative decoding / MTP is left **off** because it breaks constrained "
    "decoding. `--max-model-len 32768` rather than 262144: the measured context ceiling "
    "is a 6.4K prefix plus a ~5.2K rolling window, and a lower cap lets vLLM schedule "
    "more concurrent sequences, which is what makes the decode throughput reachable.",
    "c05_run.py": "## 5. Run the agentic REPL loop\n\n"
    "14 turns per task target (24 is the break-even at planned serving rates), 16 "
    "concurrent workers, ~9.6K prompt tokens per turn. Per-task time and turn allowances "
    "are recomputed from `remaining x workers / remaining_tasks` before every dispatch, "
    "so a slow start shortens later tasks instead of running out of clock. `xhigh` is "
    "never used ??it would cost 28h of decode alone.",
    "c06_validate.py": "## 6. Validate and forecast\n\n"
    "Re-validates the file on disk against the challenge file itself (every task id, "
    "both attempts, ints 0-9, rectangular), repairs in place if anything is wrong, and "
    "turns the measured per-turn cost into a 240-task wall-clock forecast with headroom.",
}


def build_notebook(enable_gpu: bool = False) -> dict:
    payload, zipped, raw = build_payload()
    print(f"harness payload: {raw:,} bytes source -> {zipped:,} zipped -> {len(payload):,} base64")

    cells = [markdown_cell(INTRO)]
    for name in ("c01_env.py", "c02_install.py", "c03_model.py", "c04_serve.py", "c05_run.py", "c06_validate.py"):
        source = read_cell(name)
        if name == "c01_env.py":
            if "__HARNESS_ZIP_B64__" not in source:
                raise SystemExit("c01_env.py is missing the __HARNESS_ZIP_B64__ placeholder")
            source = source.replace("__HARNESS_ZIP_B64__", payload)
        cells.append(markdown_cell(CELL_NOTES[name]))
        cells.append(code_cell(source))

    return {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": ACCELERATOR_GPU if enable_gpu else "none",
                "dataSources": [
                    {
                        "sourceType": "competition",
                        "sourceId": COMPETITION_SOURCE_ID,
                        "databundleVersionId": COMPETITION_DATABUNDLE_VERSION_ID,
                        "isSourceIdPinned": False,
                    }
                ],
                "isGpuEnabled": bool(enable_gpu),
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.11.0",
                "pygments_lexer": "ipython3",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "nbconvert_exporter": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def build_metadata(enable_gpu: bool = False) -> dict:
    return {
        "id": f"{OWNER_SLUG}/{KERNEL_SLUG}",
        "title": TITLE,
        "code_file": os.path.basename(NOTEBOOK_PATH),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": bool(enable_gpu),
        "enable_internet": False,
        "keywords": [],
        "category_ids": [],
        "competition_data_sources": [COMPETITION_SLUG],
        "dataset_data_sources": MODEL_DATASETS,
        "kernel_data_sources": [],
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="emit metadata with enable_gpu=true and an L4x4 accelerator. "
        "The committed default is False so pushing this branch cannot consume GPU quota.",
    )
    args = parser.parse_args(argv)

    notebook = build_notebook(enable_gpu=args.gpu)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(build_metadata(enable_gpu=args.gpu), handle, indent=2)
        handle.write("\n")

    size = os.path.getsize(NOTEBOOK_PATH)
    print(f"wrote {NOTEBOOK_PATH}  ({size / 1e6:.2f} MB, {len(notebook['cells'])} cells)")
    print(f"wrote {METADATA_PATH}  (enable_gpu={args.gpu})")

    # Round-trip check: the notebook must be loadable and its cells compilable.
    with open(NOTEBOOK_PATH, encoding="utf-8") as handle:
        reloaded = json.load(handle)
    code_cells = [c for c in reloaded["cells"] if c["cell_type"] == "code"]
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        compile(source, f"<cell {index}>", "exec")
    print(f"validated: {len(code_cells)} code cells all compile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
