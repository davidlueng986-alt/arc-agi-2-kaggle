#!/usr/bin/env python3
"""Validate the generated Kaggle notebook.

Checks that (1) every `%%writefile harness/X.py` cell body is byte-identical to the
tested source file, (2) every embedded module and every runtime cell parses as Python,
(3) the runtime cells appear in the intended order, and (4) no cell accidentally
enables a GPU or contradicts kernel-metadata.json.
"""

from __future__ import annotations

import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(ROOT, "notebook", "arc26_h1_pure_xhigh.ipynb")
META_PATH = os.path.join(ROOT, "notebook", "kernel-metadata.json")
MAGIC = "%" * 2 + "writefile"

EXPECTED_ORDER = [
    "10_runtime_config.py",
    "20_install_deps.py",
    "30_find_checkpoint.py",
    "40_launch_vllm.py",
    "50_probe_calibrate.py",
    "60_run_harness.py",
    "70_finalize.py",
]
ORDER_MARKERS = [
    "RUN_START_MONO = time.monotonic()",
    "def ensure_vllm()",
    "def find_checkpoint()",
    "RUNGS = [",
    "PROBE_EFFORTS =",
    "outcome = run_harness(",
    "H1 pure-xhigh run summary",
]


def source_of(cell: dict) -> str:
    return "".join(cell["source"])


def main() -> int:
    errors: list[str] = []
    with open(NB_PATH, "r", encoding="utf-8") as fh:
        nb = json.load(fh)
    with open(META_PATH, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    cells = nb["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]

    if nb.get("nbformat") != 4:
        errors.append(f"nbformat is {nb.get('nbformat')}, expected 4")

    if meta.get("enable_gpu") is not False:
        errors.append(f"kernel-metadata enable_gpu must be false, got {meta.get('enable_gpu')!r}")
    if meta.get("code_file") != os.path.basename(NB_PATH):
        errors.append(f"code_file {meta.get('code_file')!r} != {os.path.basename(NB_PATH)!r}")
    if "arc-prize-2026-arc-agi-2" not in (meta.get("competition_sources") or []):
        errors.append("competition_sources must include arc-prize-2026-arc-agi-2")
    if not (meta.get("dataset_sources") or []):
        errors.append("dataset_sources must attach the model checkpoint")

    embedded = 0
    first_writefile_index = None
    makedirs_index = None
    for cell_no, cell in enumerate(code_cells):
        src = source_of(cell)
        if src.startswith(MAGIC):
            if first_writefile_index is None:
                first_writefile_index = cell_no
            head, _, body = src.partition("\n")
            target = head.split()[1]
            real_path = os.path.join(ROOT, target.replace("/", os.sep))
            if not os.path.exists(real_path):
                errors.append(f"writefile target missing on disk: {target}")
                continue
            with open(real_path, "r", encoding="utf-8") as fh:
                real = fh.read().replace("\r\n", "\n")
            got = body.replace("\r\n", "\n")
            if not got.endswith("\n"):
                got += "\n"
            if got != real:
                errors.append(f"EMBEDDED SOURCE DRIFT: {target} differs from {real_path}")
            else:
                embedded += 1
            try:
                ast.parse(body)
            except SyntaxError as exc:
                errors.append(f"{target}: embedded module does not parse: {exc}")
        elif "os.makedirs('harness'" in src or 'os.makedirs("harness"' in src:
            if makedirs_index is None:
                makedirs_index = cell_no

    if first_writefile_index is None:
        errors.append("no %%writefile cells found")
    if makedirs_index is None:
        errors.append(
            "no cell creates harness/ before the writefile cells; IPython's "
            "%%writefile does not create parent directories and will raise "
            "FileNotFoundError"
        )
    elif first_writefile_index is not None and makedirs_index > first_writefile_index:
        errors.append(
            f"harness/ is created in code cell #{makedirs_index} but the first "
            f"%%writefile is cell #{first_writefile_index}; the directory must exist first"
        )

    if embedded != 15:
        errors.append(f"expected 15 embedded harness modules, verified {embedded}")

    runtime = [source_of(c) for c in code_cells if not source_of(c).startswith(MAGIC)]
    for i, src in enumerate(runtime):
        try:
            ast.parse(src)
        except SyntaxError as exc:
            errors.append(f"runtime cell #{i} does not parse: {exc}\n---\n{src[:300]}")

    joined = "\n".join(runtime)
    for marker in ORDER_MARKERS:
        if marker not in joined:
            errors.append(f"missing expected runtime marker: {marker!r}")
    positions = [joined.index(m) for m in ORDER_MARKERS if m in joined]
    if positions != sorted(positions):
        errors.append("runtime cells are out of order")

    for forbidden, why in (
        ('"enable_gpu": true', "GPU must stay disabled in metadata"),
        ("turbo_dfs", "legacy DFS solver is out of scope for H1"),
        ("guided_json", "structured output is unreliable in thinking mode"),
        ("response_format", "response_format doubles braces with reasoning (vLLM#48228)"),
        ("reasoning_effort='high'", "'high' raises HTTP 500 (QwenLM/Qwen3.8#217)"),
        ('"--reasoning-effort", "high"', "'high' raises HTTP 500"),
        ("nvfp4", "NVFP4 is Blackwell-native; L4 is Ada"),
    ):
        if forbidden in joined:
            errors.append(f"runtime contains {forbidden!r}: {why}")

    if first_writefile_index != makedirs_index + 1:
        errors.append(
            f"expected the first %%writefile cell (#{first_writefile_index}) to "
            f"immediately follow the makedirs cell (#{makedirs_index})"
        )

    print(f"cells: {len(cells)} total, {len(code_cells)} code, {len(md_cells)} markdown")
    print(f"embedded harness modules verified byte-identical: {embedded}/15")
    print(f"runtime cells parsed: {len(runtime)}")
    print(f"notebook size: {os.path.getsize(NB_PATH) / 1024.0:.1f} KB")
    print(f"metadata: gpu={meta.get('enable_gpu')} internet={meta.get('enable_internet')} "
          f"datasets={meta.get('dataset_sources')} competitions={meta.get('competition_sources')}")

    if errors:
        print(f"\n{len(errors)} PROBLEM(S):")
        for err in errors:
            print("  - " + err)
        return 1
    print("\nNOTEBOOK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
