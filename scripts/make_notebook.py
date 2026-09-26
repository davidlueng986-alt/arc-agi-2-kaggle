#!/usr/bin/env python3
"""Generate `notebook/arc26_h1_pure_xhigh.ipynb` from the tested harness source.

The notebook must be self-contained (a Kaggle rerun can only see attached inputs), so
every `harness/*.py` module is embedded verbatim as a `%%writefile` cell. Generating
rather than hand-copying guarantees the kernel ships exactly the code that the 214
unit tests exercise -- there is no way for the notebook to drift from the harness.

Regenerate after any harness change:  python scripts/make_notebook.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS_DIR = os.path.join(ROOT, "harness")
CELLS_DIR = os.path.join(ROOT, "notebook", "cells")
OUT_PATH = os.path.join(ROOT, "notebook", "arc26_h1_pure_xhigh.ipynb")
MAGIC = "%" + "%writefile"

MODULE_ORDER = [
    "__init__.py",
    "grids.py",
    "tasks.py",
    "prompts.py",
    "parse.py",
    "llm.py",
    "select.py",
    "budget.py",
    "submission.py",
    "scoring.py",
    "diagnostics.py",
    "config.py",
    "watchdog.py",
    "waves.py",
    "runner.py",
]


def md_cell(source: str) -> Dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code_cell(source: str) -> Dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def _lines(text: str) -> List[str]:
    text = text.replace("\r\n", "\n")
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def build() -> Dict:
    cells: List[Dict] = []

    cells.append(md_cell(read(os.path.join(CELLS_DIR, "00_header.md"))))

    cells.append(md_cell(
        "## 1. Write the harness to disk\n\n"
        "Every module below is the exact source that the offline test-suite "
        "(`pytest tests`, 223 tests) exercises. The harness itself is "
        "**standard-library only** -- the sole third-party dependency in this "
        "kernel is vLLM.\n"
    ))
    for module in MODULE_ORDER:
        path = os.path.join(HARNESS_DIR, module)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        cells.append(code_cell("".join([MAGIC, " harness/", module, "\n", read(path)])))

    cells.append(code_cell(
        "import os\n"
        "os.makedirs('harness', exist_ok=True)\n"
        "print(sorted(os.listdir('harness')))\n"
        "import harness\n"
        "print('harness import OK:', len(harness.__all__), 'exports')\n"
    ))

    cells.append(md_cell(read(os.path.join(CELLS_DIR, "05_budget.md"))))

    cells.append(md_cell(
        "## 2. Runtime\n\n"
        "Configuration, dependency install, checkpoint discovery, vLLM launch "
        "ladder, live calibration, the harness run, and final validation.\n"
    ))
    for name in sorted(os.listdir(CELLS_DIR)):
        if not name.endswith(".py"):
            continue
        cells.append(code_cell(read(os.path.join(CELLS_DIR, name))))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main() -> int:
    for directory in (HARNESS_DIR, CELLS_DIR, os.path.dirname(OUT_PATH)):
        if not os.path.isdir(directory):
            print("missing directory: %s" % directory, file=sys.stderr)
            return 1
    notebook = build()
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(notebook, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    n_md = sum(1 for c in notebook["cells"] if c["cell_type"] == "markdown")
    n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    size_kb = os.path.getsize(OUT_PATH) / 1024.0
    print("wrote %s" % OUT_PATH)
    print("  cells: %d (%d markdown, %d code), %.1f KB" % (len(notebook["cells"]), n_md, n_code, size_kb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
