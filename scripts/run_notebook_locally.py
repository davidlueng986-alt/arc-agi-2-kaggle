#!/usr/bin/env python3
"""Execute notebook/arc26_h2_duck_repl.ipynb locally, cell by cell.

The rerun happens on Kaggle with GPU and no internet; this is the closest thing
to a dry run available on a laptop.  It executes every code cell in one shared
namespace, exactly as a kernel would, with three differences forced by the local
environment:

* ``ARC_DATA_DIR`` points at the shared data copy (the notebook's own search
  would find ``/kaggle/input/...``, which does not exist here);
* ``ARC26_LIMIT`` / ``ARC26_MAX_TURNS`` keep the run short;
* there is no ``/kaggle``, so cells 2-4 find no wheelhouse, no weights and no GPU
  and degrade to the mock agent -- which is the path we most want to prove is
  safe, because ``kernel-metadata.json`` ships with ``enable_gpu: false``.

    python scripts/run_notebook_locally.py --limit 3 --data-dir <path>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebook" / "arc26_h2_duck_repl.ipynb"


def code_cells(notebook_path: Path) -> list[str]:
    with open(notebook_path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.environ.get("ARC_DATA_DIR"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--workdir", default=None, help="where /kaggle/working is simulated")
    parser.add_argument("--challenge-file", default="arc-agi_evaluation_challenges.json")
    parser.add_argument(
        "--keep-arc-data-dir-env",
        action="store_true",
        help="do not overwrite an ARC_DATA_DIR already present in the environment "
        "(used to test the notebook's override-validation path)",
    )
    parser.add_argument("--keep", action="store_true", help="keep the temporary working dir")
    args = parser.parse_args(argv)

    if not NOTEBOOK.is_file():
        print(f"notebook not found: {NOTEBOOK}\nrun: python scripts/build_notebook.py", file=sys.stderr)
        return 2
    if not args.data_dir:
        print("--data-dir (or ARC_DATA_DIR) is required", file=sys.stderr)
        return 2

    cells = code_cells(NOTEBOOK)
    print(f"notebook has {len(cells)} code cells")

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="arc26_nb_"))
    workdir.mkdir(parents=True, exist_ok=True)

    driver = workdir / "_drive.py"
    with open(driver, "w", encoding="utf-8") as handle:
        handle.write("import json, os, sys, traceback\n")
        handle.write(f"CELLS = json.loads(r'''{json.dumps(cells)}''')\n")
        handle.write(
            "ns = {'__name__': '__main__'}\n"
            "for index, source in enumerate(CELLS, start=1):\n"
            "    print(f'\\n===== cell {index} =====', flush=True)\n"
            "    try:\n"
            "        exec(compile(source, f'<cell {index}>', 'exec'), ns)\n"
            "    except Exception:\n"
            "        print(f'CELL {index} FAILED', flush=True)\n"
            "        traceback.print_exc()\n"
            "        raise SystemExit(1)\n"
            "print('\\nALL CELLS OK', flush=True)\n"
        )

    env = dict(os.environ)
    if not args.keep_arc_data_dir_env:
        env["ARC_DATA_DIR"] = str(args.data_dir)
    env["ARC26_LIMIT"] = str(args.limit)
    env["ARC26_MAX_TURNS"] = str(args.max_turns)
    env["ARC26_CHALLENGE_FILE"] = args.challenge_file
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("KAGGLE_IS_COMPETITION_RERUN", None)

    print(f"working dir: {workdir}")
    completed = subprocess.run(
        [sys.executable, str(driver)], cwd=str(workdir), env=env, text=True, capture_output=True
    )
    sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)

    submission = workdir / "submission.json"
    print(f"\nsubmission written: {submission.is_file()}")
    if not args.keep and not args.workdir:
        print(f"(temporary working dir kept at {workdir})")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
