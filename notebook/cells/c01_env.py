# --- H2 Duck-REPL: environment, deadline, and harness source -----------------
# Everything this notebook needs is either attached as a Kaggle dataset or
# embedded below, because the competition rerun has no internet.
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

NOTEBOOK_START = time.time()
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}

# 12h Kaggle limit, minus the 600s buffer the competition requires the
# submission to be written before, minus a reserve for vLLM load + CUDA graph
# capture. The watchdog finalises the submission at HARD_DEADLINE.
TOTAL_SECONDS = 12 * 3600.0
SAFETY_BUFFER = 600.0
STARTUP_RESERVE = 1200.0
HARD_DEADLINE = NOTEBOOK_START + TOTAL_SECONDS - SAFETY_BUFFER
USABLE_SECONDS = TOTAL_SECONDS - SAFETY_BUFFER - STARTUP_RESERVE

WORKING = Path("/kaggle/working") if Path("/kaggle").is_dir() else Path.cwd()
SRC = WORKING / "h2"
SUBMISSION_PATH = WORKING / "submission.json"
LOG_DIR = WORKING / "logs"
for directory in (SRC, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

COMPETITION_SLUG = "arc-prize-2026-arc-agi-2"
DATA_DIR_CANDIDATES = [
    Path(f"/kaggle/input/competitions/{COMPETITION_SLUG}"),
    Path(f"/kaggle/input/{COMPETITION_SLUG}"),
    Path("/kaggle/input/arc-prize-2026-arc-agi-2"),
    Path.cwd() / "data",
]
CHALLENGE_GLOB = "arc-agi_*_challenges.json"


def list_inputs(limit=60):
    """Diagnostic: what actually got mounted. Kaggle's competition mount path has
    moved between releases and a code competition may only expose data at rerun,
    so guessing a single path is fragile -- print what is really there."""
    root = Path("/kaggle/input")
    if not root.is_dir():
        print("  /kaggle/input does not exist (not running on Kaggle?)")
        return []
    entries = []
    for path in sorted(root.iterdir()):
        kind = "dir " if path.is_dir() else "file"
        entries.append(f"  {kind} {path}")
    for line in entries[:limit]:
        print(line)
    if len(entries) > limit:
        print(f"  ... {len(entries) - limit} more")
    return entries


def _has_challenges(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob(CHALLENGE_GLOB))


def find_data_dir():
    """Locate the ARC challenge files.

    Order: explicit ARC_DATA_DIR (validated -- an override that points somewhere
    empty must fall through rather than be trusted), then the known mount paths,
    then a search of every mounted input. The last step is what makes this
    survive a mount-path change.
    """
    override = os.environ.get("ARC_DATA_DIR", "").strip()
    if override:
        candidate = Path(override)
        if _has_challenges(candidate):
            return candidate
        print(f"WARNING: ARC_DATA_DIR={override} holds no {CHALLENGE_GLOB}; ignoring it")
    for candidate in DATA_DIR_CANDIDATES:
        if _has_challenges(candidate):
            return candidate
    root = Path("/kaggle/input")
    if root.is_dir():
        for hit in root.rglob(CHALLENGE_GLOB):
            return hit.parent
    return None


def find_sample_submission():
    """Last-resort source of the task-id -> n_test mapping."""
    for candidate in (
        Path("/kaggle/input") / f"competitions/{COMPETITION_SLUG}/sample_submission.json",
        Path.cwd() / "sample_submission.json",
    ):
        if candidate.is_file():
            return candidate
    root = Path("/kaggle/input")
    if root.is_dir():
        for hit in root.rglob("sample_submission.json"):
            return hit
    for candidate in DATA_DIR_CANDIDATES:
        path = candidate / "sample_submission.json"
        if path.is_file():
            return path
    return None


def count_gpus():
    """GPU count without importing torch (torch may not be present on a CPU run)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30
        )
        if out.returncode == 0:
            return len([ln for ln in out.stdout.splitlines() if ln.strip().startswith("GPU ")])
    except Exception:
        pass
    return 0


print("--- mounted inputs (/kaggle/input) ---")
list_inputs()
print("--------------------------------------")

DATA_DIR = find_data_dir()
SAMPLE_SUBMISSION = find_sample_submission()
N_GPU = count_gpus()

print(f"competition rerun : {TRUE_SUBMISSION}")
print(f"working dir       : {WORKING}")
print(f"data dir          : {DATA_DIR}")
print(f"sample submission : {SAMPLE_SUBMISSION}")
print(f"gpus visible      : {N_GPU}")
print(f"usable seconds    : {USABLE_SECONDS:,.0f}  hard deadline in {HARD_DEADLINE - time.time():,.0f}s")

SERVED_MODEL_NAME = "qwen3.8-27b"
VLLM_PORT = 8000
VLLM_BASE_URL = f"http://127.0.0.1:{VLLM_PORT}/v1"
# Our measured context ceiling is a 6.4K-token prefix plus a ~5.2K rolling
# window, so 32768 is >3x headroom. Capping here rather than at the upstream
# 262144 default lets vLLM schedule more concurrent sequences, which is what
# makes the decode throughput in HYPOTHESIS.md section 5.3 reachable.
MAX_MODEL_LEN = 32768
MAX_NUM_SEQS = 32
TENSOR_PARALLEL = 4
WORKERS = 16

# --- write the embedded harness source ---------------------------------------
# A zip of harness/*.py, base64'd, so this notebook is self-contained and does
# not depend on a source dataset being attached correctly.
HARNESS_ZIP_B64 = "__HARNESS_ZIP_B64__"


def materialize_harness():
    if not HARNESS_ZIP_B64.strip():
        raise RuntimeError("harness payload is empty; rebuild the notebook with scripts/build_notebook.py")
    blob = base64.b64decode(HARNESS_ZIP_B64)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        archive.extractall(SRC)
    files = sorted(p.name for p in (SRC / "harness").glob("*.py"))
    print(f"harness written to {SRC}/harness  ({len(files)} modules, {len(blob):,} bytes)")
    print("  " + ", ".join(files))
    return files


MATERIALIZED = materialize_harness()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import harness  # noqa: E402
from harness import dataio as H_dataio  # noqa: E402

print(f"harness version   : {harness.__version__}")
