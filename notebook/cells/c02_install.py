# --- offline runtime install --------------------------------------------------
# The rerun has no internet. vLLM must come from an attached wheelhouse dataset;
# if it is already importable (or absent and unobtainable) we degrade to a
# CPU-only run that still writes a schema-valid submission.
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

WHEELHOUSE_MARKERS = ("vllm", "wheelhouse", "wheels")


def have(module):
    return importlib.util.find_spec(module) is not None


def find_wheelhouse():
    root = Path("/kaggle/input")
    if not root.is_dir():
        return None
    for marker in WHEELHOUSE_MARKERS:
        for candidate in sorted(root.rglob(marker)):
            if candidate.is_dir() and any(candidate.glob("vllm*.whl")):
                return candidate
    for candidate in sorted(root.rglob("vllm*.whl")):
        return candidate.parent
    return None


def pip_install_offline(wheelhouse, packages):
    cmd = [
        sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
        "--no-warn-conflicts", "--disable-pip-version-check",
        "--find-links", str(wheelhouse), *packages,
    ]
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, stdout=subprocess.DEVNULL).returncode


RUNTIME = {"vllm": have("vllm"), "torch": have("torch"), "PIL": have("PIL"), "numpy": have("numpy")}
print("importable before install:", RUNTIME)

WHEELHOUSE = find_wheelhouse()
print(f"wheelhouse: {WHEELHOUSE}")

if not RUNTIME["vllm"] and WHEELHOUSE is not None:
    rc = pip_install_offline(WHEELHOUSE, ["vllm"])
    print(f"vllm install rc={rc}")
    RUNTIME["vllm"] = have("vllm")

if not RUNTIME["PIL"]:
    # Pillow renders the panel image. Without it the harness simply runs
    # text-only: build_task_block never depends on render succeeding.
    if WHEELHOUSE is not None:
        pip_install_offline(WHEELHOUSE, ["pillow"])
    RUNTIME["PIL"] = have("PIL")

print("importable after install:", RUNTIME)

# The sandbox needs numpy for model-written code; the harness itself does not.
if not RUNTIME["numpy"]:
    print("WARNING: numpy missing -- model-generated transforms will be restricted to pure Python")

VLLM_AVAILABLE = bool(RUNTIME["vllm"] and N_GPU >= 1)
print(f"vLLM serving possible: {VLLM_AVAILABLE}  (vllm={RUNTIME['vllm']}, gpus={N_GPU})")
if not VLLM_AVAILABLE:
    print(
        "NOTE: falling back to the deterministic mock agent. This produces a "
        "schema-valid submission so the notebook never fails silently, but its "
        "score is meaningless. Attach a vLLM wheelhouse dataset and enable GPU "
        "for a real run."
    )
