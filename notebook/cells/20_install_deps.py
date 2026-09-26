"""Install vLLM: pre-baked wheelhouse first (offline rerun), pip second (internet save)."""
import glob
import importlib
import os
import subprocess
import sys


def sh(cmd, timeout=3600):
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, timeout=timeout)
    return proc.returncode


def find_wheelhouse():
    patterns = [
        os.path.join(INPUT_ROOT, "*wheelhouse*"),
        os.path.join(INPUT_ROOT, "*wheels*"),
        os.path.join(INPUT_ROOT, "*wheel*", "wheels"),
        os.path.join(INPUT_ROOT, "*", "wheels"),
        os.path.join(INPUT_ROOT, "*", "wheelhouse"),
    ]
    hits = []
    for pattern in patterns:
        hits.extend(glob.glob(pattern))
    for hit in sorted(set(hits)):
        if os.path.isdir(hit) and (glob.glob(os.path.join(hit, "vllm*.whl")) or glob.glob(os.path.join(hit, "*.whl"))):
            return hit
    return None


def vllm_present():
    try:
        mod = importlib.import_module("vllm")
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def ensure_vllm():
    version = vllm_present()
    if version:
        log("vllm already importable, version=%s" % version)
        return True

    wheelhouse = find_wheelhouse()
    if wheelhouse:
        log("installing vllm from wheelhouse %s" % wheelhouse)
        rc = sh([
            sys.executable, "-m", "pip", "install", "--no-index",
            "--find-links", wheelhouse, "vllm",
        ])
        if rc == 0 and vllm_present():
            return True
        log("wheelhouse install failed (rc=%s); falling back to pip" % rc)

    if IS_RERUN:
        log("ERROR: competition rerun has no internet and no usable wheelhouse was attached.")
        log("Attach a pre-baked vLLM wheelhouse dataset, or bake vLLM into a custom image.")
        return False

    log("installing vllm from PyPI (internet is on for a non-rerun save)")
    rc = sh([sys.executable, "-m", "pip", "install", "--upgrade", "vllm"])
    return rc == 0 and bool(vllm_present())


VLLM_OK = ensure_vllm()
log("vllm available: %s (version=%s)" % (VLLM_OK, vllm_present()))
if not VLLM_OK:
    log("Proceeding anyway: the harness will emit a placeholder submission for all tasks.")
print("elapsed %.0fs, remaining %.2f h" % (elapsed_s(), remaining_s() / 3600.0))
