"""Launch vLLM through a flag-fallback ladder and wait for health.

Every flag here is version-sensitive and none of it could be tested without a GPU, so
the ladder degrades rather than fails: TP=4 with full caching, then progressively
fewer optional flags, then TP=2 as a last resort. `--mamba-cache-mode align` (NOT
`all`) is required for prefix caching on the hybrid Gated DeltaNet architecture.
"""
import json
import os
import signal
import subprocess
import time
import urllib.request

LAUNCH_BUDGET = max(600.0, min(LAUNCH_RESERVE_S, remaining_s() - 3600.0 - SUBMISSION_BUFFER_S))
FIRST_RUNG_FRACTION = 0.45

RUNGS = [
    ("tp4-full", [
        "--tensor-parallel-size", "4", "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.92", "--max-num-seqs", "64",
        "--reasoning-parser", "qwen3",
        "--enable-prefix-caching", "--mamba-cache-mode", "align",
    ], 1.0),
    ("tp4-noalign", [
        "--tensor-parallel-size", "4", "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.92", "--max-num-seqs", "64",
        "--reasoning-parser", "qwen3", "--enable-prefix-caching",
    ], 0.6),
    ("tp4-nocache", [
        "--tensor-parallel-size", "4", "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.92", "--max-num-seqs", "64",
        "--reasoning-parser", "qwen3",
    ], 0.6),
    ("tp4-bare", [
        "--tensor-parallel-size", "4", "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.93", "--max-num-seqs", "64",
    ], 0.6),
    ("tp2-small", [
        "--tensor-parallel-size", "2", "--max-model-len", "24576",
        "--gpu-memory-utilization", "0.95", "--max-num-seqs", "12",
        "--reasoning-parser", "qwen3",
    ], 0.6),
    ("tp2-bare", [
        "--tensor-parallel-size", "2", "--max-model-len", "24576",
        "--gpu-memory-utilization", "0.95", "--max-num-seqs", "12",
    ], 0.6),
]


def endpoint_healthy(base_url, timeout=10):
    try:
        req = urllib.request.Request(base_url + "/models")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def kill(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def tail(path, n=25):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except Exception:
        return "<no log>"


def launch_rung(name, extra_args, port, log_path, timeout):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--trust-remote-code", "--disable-log-requests",
        "--port", str(port),
    ] + extra_args
    log("rung %s: %s" % (name, " ".join(cmd[3:])))
    fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, stdout=fh, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if os.name != "nt" else None,
        env=dict(os.environ, VLLM_LOGGING_LEVEL="INFO"),
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log("rung %s: process exited rc=%s" % (name, proc.returncode))
            log(tail(log_path))
            fh.close()
            return None
        if endpoint_healthy("http://127.0.0.1:%d/v1" % port):
            log("rung %s: healthy after %.0fs" % (name, timeout - (deadline - time.monotonic())))
            return proc, fh
        time.sleep(5)
    log("rung %s: not healthy within %.0fs" % (name, timeout))
    log(tail(log_path))
    kill(proc)
    fh.close()
    return None


VLLM_PROC = None
VLLM_LOG_FH = None
SERVING = None

if MODEL_PATH and VLLM_OK:
    weights = [w for _n, _a, w in RUNGS]
    total_w = sum(weights)
    for i, (name, args, weight) in enumerate(RUNGS):
        share = FIRST_RUNG_FRACTION if i == 0 else (1.0 - FIRST_RUNG_FRACTION) * weight / (total_w - weights[0])
        timeout = max(180.0, LAUNCH_BUDGET * share)
        if remaining_s() - timeout < 3600.0 + SUBMISSION_BUFFER_S:
            log("not enough time left for rung %s (%.0fs); stopping the ladder" % (name, timeout))
            break
        result = launch_rung(name, args, VLLM_PORT, VLLM_LOG, timeout)
        if result is not None:
            VLLM_PROC, VLLM_LOG_FH = result
            SERVING = {"rung": name, "tp": args[args.index("--tensor-parallel-size") + 1], "base_url": BASE_URL}
            break

if SERVING:
    log("SERVING via %s at %s" % (SERVING["rung"], SERVING["base_url"]))
else:
    log("NO SERVER: every rung failed. The run will emit a schema-valid placeholder submission.")
print("elapsed %.0fs, remaining %.2f h" % (elapsed_s(), remaining_s() / 3600.0))
