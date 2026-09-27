# --- launch vLLM (offline) ----------------------------------------------------
# Official Qwen3.8 serving pattern, with the traps from HYPOTHESIS.md handled:
#   * --reasoning-parser qwen3 and --tool-call-parser qwen3_coder (native tools)
#   * --mamba-cache-mode align, required for prefix caching on the GatedDeltaNet
#     hybrid; without it prefix caching is unsupported
#   * no --speculative-config: MTP/speculative decoding breaks constrained
#     decoding on this model
#   * reasoning_effort is never set to `high` (HTTP 500, QwenLM/Qwen3.8#217);
#     the client validates against {low, medium, xhigh} and drops the field
#     entirely if the server still 500s
# Flags are probed against --help first, because vLLM renames them between
# releases and the rerun has no internet to check against.
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

VLLM_LOG = LOG_DIR / "vllm.log"
SERVER_PROC = None
SERVER_READY = False


def module_args():
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]


def probe_help():
    try:
        out = subprocess.run(
            module_args() + ["--help"], capture_output=True, text=True, timeout=300
        )
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:
        print(f"--help probe failed: {exc}")
        return ""


HELP_TEXT = probe_help() if VLLM_AVAILABLE else ""
print(f"--help probe returned {len(HELP_TEXT):,} chars")


def supported(flag):
    return flag in HELP_TEXT


def build_command(tensor_parallel):
    cmd = module_args() + [
        "--model", MODEL_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--tensor-parallel-size", str(tensor_parallel),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--gpu-memory-utilization", "0.90",
        "--trust-remote-code",
        "--port", str(VLLM_PORT),
        "--host", "127.0.0.1",
        "--disable-log-requests",
    ]
    optional = [
        ["--reasoning-parser", "qwen3"],
        ["--enable-auto-tool-choice"],
        ["--tool-call-parser", "qwen3_coder"],
        ["--enable-prefix-caching"],
        ["--mamba-cache-mode", "align"],
    ]
    for flag in optional:
        if not HELP_TEXT or supported(flag[0]):
            cmd.extend(flag)
        else:
            print(f"  skipping unsupported flag {flag[0]}")
    if not HELP_TEXT or supported("--limit-mm-per-prompt"):
        cmd.extend(["--limit-mm-per-prompt", json.dumps({"image": 2})])
    return cmd


def wait_ready(timeout_s, proc):
    url = f"{VLLM_BASE_URL}/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, f"server exited with code {proc.returncode}"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode())
                    ids = [m.get("id") for m in payload.get("data", [])]
                    return True, f"ready, models={ids}"
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(5)
    return False, f"not ready after {timeout_s:.0f}s"


def launch():
    global SERVER_PROC, SERVER_READY
    if not VLLM_AVAILABLE or not MODEL_PATH:
        print("skipping vLLM launch (unavailable)")
        return False

    tp = min(TENSOR_PARALLEL, max(1, N_GPU))
    # 27 GB of fp8 weights does not fit in one 24 GB L4, so TP>=2 is mandatory
    # for this model. Prefer the documented TP=4 over a single 96 GB pool.
    if tp < 2:
        print(f"only {tp} GPU(s): FP8 27B needs >=2 (27 GB > 24 GB per L4)")
        return False

    load_budget = min(2400.0, max(600.0, (HARD_DEADLINE - time.time()) * 0.25))
    attempts = [tp]
    if tp == 4:
        attempts.append(2)  # TP=2 x 2 instances halves all-reduce participants
    for attempt_tp in attempts:
        cmd = build_command(attempt_tp)
        print("$ " + " ".join(shlex.quote(c) for c in cmd))
        env = dict(os.environ)
        env["VLLM_LOGGING_LEVEL"] = "INFO"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        handle = open(VLLM_LOG, "ab")
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
        ok, detail = wait_ready(load_budget, proc)
        print(f"TP={attempt_tp}: {detail}")
        if ok:
            SERVER_PROC, SERVER_READY = proc, True
            print(f"vLLM ready in {time.time() - NOTEBOOK_START:.0f}s since notebook start")
            return True
        proc.kill()
        try:
            tail = Path(VLLM_LOG).read_text(encoding="utf-8", errors="replace")[-3000:]
            print("--- vllm.log tail ---\n" + tail)
        except Exception:
            pass
    return False


SERVER_READY = launch()
print(f"\nSERVER_READY = {SERVER_READY}")
print(f"elapsed since notebook start: {time.time() - NOTEBOOK_START:.0f}s")
