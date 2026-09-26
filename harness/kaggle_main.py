"""Kaggle rerun orchestration: locate the attached model, launch vLLM (TP=4,
offline), run the pipeline over the test challenges with the watchdog, and
guarantee a schema-valid /kaggle/working/submission.json.

Everything degrades to a fallback-only submission (never lose the submission):
no GPU -> fallback; vLLM missing -> wheelhouse install -> fallback; server never
becomes healthy -> fallback.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Optional

COMPETITION_DIR = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
WORKING_DIR = "/kaggle/working"
MODEL_CANDIDATES = (
    "/kaggle/input/qwen3-8-27b-fp8",
    "/kaggle/input/qwen3-8-27b-fp8-hf-snapshot",
    "/kaggle/input/qwen3-8-27b-bf16",
    "/kaggle/input/qwen3-8-27b",
)
VLLM_ARGS = [
    "--served-model-name", "qwen3.8-27b",
    "--tensor-parallel-size", "4",
    "--max-model-len", "49152",
    "--gpu-memory-utilization", "0.90",
    "--reasoning-parser", "qwen3",
    "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
    "--enable-prefix-caching", "--mamba-cache-mode", "align",
    "--max-num-seqs", "16",
    "--port", "8000",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_model_dir() -> Optional[str]:
    for cand in MODEL_CANDIDATES:
        if os.path.isdir(cand):
            for root, _dirs, files in os.walk(cand):
                if any(f.endswith(".safetensors") for f in files) and any(
                    f == "config.json" for f in files
                ):
                    return root
    return None


def have_vllm() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except Exception:
        return False


def install_vllm_from_wheelhouse() -> bool:
    """Offline rerun: pip-install from an attached wheelhouse dataset if present."""
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return False
    wheels = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        try:
            if os.path.isdir(d) and any(f.endswith(".whl") for f in os.listdir(d)):
                wheels.append(d)
        except OSError:
            continue
    for d in wheels:
        log(f"trying wheelhouse {d}")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", d, "vllm"],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and have_vllm():
            return True
    return False


def launch_vllm(model_dir: str, log_path: str) -> Optional[subprocess.Popen]:
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", model_dir] + VLLM_ARGS
    # env pins: never let reasoning_effort default (xhigh) matter — we pass it
    # per request; disable telemetry noise.
    env = dict(os.environ)
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    log(f"launching: {' '.join(cmd)}")
    with open(log_path, "w", encoding="utf-8") as lf:
        return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)


def wait_for_server(base_url: str, timeout_s: float, proc: Optional[subprocess.Popen]) -> bool:
    import urllib.request

    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if proc is not None and proc.poll() is not None:
            log(f"vLLM process exited with code {proc.returncode}")
            return False
        try:
            with urllib.request.urlopen(base_url + "/models", timeout=5) as r:
                if r.status == 200:
                    log("vLLM server healthy")
                    return True
        except Exception:
            time.sleep(10)
    return False


def load_challenges() -> tuple[dict, Optional[dict]]:
    rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    if rerun:
        path = os.path.join(COMPETITION_DIR, "arc-agi_test_challenges.json")
        solutions = None
    else:
        path = os.path.join(COMPETITION_DIR, "arc-agi_evaluation_challenges.json")
        solutions = None
        sol_path = os.path.join(COMPETITION_DIR, "arc-agi_evaluation_solutions.json")
        if os.path.exists(sol_path):
            with open(sol_path, encoding="utf-8") as f:
                solutions = json.load(f)
    with open(path, encoding="utf-8") as f:
        challenges = json.load(f)
    log(f"loaded {len(challenges)} tasks from {path} (rerun={rerun})")
    return challenges, solutions


def main(total_time_s: Optional[float] = None) -> dict:
    from harness.budget import Deadline
    from harness.config import PipelineConfig
    from harness.llm import OpenAICompatClient
    from harness.runner import TaskRunner

    out_path = os.path.join(WORKING_DIR, "submission.json")
    challenges, solutions = load_challenges()

    # 12h - 600s buffer (lb3389 pattern); minus the serving setup measured below.
    setup_start = time.monotonic()
    cfg = PipelineConfig()
    deadline = Deadline(total_time_s if total_time_s is not None else cfg.total_time_s)

    writer_ready = False
    proc = None
    base_url = "http://127.0.0.1:8000/v1"
    model_dir = find_model_dir()
    gpu = shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi"], capture_output=True
    ).returncode == 0

    if not gpu:
        log("NO GPU visible — fallback-only submission mode")
    elif model_dir is None:
        log("model dataset not found — fallback-only submission mode")
    else:
        if not have_vllm():
            log("vllm not importable; trying offline wheelhouse")
            if not install_vllm_from_wheelhouse():
                log("vLLM unavailable — fallback-only submission mode")
                model_dir = None
        if model_dir is not None:
            proc = launch_vllm(model_dir, os.path.join(WORKING_DIR, "vllm.log"))
            load_budget = 1800.0  # generous: FP8 27GB load + graph capture
            if not wait_for_server(base_url, load_budget, proc):
                log("vLLM never became healthy — fallback-only submission mode")
                if proc is not None:
                    proc.kill()
                proc = None
            else:
                # shrink the usable clock by the time setup actually took
                spent = time.monotonic() - setup_start
                deadline = Deadline(max(600.0, (total_time_s or cfg.total_time_s) - spent))
                writer_ready = True

    runner = TaskRunner(
        challenges,
        client=OpenAICompatClient(base_url=base_url, model=cfg.model),
        cfg=cfg,
        out_path=out_path,
        solutions=solutions,
        deadline=deadline,
    )
    runner.start_watchdog(finalize_before_s=60.0)

    if not writer_ready:
        summary = runner.writer.finalize()
        log(f"fallback submission written; validation errors: {summary['errors']}")
        return {"mode": "fallback", **summary}

    summary = asyncio.run(runner.run())
    log("summary: " + json.dumps({k: v for k, v in summary.items() if k != "tokens"}))
    log("tokens: " + json.dumps(summary.get("tokens", {})))
    if proc is not None:
        proc.terminate()
    return {"mode": "full", **summary}


if __name__ == "__main__":
    total = float(sys.argv[1]) if len(sys.argv) > 1 else None
    main(total)
