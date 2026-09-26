"""Sandbox worker process (runs under `python -I` with a clean environment).

Protocol: reads newline-delimited JSON jobs on stdin:
    {"id": int, "code": str, "grid": [[int]]}
Writes one JSON result line per job to stdout (flushed immediately):
    {"id": int, "ok": bool, "out": [[int]]|null, "err": str|null, "ms": float}

The parent enforces a per-job deadline by killing this process when the next
result line does not arrive in time, then resumes the remaining jobs in a
fresh worker (see sandbox.py). On POSIX we additionally set SIGALRM per job
and RLIMIT_AS as defense in depth.
"""
from __future__ import annotations

import io
import json
import sys
import time

MAX_OUTPUT_CELLS = 64 * 64
MAX_CODE_BYTES = 128 * 1024
MAX_ERR_CHARS = 600


def _setup_posix_limits() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))  # 2 GB
        resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
    except Exception:
        pass


def _setup_posix_alarm(timeout_s: float):
    import signal

    class _JobTimeout(BaseException):
        pass

    def handler(signum, frame):
        raise _JobTimeout()

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    return _JobTimeout


def _cancel_posix_alarm() -> None:
    import signal

    signal.setitimer(signal.ITIMER_REAL, 0)


def main() -> None:
    is_posix = sys.platform != "win32"
    if is_posix:
        _setup_posix_limits()

    import numpy as np

    try:
        np.seterr(all="ignore")
    except Exception:
        pass

    out_write = sys.stdout
    code_cache: dict[int, dict] = {}
    job_timeout_s = 5.0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception:
            continue
        if job.get("op") == "config":
            job_timeout_s = float(job.get("timeout_s", job_timeout_s))
            continue
        jid = job.get("id")
        t0 = time.perf_counter()
        result = {"id": jid, "ok": False, "out": None, "err": None, "ms": 0.0}
        alarm_exc = None
        try:
            code = job["code"]
            if len(code.encode("utf-8", "replace")) > MAX_CODE_BYTES:
                raise ValueError("program too large")
            grid = np.array(job["grid"], dtype=np.int64)
            if grid.ndim != 2:
                raise ValueError("input grid must be 2-D")

            h = hash(code)
            ns = code_cache.get(h)
            if ns is None:
                ns = {"__builtins__": __builtins__}
                buf = io.StringIO()
                old = sys.stdout
                sys.stdout = buf
                try:
                    exec(compile(code, "<candidate>", "exec"), ns)
                finally:
                    sys.stdout = old
                if not callable(ns.get("transform")):
                    raise ValueError("program does not define callable transform(grid)")
                code_cache[h] = ns

            if is_posix:
                alarm_exc = _setup_posix_alarm(job_timeout_s)
            try:
                old = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    out = ns["transform"](grid.copy())
                finally:
                    sys.stdout = old
            finally:
                if is_posix:
                    _cancel_posix_alarm()

            out = np.asarray(out)
            if out.dtype == bool:
                out = out.astype(np.int64)
            if np.issubdtype(out.dtype, np.floating):
                if out.size and not np.all(np.isfinite(out)):
                    raise ValueError("output contains non-finite values")
                out = np.rint(out).astype(np.int64)
            out = out.astype(np.int64, copy=False)
            if out.ndim != 2:
                raise ValueError(f"output must be 2-D, got ndim={out.ndim}")
            if out.size > MAX_OUTPUT_CELLS:
                raise ValueError(f"output too large: {out.shape}")
            if out.size and (out.min() < -999 or out.max() > 999):
                raise ValueError("output values out of sane range")
            result["ok"] = True
            result["out"] = out.tolist()
        except BaseException as e:  # noqa: BLE001 - sandbox must never crash
            if alarm_exc is not None and isinstance(e, alarm_exc):
                result["err"] = "timeout"
            else:
                result["err"] = f"{type(e).__name__}: {e}"[:MAX_ERR_CHARS]
        result["ms"] = round((time.perf_counter() - t0) * 1000, 2)
        out_write.write(json.dumps(result) + "\n")
        out_write.flush()


if __name__ == "__main__":
    main()
