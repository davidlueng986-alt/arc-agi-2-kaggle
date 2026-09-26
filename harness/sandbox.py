"""Subprocess-isolated sandbox for model-generated programs.

Safety model:
- worker runs under `python -I` (isolated: ignores env vars & user site) with a
  minimal clean environment (no CUDA, no user env);
- parent-side per-job deadline: the reader kills and restarts the worker when
  the next result line does not arrive in time, so infinite loops cannot wedge
  a batch (portable: works on Windows dev machines and Linux/Kaggle);
- POSIX-only defense in depth: SIGALRM per job + RLIMIT_AS 2GB inside worker;
- output caps: grids > 64x64 rejected, error strings truncated, code size capped.

The worker reads NDJSON jobs from stdin and streams NDJSON results to stdout,
so a whole verification/disambiguation battery (hundreds of grid evaluations)
runs in ONE subprocess, amortizing interpreter+numpy startup (~0.2-0.4 s).
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .types import ExecResult, Grid

_WORKER = Path(__file__).with_name("_sandbox_worker.py")


def _clean_env(tmpdir: str) -> dict[str, str]:
    env = {
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",  # never let sandboxed code touch the GPU
        "TMPDIR": tmpdir,
        "TEMP": tmpdir,
        "TMP": tmpdir,
    }
    if os.name == "nt":
        for k in ("SYSTEMROOT", "COMSPEC", "PATHEXT"):
            if k in os.environ:
                env[k] = os.environ[k]
        env["HOME"] = tmpdir
        env["USERPROFILE"] = tmpdir
    else:
        env["HOME"] = tmpdir
        # keep dynamic loader paths so cv2/scipy work when installed (Kaggle)
        for k in ("LD_LIBRARY_PATH", "PATH"):
            if k in os.environ:
                env[k] = os.environ[k]
    return env


class Sandbox:
    """Batch executor: `run_batch([(code, grid), ...]) -> [ExecResult, ...]`."""

    def __init__(
        self,
        per_job_timeout: float = 2.0,
        startup_timeout: float = 30.0,
        python_exe: Optional[str] = None,
    ):
        self.per_job_timeout = per_job_timeout
        self.startup_timeout = startup_timeout
        self.python_exe = python_exe or sys.executable
        self._tmp: Optional[tempfile.TemporaryDirectory] = None
        self.worker_restarts = 0

    # -- lifecycle ----------------------------------------------------------

    def _tmpdir(self) -> str:
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="h3_sandbox_")
        return self._tmp.name

    def close(self) -> None:
        if self._tmp is not None:
            try:
                self._tmp.cleanup()
            except OSError:
                pass
            self._tmp = None

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- public API ---------------------------------------------------------

    def run(self, code: str, grid: Grid) -> ExecResult:
        return self.run_batch([(code, grid)])[0]

    def run_batch(
        self,
        jobs: Sequence[tuple[str, Grid]],
        on_result: Optional[Callable[[int, ExecResult], None]] = None,
    ) -> list[ExecResult]:
        n = len(jobs)
        results: list[Optional[ExecResult]] = [None] * n
        pending: list[int] = list(range(n))
        poison: Counter[int] = Counter()
        restarts = 0
        max_restarts = 2 * n + 4
        tmpdir = self._tmpdir()

        while pending and restarts <= max_restarts:
            wedge = self._run_worker(jobs, pending, results, tmpdir, on_result)
            restarts += 1
            self.worker_restarts += 1
            if wedge is not None:
                idx, kind = wedge
                if kind == "crash":
                    poison[idx] += 1
                    if poison[idx] >= 2:
                        results[idx] = ExecResult(
                            ok=False, output=None, error="worker died on job twice", timed_out=False
                        )
                    # poison==1: leave result None -> job retried in a fresh worker
                # kind == "timeout": result already recorded by _run_worker
            pending = [i for i in pending if results[i] is None]

        for i in pending:  # restart budget exceeded
            results[i] = ExecResult(ok=False, error="sandbox restart budget exceeded", timed_out=True)
        out = [r if r is not None else ExecResult(ok=False, error="missing result") for r in results]
        return out

    # -- internals ----------------------------------------------------------

    def _run_worker(
        self,
        jobs: Sequence[tuple[str, Grid]],
        pending: list[int],
        results: list[Optional[ExecResult]],
        tmpdir: str,
        on_result: Optional[Callable[[int, ExecResult], None]],
    ) -> Optional[tuple[int, str]]:
        """Run one worker over `pending` jobs.

        Returns `(job_index, kind)` for the job that wedged the worker, where
        kind is "timeout" (parent deadline fired; result already recorded) or
        "crash" (worker died before producing a result), or None on clean run.
        """
        proc = subprocess.Popen(
            [self.python_exe, "-I", "-u", str(_WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=tmpdir,
            env=_clean_env(tmpdir),
        )
        lines_q: queue.Queue = queue.Queue()

        def reader() -> None:
            assert proc.stdout is not None
            try:
                for raw in proc.stdout:
                    lines_q.put(raw)
            except Exception:
                pass
            finally:
                lines_q.put(None)  # EOF sentinel

        def writer() -> None:
            assert proc.stdin is not None
            try:
                proc.stdin.write(
                    (json.dumps({"op": "config", "timeout_s": self.per_job_timeout}) + "\n").encode()
                )
                for idx in pending:
                    code, grid = jobs[idx]
                    job = {"id": idx, "code": code, "grid": np.asarray(grid).tolist()}
                    proc.stdin.write((json.dumps(job) + "\n").encode())
                proc.stdin.flush()
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        rt = threading.Thread(target=reader, daemon=True)
        wt = threading.Thread(target=writer, daemon=True)
        rt.start()
        wt.start()

        expected = list(pending)
        pos = 0
        first_line = True
        wedge: Optional[tuple[int, str]] = None
        try:
            while pos < len(expected):
                deadline = self.per_job_timeout + (self.startup_timeout if first_line else 0.5)
                try:
                    raw = lines_q.get(timeout=deadline)
                except queue.Empty:
                    # worker wedged on job expected[pos] -> kill; caller reruns rest
                    wedge = (expected[pos], "timeout")
                    results[wedge[0]] = ExecResult(ok=False, error="timeout", timed_out=True)
                    if on_result:
                        on_result(wedge[0], results[wedge[0]])
                    break
                first_line = False
                if raw is None:  # EOF: worker died before emitting a result
                    wedge = (expected[pos], "crash")
                    break
                try:
                    res = json.loads(raw.decode("utf-8", "replace"))
                except Exception:
                    continue
                idx = int(res["id"])
                out = None
                if res.get("ok") and res.get("out") is not None:
                    out = np.array(res["out"], dtype=np.int64)
                er = ExecResult(
                    ok=bool(res.get("ok")),
                    output=out,
                    error=res.get("err"),
                    timed_out=(res.get("err") == "timeout"),
                    ms=float(res.get("ms", 0.0)),
                )
                if 0 <= idx < len(results):
                    results[idx] = er
                    if on_result:
                        on_result(idx, er)
                if pos < len(expected) and idx == expected[pos]:
                    pos += 1
        finally:
            if proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            wt.join(timeout=1)
        return wedge


class LocalExecutor:
    """In-process executor for tests only (NO isolation). Same API as Sandbox."""

    def __init__(self, per_job_timeout: float = 2.0):
        self.per_job_timeout = per_job_timeout
        self.worker_restarts = 0
        self._cache: dict[int, dict] = {}

    def run(self, code: str, grid: Grid) -> ExecResult:
        return self.run_batch([(code, grid)])[0]

    def run_batch(
        self,
        jobs: Sequence[tuple[str, Grid]],
        on_result: Optional[Callable[[int, ExecResult], None]] = None,
    ) -> list[ExecResult]:
        out: list[ExecResult] = []
        for i, (code, grid) in enumerate(jobs):
            t0 = time.perf_counter()
            try:
                h = hash(code)
                ns = self._cache.get(h)
                if ns is None:
                    ns = {}
                    import numpy as _np  # noqa: F401

                    exec(compile(code, "<candidate>", "exec"), ns)
                    if not callable(ns.get("transform")):
                        raise ValueError("no transform()")
                    self._cache[h] = ns
                res = ns["transform"](np.asarray(grid).copy())
                arr = np.asarray(res)
                if arr.ndim != 2:
                    raise ValueError("output not 2-D")
                arr = arr.astype(np.int64)
                if arr.size > 64 * 64:
                    raise ValueError("output too large")
                er = ExecResult(True, arr, None, False, (time.perf_counter() - t0) * 1000)
            except Exception as e:  # noqa: BLE001
                er = ExecResult(False, None, f"{type(e).__name__}: {e}", False, (time.perf_counter() - t0) * 1000)
            out.append(er)
            if on_result:
                on_result(i, er)
        return out

    def close(self) -> None:
        pass

    def __enter__(self) -> "LocalExecutor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def os_name_check() -> str:  # pragma: no cover - trivial helper used in reports
    return os.name
