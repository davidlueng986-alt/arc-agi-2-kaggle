"""Isolated Python REPL for model-generated code.

Faithful to the Duck's tool contract, retargeted at ARC-AGI-2:

* **One tool** (``python``), **ephemeral state** -- a fresh OS process per call,
  so nothing leaks between turns (the Duck resets REPL state between every tool
  call; we get it for free from process-per-call).
* **Hard limits chosen for a model, not a human**: 30s wall clock per call,
  4096-char output cap, RLIMIT_CPU / RLIMIT_AS / RLIMIT_FSIZE / RLIMIT_NOFILE
  inside the child, a clean environment, a throwaway cwd, and ``-I`` (isolated:
  no cwd on ``sys.path``, no user site, env vars ignored).
* **A host round-trip primitive.**  The Duck's sandbox exposes ``action(...)``
  which suspends the snippet, sends the action to the host, and refreshes the
  preloaded variables from the reply.  Our equivalent is ``verify(fn)``: it
  suspends the snippet, hands the candidate program plus its exact train/test
  behaviour to the host (which registers it in the ranked candidate pool), and
  returns exact pass/fail with diff diagnostics.  ``finish(reason)`` is the
  matching control-flow primitive.

Security posture, stated plainly.  The Duck's authors note their REPL "is not
fully isolated".  We do better where it is cheap -- a real OS process boundary
with rlimits, an empty environment, ``CUDA_VISIBLE_DEVICES=""`` so model code
cannot touch the GPU that is serving the model, no network at rerun, and a
host-side process-group kill on timeout.  We deliberately do **not** pretend
that restricting ``__builtins__`` is a security boundary: any allowlist loose
enough to include numpy is escapable in one line, and the Duck's own ablation
found that hand-constraining the tool surface hurt.  So imports are filtered by
a **denylist** of the genuinely dangerous modules (process, filesystem, network,
interpreter internals) and everything else is available.
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
from dataclasses import dataclass, field
from typing import Any, Callable

from . import oracle as _oracle
from . import segmentation as _segmentation
from .grids import Task, task_to_payload

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_OUTPUT_CHARS = 4096
DEFAULT_VERIFY_CALL_SECONDS = 1.5
DEFAULT_VERIFY_TOTAL_SECONDS = 12.0

BLOCKED_MODULES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "ctypes",
        "multiprocessing",
        "concurrent",
        "signal",
        "importlib",
        "builtins",
        "pickle",
        "marshal",
        "shelve",
        "tempfile",
        "pathlib",
        "webbrowser",
        "http",
        "urllib",
        "requests",
        "email",
        "ftplib",
        "smtplib",
        "telnetlib",
        "pty",
        "resource",
        "threading",
        "_thread",
        "asyncio",
        "posix",
        "nt",
        "msvcrt",
        "winreg",
        "mmap",
        "fcntl",
        "termios",
        "tty",
        "sched",
        "pdb",
        "code",
        "codeop",
        "compileall",
        "py_compile",
        "runpy",
        "zipimport",
        "pkgutil",
        "distutils",
        "setuptools",
        "pip",
        "gc",
        "atexit",
        "faulthandler",
        "tracemalloc",
    }
)


@dataclass
class Verification:
    """One registered candidate program, scored by the exact oracle."""

    index: int
    source: str | None
    exact: bool
    soft_score: float
    pass_rate: float
    n_pairs: int
    summary: str
    test_outputs: list[Any] = field(default_factory=list)
    test_shapes: list[list[int]] = field(default_factory=list)
    test_errors: list[str] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    stdout: str = ""
    result: Any = None
    error: str = ""
    truncated: bool = False
    timed_out: bool = False
    verifications: list[Verification] = field(default_factory=list)
    finished: bool = False
    finish_reason: str = ""
    elapsed_s: float = 0.0
    calls: int = 0

    def render(self, max_chars: int = DEFAULT_MAX_OUTPUT_CHARS) -> str:
        """Model-facing text for this tool call (already capped by the sandbox)."""
        parts: dict[str, Any] = {}
        if self.timed_out:
            parts["error"] = self.error or "tool call timed out"
        elif self.error:
            parts["error"] = self.error
        if self.stdout.strip():
            text = self.stdout.rstrip()
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"
                self.truncated = True
            parts["stdout"] = text
        if self.result is not None:
            parts["result"] = self.result
        if self.verifications:
            parts["verify"] = [v.report for v in self.verifications]
        if self.truncated:
            parts["truncated"] = True
            parts["truncation_note"] = (
                f"Tool output was cut to ~{max_chars} chars. Print compact derived "
                "summaries (counts, coordinates, diffs) -- never whole grids."
            )
        if self.finished:
            parts["finished"] = True
            if self.finish_reason:
                parts["finish_reason"] = self.finish_reason
        if not parts:
            parts["ok"] = True
        return json.dumps(parts, indent=1, ensure_ascii=True, default=str)


_BOOTSTRAP_TEMPLATE = r'''
import builtins
import inspect
import io
import json
import linecache
import os
import sys
import traceback

try:
    import resource
except ImportError:
    resource = None

HOST_STDOUT = sys.stdout

__SEGMENTATION_SOURCE__

__ORACLE_SOURCE__

BLOCKED_MODULES = __BLOCKED_MODULES__
COLOR_CHARS = ".brgysmocd"


class _CappedStream(io.StringIO):
    """stdout sink that stops growing past ``limit`` characters."""

    def __init__(self, limit):
        super().__init__()
        self._limit = max(64, int(limit))
        self.overflow = 0

    def write(self, text):
        text = "" if text is None else str(text)
        room = self._limit - self.tell()
        if room <= 0:
            self.overflow += len(text)
            return len(text)
        if len(text) > room:
            self.overflow += len(text) - room
            text = text[:room]
        return super().write(text)


def _send(payload):
    HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    HOST_STDOUT.flush()


def _recv():
    line = sys.stdin.readline()
    if not line:
        raise EOFError("sandbox input closed")
    return json.loads(line)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name or "").split(".", 1)[0]
    if root in BLOCKED_MODULES:
        raise ImportError(
            "module %r is blocked in the REPL sandbox (process/filesystem/network/interpreter "
            "access is not available). numpy and the standard data/algorithm libraries are." % (name,)
        )
    return builtins.__import__(name, globals, locals, fromlist, level)


def _set_limits(timeout_seconds, memory_mb):
    if resource is None:
        return
    cpu_limit = max(1, int(timeout_seconds)) + 2
    wanted = (
        (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
        (getattr(resource, "RLIMIT_FSIZE", None), 2_000_000),
        (getattr(resource, "RLIMIT_NOFILE", None), 64),
        (getattr(resource, "RLIMIT_AS", None), int(memory_mb) * 1024 * 1024),
        (getattr(resource, "RLIMIT_CORE", None), 0),
    )
    for limit, value in wanted:
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, (value, value))
        except (OSError, ValueError):
            pass


def _sanitize_exception(exc):
    extracted = traceback.extract_tb(exc.__traceback__)
    user_frames = [f for f in extracted if f.filename == "<repl>"]
    lines = ["Traceback (most recent call last):"]
    for frame in user_frames[-4:] or extracted[-1:]:
        lines.append('  File "<repl>", line %s, in %s' % (frame.lineno, frame.name))
        if frame.line:
            lines.append("    " + frame.line.strip()[:200])
    lines.append("%s: %s" % (exc.__class__.__name__, str(exc)[:600]))
    return "\n".join(lines)


def _source_of(obj, fallback):
    try:
        text = inspect.getsource(obj)
        if text and text.strip():
            return text
    except Exception:
        pass
    return fallback


class Grid(list):
    """list[list[int]] with cheap views. Behaves exactly like a list of lists."""

    @property
    def shape(self):
        return (len(self), len(self[0]) if len(self) else 0)

    @property
    def rows(self):
        return len(self)

    @property
    def cols(self):
        return len(self[0]) if len(self) else 0

    @property
    def ascii(self):
        return ascii_grid(self, COLOR_CHARS)

    @property
    def csv(self):
        return "\n".join(",".join(str(v) for v in row) for row in self)

    @property
    def segmentation(self):
        return segment_grid(self, COLOR_CHARS)

    def seg(self, max_nodes=24):
        return segment_summary(self, COLOR_CHARS, max_nodes)

    def colors(self):
        seen = set()
        for row in self:
            seen.update(row)
        return sorted(seen)

    def histogram(self):
        hist = {}
        for row in self:
            for v in row:
                hist[v] = hist.get(v, 0) + 1
        return dict(sorted(hist.items()))

    def copy_grid(self):
        return Grid(Grid(row) for row in self)

    def __str__(self):
        r, c = self.shape
        return "Grid(%dx%d, colors=%s)" % (r, c, self.colors())

    __repr__ = __str__


def _as_grid(raw):
    if isinstance(raw, Grid):
        return raw
    if isinstance(raw, (list, tuple)):
        return Grid(Grid(row) if not isinstance(row, Grid) else row for row in raw)
    try:
        import numpy as _np

        if isinstance(raw, _np.ndarray):
            return Grid(Grid(int(v) for v in row) for row in raw.tolist())
    except Exception:
        pass
    raise TypeError("not a grid: %r" % (type(raw).__name__,))


class Pair:
    __slots__ = ("input", "output", "index")

    def __init__(self, index, grid_in, grid_out):
        self.index = index
        self.input = grid_in
        self.output = grid_out

    def __str__(self):
        return "Pair(%d, in=%s, out=%s)" % (self.index, self.input.shape, self.output.shape)

    __repr__ = __str__


def main():
    initial = _recv()
    code = str(initial.get("code", ""))
    timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
    memory_mb = int(initial.get("memory_mb", 3072))
    max_output_chars = int(initial.get("max_output_chars", 4096))
    verify_call_seconds = float(initial.get("verify_call_seconds", 1.5))
    verify_total_seconds = float(initial.get("verify_total_seconds", 12.0))
    sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
    if sandbox_cwd:
        os.chdir(sandbox_cwd)
    _set_limits(timeout_seconds, memory_mb)

    task = initial.get("task") or {}
    train = [
        Pair(i, _as_grid(p["input"]), _as_grid(p["output"]))
        for i, p in enumerate(task.get("train") or [])
    ]
    test = [_as_grid(g) for g in (task.get("test") or [])]

    # Make `inspect.getsource` work for defs inside the ephemeral snippet.
    lines = code.splitlines(True)
    linecache.cache["<repl>"] = (len(code), None, lines, "<repl>")

    verifications = []
    finished = {"flag": False, "reason": ""}

    def verify(fn, aug=True):
        """Exact oracle: run `fn` on every train pair and return pass/fail + diffs.

        This is the only ground truth available at inference time and it is free.
        The candidate (source, train score, test predictions) is registered with
        the host, so the best verified program becomes the answer -- you never
        have to print a grid.
        """
        source = _source_of(fn, code)
        report = verify_candidate(
            [(_plain(p.input), _plain(p.output)) for p in train],
            [_plain(g) for g in test],
            fn,
            per_call_seconds=verify_call_seconds,
            total_seconds=verify_total_seconds,
            use_augmentations=bool(aug),
        )
        payload = {"source": source, "report": report}
        _send({"type": "verify", "payload": payload})
        ack = _recv()
        shown = strip_private(report)
        shown["registered"] = bool(ack.get("registered"))
        if ack.get("reason"):
            shown["note"] = ack.get("reason")
        shown["source_chars"] = len(source or "")
        verifications.append(shown)
        return shown

    def finish(reason=""):
        """Declare you are done with this task. Saves budget; only call it when a
        verified program exists or you have nothing better to try."""
        finished["flag"] = True
        finished["reason"] = str(reason)[:200]
        _send({"type": "finish", "reason": finished["reason"]})
        _recv()
        return True

    def _plain(grid):
        return [list(row) for row in grid]

    def segment(grid):
        return segment_grid(_as_grid(grid), COLOR_CHARS)

    def show(grid, mode="ascii"):
        g = _as_grid(grid)
        return g.ascii if mode == "ascii" else g.csv

    def shapes(pairs=None):
        items = train if pairs is None else pairs
        return [(tuple(_as_grid(p.input).shape), tuple(_as_grid(p.output).shape)) for p in items]

    runtime_globals = {
        "__builtins__": builtins.__dict__,
        "__name__": "__repl__",
        "__doc__": None,
        "result": None,
        # pre-loaded task state
        "TRAIN": train,
        "TEST": test,
        "TRAIN_PAIRS": [(_plain(p.input), _plain(p.output)) for p in train],
        "TEST_INPUTS": [_plain(g) for g in test],
        "TASK_ID": str(task.get("task_id", "")),
        # helpers
        "verify": verify,
        "finish": finish,
        "segment": segment,
        "segment_grid": segment_grid,
        "segment_summary": segment_summary,
        "show": show,
        "shapes": shapes,
        "Grid": Grid,
        "Pair": Pair,
        "COLOR_CHARS": COLOR_CHARS,
        "COLOR_LEGEND": COLOR_LEGEND,
        "ascii_grid": ascii_grid,
        "augment": augment,
        "AUGMENTATIONS": AUGMENTATIONS,
        "grid_diff": grid_diff,
        "cell_accuracy": cell_accuracy,
        "normalize_grid": normalize_grid,
    }
    # Restricted import for user code only: the bootstrap above already ran with
    # the real builtins, so it can import signal/resource/inspect freely.
    user_builtins = dict(builtins.__dict__)
    user_builtins["__import__"] = _safe_import
    runtime_globals["__builtins__"] = user_builtins

    stdout = _CappedStream(max_output_chars)
    try:
        compiled = compile(code, "<repl>", "exec")
        saved_stdout = sys.stdout
        sys.stdout = stdout
        try:
            exec(compiled, runtime_globals, runtime_globals)
        finally:
            sys.stdout = saved_stdout
        _send(
            {
                "type": "final",
                "stdout": stdout.getvalue(),
                "overflow_chars": stdout.overflow,
                "result": _json_safe(runtime_globals.get("result")),
                "verifications": [
                    {k: v for k, v in item.items() if k != "test"} for item in verifications
                ],
                "finished": finished["flag"],
                "finish_reason": finished["reason"],
            }
        )
    except BaseException as exc:
        _send(
            {
                "type": "error",
                "error": _sanitize_exception(exc) if isinstance(exc, Exception) else repr(exc)[:400],
                "stdout": stdout.getvalue(),
                "overflow_chars": stdout.overflow,
                "verifications": [
                    {k: v for k, v in item.items() if k != "test"} for item in verifications
                ],
                "finished": finished["flag"],
                "finish_reason": finished["reason"],
            }
        )


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


if __name__ == "__main__":
    main()
'''


def _build_bootstrap() -> str:
    """Splice the two self-contained modules into the sandbox bootstrap.

    Both ``segmentation.py`` and ``oracle.py`` are written to be spliceable:
    standard library only, no project imports, no ``from __future__``.  This is
    exactly the Duck's trick for ``inference/utils/segmentation.py`` and it buys
    a single tested implementation of the oracle shared by sandbox, host pool
    and unit tests.
    """
    global _BOOTSTRAP_CACHE
    if _BOOTSTRAP_CACHE is None:
        _BOOTSTRAP_CACHE = (
            _BOOTSTRAP_TEMPLATE.replace(
                "__SEGMENTATION_SOURCE__", _read_module_source(_segmentation)
            )
            .replace("__ORACLE_SOURCE__", _read_module_source(_oracle))
            .replace("__BLOCKED_MODULES__", "frozenset(%r)" % (sorted(BLOCKED_MODULES),))
        )
    return _BOOTSTRAP_CACHE


_BOOTSTRAP_CACHE: str | None = None


def _read_module_source(module) -> str:
    """Verbatim module source.  The docstring becomes a harmless no-op statement."""
    import inspect as _inspect

    return _inspect.getsource(module)


def _child_env() -> dict[str, str]:
    env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": "",
        # Keep model-generated code off the GPU that is serving the model.
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    path = os.environ.get("PATH", "")
    if path:
        env["PATH"] = path
    if os.name == "nt":
        for key in ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP", "PATHEXT"):
            if key in os.environ:
                env[key] = os.environ[key]
    else:
        env["HOME"] = "/tmp"
        env["TMPDIR"] = "/tmp"
        env["LANG"] = os.environ.get("LANG", "C.UTF-8")
    return env


def _kill_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return
        except Exception:
            pass
    else:
        try:
            import signal as _signal

            os.killpg(os.getpgid(process.pid), _signal.SIGKILL)
            return
        except Exception:
            pass
    try:
        process.kill()
    except Exception:
        pass


class ReplSandbox:
    """Runs one ephemeral Python snippet against a task's preloaded state."""

    def __init__(
        self,
        task: Task,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        memory_mb: int = 3072,
        verify_call_seconds: float = DEFAULT_VERIFY_CALL_SECONDS,
        verify_total_seconds: float = DEFAULT_VERIFY_TOTAL_SECONDS,
        python_exe: str | None = None,
        on_verify: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        bootstrap: str | None = None,
    ) -> None:
        self.task = task
        self.timeout_s = float(timeout_s)
        self.max_output_chars = int(max_output_chars)
        self.memory_mb = int(memory_mb)
        self.verify_call_seconds = float(verify_call_seconds)
        self.verify_total_seconds = float(verify_total_seconds)
        self.python_exe = python_exe or sys.executable
        self._on_verify = on_verify
        self._bootstrap = bootstrap if bootstrap is not None else _build_bootstrap()
        self._payload = task_to_payload(task)

    # ------------------------------------------------------------------ public

    def run(self, code: str) -> ToolResult:
        """Execute one snippet.  Fresh process == fresh state, every call."""
        code = (code or "").strip()
        if not code:
            return ToolResult(error="python requires a non-empty `code` string.")
        try:
            compile(code, "<repl>", "exec")
        except SyntaxError as exc:
            return ToolResult(error=f"Python syntax error: {exc}")

        verifications: list[Verification] = []
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="arc26_repl_") as workdir:
            # Written to a file rather than passed with `-c`: the spliced
            # bootstrap is ~35KB and Windows caps a command line at ~32KB
            # (WinError 206).  Linux is fine either way, and `-I` still keeps the
            # script's directory off sys.path.
            script = os.path.join(workdir, "repl_bootstrap.py")
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(self._bootstrap)
            kwargs: dict[str, Any] = {}
            if os.name != "nt":
                kwargs["start_new_session"] = True
            else:
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            try:
                process = subprocess.Popen(
                    [self.python_exe, "-I", script],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=workdir,
                    env=_child_env(),
                    **kwargs,
                )
            except OSError as exc:
                return ToolResult(error=f"sandbox process could not start: {exc}")

            assert process.stdin is not None and process.stdout is not None
            lines: queue.Queue = queue.Queue()

            def _reader() -> None:
                try:
                    for raw in process.stdout:
                        lines.put(raw)
                except Exception:
                    pass
                finally:
                    lines.put(None)

            threading.Thread(target=_reader, daemon=True).start()

            self._send(
                process.stdin,
                {
                    "code": code,
                    "timeout_seconds": self.timeout_s,
                    "memory_mb": self.memory_mb,
                    "max_output_chars": self.max_output_chars,
                    "verify_call_seconds": self.verify_call_seconds,
                    "verify_total_seconds": self.verify_total_seconds,
                    "sandbox_cwd": workdir,
                    "task": self._payload,
                },
            )

            deadline = started + self.timeout_s
            finished = False
            finish_reason = ""
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    _kill_tree(process)
                    self._reap(process)
                    return ToolResult(
                        error=f"tool call timed out after {self.timeout_s:.0f}s",
                        timed_out=True,
                        verifications=verifications,
                        finished=finished,
                        finish_reason=finish_reason,
                        elapsed_s=time.perf_counter() - started,
                        calls=len(verifications),
                    )
                try:
                    line = lines.get(timeout=min(remaining, 0.2))
                except queue.Empty:
                    continue
                if line is None:
                    stderr = self._drain_stderr(process)
                    _kill_tree(process)
                    self._reap(process)
                    return ToolResult(
                        error=self._host_error(stderr),
                        verifications=verifications,
                        finished=finished,
                        finish_reason=finish_reason,
                        elapsed_s=time.perf_counter() - started,
                        calls=len(verifications),
                    )
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = str(message.get("type", ""))

                if kind == "verify":
                    verification, ack = self._handle_verify(message.get("payload") or {}, len(verifications))
                    if verification is not None:
                        verifications.append(verification)
                    self._send(process.stdin, ack)
                    continue

                if kind == "finish":
                    finished = True
                    finish_reason = str(message.get("reason", ""))[:200]
                    self._send(process.stdin, {"type": "finish_ack"})
                    continue

                if kind in ("final", "error"):
                    self._reap(process)
                    stdout_text = str(message.get("stdout", "") or "")
                    truncated = int(message.get("overflow_chars", 0) or 0) > 0
                    if truncated:
                        stdout_text += (
                            f"\n... [output capped at {self.max_output_chars} chars; "
                            f"{message.get('overflow_chars')} more were dropped]"
                        )
                    return ToolResult(
                        stdout=stdout_text,
                        result=message.get("result"),
                        error=str(message.get("error", "") or ""),
                        truncated=truncated,
                        verifications=verifications,
                        finished=bool(message.get("finished")) or finished,
                        finish_reason=str(message.get("finish_reason") or finish_reason),
                        elapsed_s=time.perf_counter() - started,
                        calls=len(verifications),
                    )

                _kill_tree(process)
                self._reap(process)
                return ToolResult(
                    error="sandbox returned an unknown message type",
                    verifications=verifications,
                    elapsed_s=time.perf_counter() - started,
                )

    # ----------------------------------------------------------------- helpers

    def _handle_verify(self, payload: dict[str, Any], index: int) -> tuple[Verification | None, dict[str, Any]]:
        report = payload.get("report") or {}
        source = payload.get("source")
        tests = report.get("test") or []
        test_outputs = [entry.get("_grid") for entry in tests]
        test_shapes = [entry.get("shape") for entry in tests if entry.get("shape")]
        test_errors = [entry.get("error") for entry in tests if entry.get("error")]
        verification = Verification(
            index=index,
            source=source,
            exact=bool(report.get("exact")),
            soft_score=float(report.get("soft_score") or 0.0),
            pass_rate=float(report.get("pass_rate") or 0.0),
            n_pairs=int(report.get("pairs") or 0),
            summary=str(report.get("summary") or ""),
            test_outputs=test_outputs,
            test_shapes=test_shapes,
            test_errors=test_errors,
            report=_oracle.strip_private(report),
        )
        ack: dict[str, Any] = {"type": "verify_ack", "registered": True}
        if self._on_verify is not None:
            try:
                extra = self._on_verify(
                    {
                        "index": index,
                        "source": source,
                        "exact": verification.exact,
                        "soft_score": verification.soft_score,
                        "pass_rate": verification.pass_rate,
                        "n_pairs": verification.n_pairs,
                        "summary": verification.summary,
                        "test_outputs": test_outputs,
                        "test_errors": test_errors,
                    }
                )
                if isinstance(extra, dict):
                    ack.update({k: v for k, v in extra.items() if k in {"registered", "reason"}})
            except Exception as exc:  # never let the host callback kill the snippet
                ack.update({"registered": False, "reason": f"host rejected candidate: {exc}"})
        return verification, ack

    @staticmethod
    def _send(handle, payload: dict[str, Any]) -> None:
        try:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            handle.flush()
        except (OSError, ValueError):
            pass

    @staticmethod
    def _drain_stderr(process: subprocess.Popen) -> str:
        if process.stderr is None:
            return ""
        try:
            return process.stderr.read() or ""
        except Exception:
            return ""

    @staticmethod
    def _reap(process: subprocess.Popen, timeout: float = 2.0) -> None:
        try:
            process.wait(timeout=timeout)
            return
        except Exception:
            pass
        _kill_tree(process)
        try:
            process.wait(timeout=timeout)
        except Exception:
            pass

    @staticmethod
    def _host_error(stderr: str) -> str:
        text = (stderr or "").strip()
        if not text:
            return "sandbox process exited unexpectedly"
        # Surface the real Python error but never the bootstrap source.
        keep = [line for line in text.splitlines() if "<repl>" in line or "Error" in line or "error" in line]
        tail = keep[-6:] if keep else text.splitlines()[-4:]
        return "\n".join(tail)[:1200]
